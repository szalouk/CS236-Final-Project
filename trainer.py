import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tbx
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torchmetrics import IS, FID, KID
from copy import deepcopy
import numpy as np

from external.diffaug import DiffAugment
policy = 'color,translation'

import external.lpips as lpips
# percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=False)

def prepare_data_for_inception(x, device):
    r"""
    Preprocess data to be feed into the Inception model.
    """

    x = F.interpolate(x, 299, mode="bicubic", align_corners=False)
    minv, maxv = float(x.min()), float(x.max())
    x.clamp_(min=minv, max=maxv).add_(-minv).div_(maxv - minv + 1e-5)
    x.mul_(255).add_(0.5).clamp_(0, 255)

    return x.to(device).to(torch.uint8)


def prepare_data_for_gan(x, nz, device):
    r"""
    Helper function to prepare inputs for model.
    """

    return (
        x.to(device),
        torch.randn((x.size(0), nz)).to(device),
    )


def compute_prob(logits):
    r"""
    Computes probability from model output.
    """

    return torch.sigmoid(logits).mean()


def hinge_loss_g(fake_preds):
    r"""
    Computes generator hinge loss.
    """

    return -fake_preds.mean()


def hinge_loss_d(real_preds, fake_preds):
    r"""
    Computes discriminator hinge loss.
    """

    return F.relu(1.0 - real_preds).mean() + F.relu(1.0 + fake_preds).mean()

def compute_loss_g(net_g, net_d, z, loss_func_g):
    r"""
    General implementation to compute generator loss.
    """

    fakes = net_g(z)
    fake_preds = net_d(fakes).view(-1)
    loss_g = loss_func_g(fake_preds)

    return loss_g, fakes, fake_preds


def compute_loss_d(net_g, net_d, reals, z, loss_func_d):
    r"""
    General implementation to compute discriminator loss.
    """

    real_preds = net_d(reals).view(-1)
    fakes = net_g(z).detach()
    fake_preds = net_d(fakes).view(-1)
    loss_d = loss_func_d(real_preds, fake_preds)

    return loss_d, fakes, real_preds, fake_preds


def train_step(net, opt, compute_loss):
    r"""
    General implementation to perform a training step.
    """

    net.train()
    loss = compute_loss()
    net.zero_grad()
    loss.backward()
    opt.step()
    # sch.step()

    return loss


def evaluate(net_g, net_d, dataloader, nz, device, samples_z=None):
    r"""
    Evaluates model and logs metrics.
    Attributes:
        net_g (Module): Torch generator model.
        net_d (Module): Torch discriminator model.
        dataloader (Dataloader): Torch evaluation set dataloader.
        nz (int): Generator input / noise dimension.
        device (Device): Torch device to perform evaluation on.
        samples_z (Tensor): Noise tensor to generate samples.
    """

    net_g.to(device).eval()
    net_d.to(device).eval()

    with torch.no_grad():

        # Initialize metrics
        is_, fid, kid, loss_gs, loss_ds, real_preds, fake_preds = (
            IS().to(device),
            FID().to(device),
            KID().to(device),
            [],
            [],
            [],
            [],
        )

        for data, _ in tqdm(dataloader, desc="Evaluating Model"):

            # Compute losses and save intermediate outputs
            reals, z = prepare_data_for_gan(data, nz, device)
            loss_d, fakes, real_pred, fake_pred = compute_loss_d(
                net_g,
                net_d,
                reals,
                z,
                hinge_loss_d,
            )
            loss_g, _, _ = compute_loss_g(
                net_g,
                net_d,
                z,
                hinge_loss_g,
            )

            # Update metrics
            loss_gs.append(loss_g)
            loss_ds.append(loss_d)
            real_preds.append(compute_prob(real_pred))
            fake_preds.append(compute_prob(fake_pred))
            reals = prepare_data_for_inception(reals, device)
            fakes = prepare_data_for_inception(fakes, device)
            is_.update(fakes)
            fid.update(reals, real=True)
            fid.update(fakes, real=False)
            kid.update(reals, real=True)
            kid.update(fakes, real=False)

        # Process metrics
        metrics = {
            "L(G)": torch.stack(loss_gs).mean().item(),
            "L(D)": torch.stack(loss_ds).mean().item(),
            "D(x)": torch.stack(real_preds).mean().item(),
            "D(G(z))": torch.stack(fake_preds).mean().item(),
            "IS": is_.compute()[0].item(),
            "FID": fid.compute().item(),
            "KID": kid.compute()[0].item(),
        }

        # Create samples
        if samples_z is not None:
            samples = net_g(samples_z)[0]
            samples = F.interpolate(samples, 256).cpu()
            samples = vutils.make_grid(samples, nrow=6, padding=4, normalize=True)

    return metrics if samples_z is None else (metrics, samples)


class Trainer:
    r"""
    Trainer performs GAN training, checkpointing and logging.
    Attributes:
        net_g (Module): Torch generator model.
        net_d (Module): Torch discriminator model.
        opt_g (Optimizer): Torch optimizer for generator.
        opt_d (Optimizer): Torch optimizer for discriminator.
        sch_g (Scheduler): Torch lr scheduler for generator.
        sch_d (Scheduler): Torch lr scheduler for discriminator.
        train_dataloader (Dataloader): Torch training set dataloader.
        eval_dataloader (Dataloader): Torch evaluation set dataloader.
        nz (int): Generator input / noise dimension.
        log_dir (str): Path to store log outputs.
        ckpt_dir (str): Path to store and load checkpoints.
        device (Device): Torch device to perform training on.
    """

    def __init__(
        self,
        net_g,
        net_d,
        opt_g,
        opt_d,
        sch_g,
        sch_d,
        train_dataloader,
        eval_dataloader,
        nz,
        log_dir,
        ckpt_dir,
        device,
    ):
        # Setup models, dataloader, optimizers
        self.net_g = net_g.to(device)
        self.net_d = net_d.to(device)
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.sch_g = sch_g
        self.sch_d = sch_d
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Setup training parameters
        self.device = device
        self.nz = nz
        self.step = 0

        # Setup checkpointing, evaluation and logging
        self.fixed_z = torch.randn((36, nz), device=device)
        self.logger = tbx.SummaryWriter(log_dir)
        self.ckpt_dir = ckpt_dir

    def _state_dict(self):
        return {
            "net_g": self.net_g.state_dict(),
            "net_d": self.net_d.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "sch_g": self.sch_g.state_dict(),
            "sch_d": self.sch_d.state_dict(),
            "step": self.step,
        }

    def _load_state_dict(self, state_dict):
        self.net_g.load_state_dict(state_dict["net_g"])
        self.net_d.load_state_dict(state_dict["net_d"])
        self.opt_g.load_state_dict(state_dict["opt_g"])
        self.opt_d.load_state_dict(state_dict["opt_d"])
        self.sch_g.load_state_dict(state_dict["sch_g"])
        self.sch_d.load_state_dict(state_dict["sch_d"])
        self.step = state_dict["step"]

    def _load_checkpoint(self):
        r"""
        Finds the last checkpoint in ckpt_dir and load states.
        """

        ckpt_paths = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pth")]
        if ckpt_paths:  # Train from scratch if no checkpoints were found
            ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
            self._load_state_dict(torch.load(ckpt_path))

    def _save_checkpoint(self):
        r"""
        Saves model, optimizer and trainer states.
        """

        ckpt_path = os.path.join(self.ckpt_dir, f"{self.step}.pth")
        torch.save(self._state_dict(), ckpt_path)

    def _log(self, metrics, samples):
        r"""
        Logs metrics and samples to Tensorboard.
        """

        for k, v in metrics.items():
            self.logger.add_scalar(k, v, self.step)
        self.logger.add_image("Samples", samples, self.step)
        self.logger.flush()

    def _train_step_g(self, z):
        r"""
        Performs a generator training step.
        """

        return train_step(
            self.net_g,
            self.opt_g,
            self.sch_g,
            lambda: compute_loss_g(
                self.net_g,
                self.net_d,
                z,
                hinge_loss_g,
            )[0],
        )

    def _train_step_d(self, reals, z):
        r"""
        Performs a discriminator training step.
        """

        return train_step(
            self.net_d,
            self.opt_d,
            self.sch_d,
            lambda: compute_loss_d(
                self.net_g,
                self.net_d,
                reals,
                z,
                hinge_loss_d,
            )[0],
        )

    def train(self, max_steps, repeat_d, eval_every, ckpt_every):
        r"""
        Performs GAN training, checkpointing and logging.
        Attributes:
            max_steps (int): Number of steps before stopping.
            repeat_d (int): Number of discriminator updates before a generator update.
            eval_every (int): Number of steps before logging to Tensorboard.
            ckpt_every (int): Number of steps before checkpointing models.
        """

        self._load_checkpoint()

        while True:
            pbar = tqdm(self.train_dataloader)
            for data, _ in pbar:

                # Training step
                reals, z = prepare_data_for_gan(data, self.nz, self.device)
                loss_d = self._train_step_d(reals, z)
                if self.step % repeat_d == 0:
                    loss_g = self._train_step_g(z)

                pbar.set_description(
                    f"L(G):{loss_g.item():.2f}|L(D):{loss_d.item():.2f}|{self.step}/{max_steps}"
                )

                if self.step != 0 and self.step % eval_every == 0:
                    self._log(
                        *evaluate(
                            self.net_g,
                            self.net_d,
                            self.eval_dataloader,
                            self.nz,
                            self.device,
                            samples_z=self.fixed_z,
                        )
                    )

                if self.step != 0 and self.step % ckpt_every == 0:
                    self._save_checkpoint()

                self.step += 1
                if self.step > max_steps:
                    return


# FastGAN
class ConditionalContrastiveLoss(torch.nn.Module):
    #https://arxiv.org/pdf/2006.12681v3.pdf
    # Code from: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
    def __init__(self, num_classes, temperature, master_rank):
        super(ConditionalContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.master_rank = master_rank
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed, proxy, label):
        sim_matrix = self.calculate_similarity_matrix(embed, embed)
        sim_matrix = torch.exp(self._remove_diag(sim_matrix) / self.temperature)
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])
        sim_pos_only = neg_removal_mask * sim_matrix

        emb2proxy = torch.exp(self.cosine_similarity(embed, proxy) / self.temperature)

        numerator = emb2proxy + sim_pos_only.sum(dim=1)
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)
        return -torch.log(numerator / denomerator).mean()

def fastgan_loss_g(fake_preds):
    r"""
    Computes fastgan generator loss.
    """

    return hinge_loss_g(fake_preds)


def fastgan_loss_d(real_preds, fake_preds):
    r"""
    Computes FastGAN discriminator loss.
    """
    return F.relu(  torch.rand_like(real_preds) * 0.2 + 0.8 -  real_preds).mean() + \
           F.relu(  torch.rand_like(fake_preds) * 0.2 + 0.8 + fake_preds).mean()

def compute_FastGAN_loss_g(net_g, net_d, z, loss_func_g, labels=None):
    r"""
    General implementation to compute FastGAN generator loss.
    """

    if labels is not None:
        fakes = net_g(z, labels)
    else:
        fakes = net_g(z)
    fakes = [DiffAugment(fake, policy=policy) for fake in fakes]
    
    if labels is not None:
        fake_preds, embded, proxy = net_d(fakes, 'fake', labels)
        fake_preds = fake_preds.view(-1)
    else:
        fake_preds = net_d(fakes, 'fake').view(-1)
    loss_g = loss_func_g(fake_preds)

    return loss_g, fakes, fake_preds

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def compute_FastGAN_loss_d(net_g, net_d, reals, z, loss_func_d, percept, labels=None, cond_loss=None):
    r"""
    General implementation to compute discriminator loss.
    """
    reals = DiffAugment(reals, policy=policy)
    if labels is not None:
        real_preds, [rec_all, rec_small, rec_part], part, embed, proxy = net_d(reals, 'real', labels)
        cond_loss_real = cond_loss(embed, proxy, labels)
    else:
        real_preds, [rec_all, rec_small, rec_part], part = net_d(reals, 'real')
    real_preds = real_preds.view(-1)
    if labels is not None:
        fakes = net_g(z, labels)
    else:
        fakes = net_g(z)
    fakes = [DiffAugment(fake, policy=policy).detach() for fake in fakes]
    if labels is not None:
        fake_preds, embed, proxy = net_d(fakes, 'fake', labels)
        fake_preds = fake_preds.view(-1)
        cond_loss_fake = cond_loss(embed, proxy, labels)
    else:
        fake_preds = net_d(fakes, 'fake').view(-1)
    loss_d = loss_func_d(real_preds, fake_preds) + \
        percept( rec_all, F.interpolate(reals, rec_all.shape[2]) ).sum() + \
        percept( rec_small, F.interpolate(reals, rec_small.shape[2]) ).sum() + \
        percept( rec_part, F.interpolate(crop_image_by_part(reals, part), rec_part.shape[2])).sum()   
    
    if labels is not None:
        loss_d += cond_loss_real + cond_loss_fake

    return loss_d, fakes, real_preds, fake_preds

class FastGANTrainer:
    r"""
    Trainer performs GAN training, checkpointing and logging.
    Attributes:
        net_g (Module): Torch generator model.
        net_d (Module): Torch discriminator model.
        opt_g (Optimizer): Torch optimizer for generator.
        opt_d (Optimizer): Torch optimizer for discriminator.
        sch_g (Scheduler): Torch lr scheduler for generator.
        sch_d (Scheduler): Torch lr scheduler for discriminator.
        train_dataloader (Dataloader): Torch training set dataloader.
        eval_dataloader (Dataloader): Torch evaluation set dataloader.
        nz (int): Generator input / noise dimension.
        log_dir (str): Path to store log outputs.
        ckpt_dir (str): Path to store and load checkpoints.
        device (Device): Torch device to perform training on.
    """

    def __init__(
        self,
        net_g,
        net_d,
        opt_g,
        opt_d,
        train_dataloader,
        eval_dataloader,
        nz,
        log_dir,
        ckpt_dir,
        device,
        num_classes=4,
        cond=False
    ):
        # Setup models, dataloader, optimizers
        self.net_g = net_g.to(device)
        self.net_d = net_d.to(device)
        self.g_ema = deepcopy(list(p.data for p in self.net_g.parameters()))

        self.opt_g = opt_g
        self.opt_d = opt_d
        # self.sch_g = sch_g
        # self.sch_d = sch_d
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Setup training parameters
        self.device = device
        self.nz = nz
        self.step = 0

        # Setup checkpointing, evaluation and logging
        self.fixed_z = torch.randn((12, nz), device=device)
        self.logger = tbx.SummaryWriter(log_dir)
        self.ckpt_dir = ckpt_dir

        # For FastGAN, set up perceptual distance function
        self.percept_fn = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=torch.cuda.is_available())
        print(f'cond = {cond}')
        self.cond = cond
        self.cond_loss = ConditionalContrastiveLoss(num_classes=num_classes, temperature=1.0, master_rank=device)
    def _state_dict(self):
        return {
            "net_g": self.net_g.state_dict(),
            "net_d": self.net_d.state_dict(),
            "g_ema": self.g_ema,
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            # "sch_g": self.sch_g.state_dict(),
            # "sch_d": self.sch_d.state_dict(),
            "step": self.step,
        }

    def _load_state_dict(self, state_dict):
        self.net_g.load_state_dict(state_dict["net_g"])
        self.net_d.load_state_dict(state_dict["net_d"])
        self.g_ema = state_dict["g_ema"]
        self.opt_g.load_state_dict(state_dict["opt_g"])
        self.opt_d.load_state_dict(state_dict["opt_d"])
        # self.sch_g.load_state_dict(state_dict["sch_g"])
        # self.sch_d.load_state_dict(state_dict["sch_d"])
        self.step = state_dict["step"]

    def _load_checkpoint(self):
        r"""
        Finds the last checkpoint in ckpt_dir and load states.
        """

        ckpt_paths = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pth")]
        if ckpt_paths:  # Train from scratch if no checkpoints were found
            ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
            self._load_state_dict(torch.load(ckpt_path))

    def _save_checkpoint(self):
        r"""
        Saves model, optimizer and trainer states.
        """

        ckpt_path = os.path.join(self.ckpt_dir, f"{self.step}.pth")
        torch.save(self._state_dict(), ckpt_path)

    def _log(self, metrics, samples):
        r"""
        Logs metrics and samples to Tensorboard.
        """

        for k, v in metrics.items():
            self.logger.add_scalar(k, v, self.step)
        self.logger.add_image("Samples", samples, self.step)
        self.logger.flush()

    def _train_step_g(self, z, labels=None):
        r"""
        Performs a generator training step.
        """

        return train_step(
            self.net_g,
            self.opt_g,
            # self.sch_g,
            lambda: compute_FastGAN_loss_g(
                self.net_g,
                self.net_d,
                z,
                fastgan_loss_g,
                labels
            )[0],
        )

    def _train_step_d(self, reals, z, labels=None):
        r"""
        Performs a discriminator training step.
        """

        return train_step(
            self.net_d,
            self.opt_d,
            # self.sch_d,
            lambda: compute_FastGAN_loss_d(
                self.net_g,
                self.net_d,
                reals,
                z,
                fastgan_loss_d,
                self.percept_fn,
                labels,
                self.cond_loss
            )[0],
        )

    def train(self, max_steps, repeat_d, eval_every, ckpt_every):
        r"""
        Performs GAN training, checkpointing and logging.
        Attributes:
            max_steps (int): Number of steps before stopping.
            repeat_d (int): Number of discriminator updates before a generator update.
            eval_every (int): Number of steps before logging to Tensorboard.
            ckpt_every (int): Number of steps before checkpointing models.
        """

        self._load_checkpoint()

        while True:
            pbar = tqdm(self.train_dataloader)
            for data, labels in pbar:
                
                if not self.cond:
                    labels = None

                # Training step
                reals, z = prepare_data_for_gan(data, self.nz, self.device)
                loss_d = self._train_step_d(reals, z, labels)
                if self.step % repeat_d == 0:
                    loss_g = self._train_step_g(z, labels)
                    for p, avg_p in zip(self.net_g.parameters(), self.g_ema):
                        avg_p.mul_(0.999).add_(0.001 * p.data)

                pbar.set_description(
                    f"L(G):{loss_g.item():.2f}|L(D):{loss_d.item():.2f}|{self.step}/{max_steps}"
                )

                if self.step != 0 and self.step % eval_every == 0:
                    # Load EMA of net_g
                    backup_g_params = deepcopy(list(p.data for p in self.net_g.parameters()))
                    for p, new_p in zip(self.net_g.parameters(), self.g_ema):
                        p.data.copy_(new_p)
                    self._log(
                        *evaluate_FastGAN(
                            self.net_g,
                            self.net_d,
                            self.eval_dataloader,
                            self.nz,
                            self.device,
                            samples_z=self.fixed_z,
                            percept_fn=self.percept_fn,
                            cond_fn=self.cond_loss,
                            cond=self.cond
                        )
                    )
                    # Restore original net_g params
                    for p, new_p in zip(self.net_g.parameters(), backup_g_params):
                        p.data.copy_(new_p)

                if self.step != 0 and self.step % ckpt_every == 0:
                    self._save_checkpoint()

                self.step += 1
                if self.step > max_steps:
                    return

def evaluate_FastGAN(net_g, net_d, dataloader, nz, device, samples_z=None, percept_fn=None, cond_fn=None, cond=False):
    r"""
    Evaluates model and logs metrics.
    Attributes:
        net_g (Module): Torch generator model.
        net_d (Module): Torch discriminator model.
        dataloader (Dataloader): Torch evaluation set dataloader.
        nz (int): Generator input / noise dimension.
        device (Device): Torch device to perform evaluation on.
        samples_z (Tensor): Noise tensor to generate samples.
    """

    net_g.to(device).eval()
    net_d.to(device).eval()

    with torch.no_grad():

        # Initialize metrics
        is_, fid, kid, loss_gs, loss_ds, real_preds, fake_preds = (
            IS().to(device),
            FID().to(device),
            KID().to(device),
            [],
            [],
            [],
            [],
        )

        for data, labels in tqdm(dataloader, desc="Evaluating Model"):
            
            if not cond:
                labels = None

            # Compute losses and save intermediate outputs
            reals, z = prepare_data_for_gan(data, nz, device)
            loss_d, fakes, real_pred, fake_pred = compute_FastGAN_loss_d(
                net_g,
                net_d,
                reals,
                z,
                fastgan_loss_d,
                percept_fn,
                labels,
                cond_fn
            )
            loss_g, _, _ = compute_FastGAN_loss_g(
                net_g,
                net_d,
                z,
                fastgan_loss_g,
                labels
            )

            # Update metrics
            loss_gs.append(loss_g)
            loss_ds.append(loss_d)
            real_preds.append(compute_prob(real_pred))
            fake_preds.append(compute_prob(fake_pred))
            reals = prepare_data_for_inception(reals, device)
            fakes = prepare_data_for_inception(fakes[0], device)
            is_.update(fakes)
            fid.update(reals, real=True)
            fid.update(fakes, real=False)
            kid.update(reals, real=True)
            kid.update(fakes, real=False)

        # Process metrics
        metrics = {
            "L(G)": torch.stack(loss_gs).mean().item(),
            "L(D)": torch.stack(loss_ds).mean().item(),
            "D(x)": torch.stack(real_preds).mean().item(),
            "D(G(z))": torch.stack(fake_preds).mean().item(),
            "IS": is_.compute()[0].item(),
            "FID": fid.compute().item(),
            "KID": kid.compute()[0].item(),
        }

        # Create samples
        if samples_z is not None:
            if cond:
                y = torch.randint(0, 4, (samples_z.size(0), 1))
                samples = net_g(samples_z, y)[0]
            else:
                samples = net_g(samples_z)[0]
            samples = F.interpolate(samples, 256).cpu()
            samples = vutils.make_grid(samples, nrow=6, padding=4, normalize=True)

    return metrics if samples_z is None else (metrics, samples)
