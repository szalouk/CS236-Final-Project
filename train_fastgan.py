import os
import pprint
import argparse

import torch
import torch.optim as optim

import util
from model import *
from trainer import Trainer

import os
import pprint
import argparse

import torch
import torch.optim as optim

import util
from model import *
from trainer import Trainer, FastGANTrainer


def parse_args():
    r"""
    Parses command line arguments.
    """

    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(root_dir, "out"),
        help=(
            "Path to output directory. "
            "A new one will be created if the directory does not exist."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help=(
            "Name of the current experiment."
            "Checkpoints will be stored in '{out_dir}/{name}/ckpt/'. "
            "Logs will be stored in '{out_dir}/{name}/log/'. "
            "If there are existing checkpoints in '{out_dir}/{name}/ckpt/', "
            "training will resume from the last checkpoint."
        ),
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help=(
            "Resumes training using the last checkpoint in '{out_dir}/{name}/ckpt/' if set. "
            "Throws error if '{out_dir}/{name}/' already exists by default."
        ),
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help=(
            "Overwrite last run with same name if it exists."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Manual seed for reproducibility."
    )
    parser.add_argument(
        "--im_size",
        type=int,
        default=256,
        help=(
            "Images are resized to this resolution. "
            "Models are automatically selected based on resolution."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Minibatch size used during training.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=50000, help="Number of steps to train for."
    )
    parser.add_argument(
        "--repeat_d",
        type=int,
        default=1,
        help="Number of discriminator updates before a generator update.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=500,
        help="Number of steps between model evaluation.",
    )
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=5000,
        help="Number of steps between checkpointing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to train on.",
    )
    parser.add_argument(
        "--cond",
        default=False,
        action="store_true",
        help=(
            "Use conditional GAN model."
        ),
    )

    return parser.parse_args()

def train(args):
    r"""
    Configures and trains model.
    """

    # Print command line arguments and architectures
    pprint.pprint(vars(args))

    # Setup dataset
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory 'args.data_dir' is not found.")

    # Check existing experiment
    exp_dir = os.path.join(args.out_dir, args.name)
    if os.path.exists(exp_dir) and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Directory '{exp_dir}' already exists. "
            "Set '--resume' if you wish to resume training or "
            "change '--name' if you wish to start a new experiment."
        )

    # Setup output directories
    log_dir = os.path.join(exp_dir, "log")
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    for d in [args.out_dir, exp_dir, log_dir, ckpt_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    # Fixed seed
    torch.manual_seed(args.seed)

    # Set parameters
    ndf, ngf, nz, lr, betas, eval_size, num_workers = (64, 64, 256, 2e-6, (0.1, 0.999), 1000, 4)

    if args.cond:
        net_g = CondFastGAN_Generator(num_classes=4, ngf=ngf, nz=nz, im_size=args.im_size)
        net_d = CondFastGAN_Discriminator(num_classes=4, ndf=ndf, im_size=args.im_size)
    else:
        net_g = FastGAN_Generator(ngf=ngf, nz=nz, im_size=args.im_size)
        net_d = FastGAN_Discriminator(ndf=ndf, im_size=args.im_size)

    net_g.apply(weights_init)
    net_d.apply(weights_init)

    # Configure optimizers
    opt_g = optim.Adam(net_g.parameters(), lr, betas)
    opt_d = optim.Adam(net_d.parameters(), lr, betas)

    # Configure schedulers
    # sch_g = optim.lr_scheduler.LambdaLR(
    #     opt_g, lr_lambda=lambda s: 1.0 - ((s * args.repeat_d) / args.max_steps)
    # )
    # sch_d = optim.lr_scheduler.LambdaLR(
    #     opt_d, lr_lambda=lambda s: 1.0 - (s / args.max_steps)
    # )

    # Configure dataloaders
    train_dataloader, eval_dataloader = util.get_dataloaders(
        args.data_dir, args.im_size, args.batch_size, eval_size, num_workers
    )
    
    # Configure trainer
    trainer = FastGANTrainer(
        net_g,
        net_d,
        opt_g,
        opt_d,
        # sch_g,
        # sch_d,
        train_dataloader,
        eval_dataloader,
        nz,
        log_dir,
        ckpt_dir,
        torch.device(args.device),
        4,
        args.cond
    )

    # Train model
    trainer.train(args.max_steps, args.repeat_d, args.eval_every, args.ckpt_every)

    # z = torch.Tensor(args.batch_size, nz).normal_(0, 1)

    # fake_images = net_g(z)
    # fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

    # print(fake_images[0].shape)
    # print(fake_images[1].shape)

    # pred = net_d(fake_images, label='fake')
    # loss = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
    # print(pred.shape)
    # print(loss)

    # loss.backward()

if __name__ == "__main__":
    train(parse_args())