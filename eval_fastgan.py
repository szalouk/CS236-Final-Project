import os
import pprint
import argparse

from tqdm import tqdm
import torch
from torchmetrics.image.fid import NoTrainInceptionV3

import util
from model import *
from trainer import evaluate_FastGAN, prepare_data_for_gan, prepare_data_for_inception, ConditionalContrastiveLoss
import external.lpips as lpips

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
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--im_size",
        type=int,
        required=True,
        help=(
            "Images are resized to this resolution. "
            "Models are automatically selected based on resolution."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size used during evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to evaluate on.",
    )
    parser.add_argument(
        "--submit",
        default=False,
        action="store_true",
        help="Generate Inception embeddings used for leaderboard submission.",
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


def generate_submission(net_g, dataloader, nz, device, path="submission.pth"):
    r"""
    Generates Inception embeddings for leaderboard submission.
    """

    net_g.to(device).eval()
    inception = NoTrainInceptionV3(
        name="inception-v3-compat", features_list=["2048"]
    ).to(device)

    with torch.no_grad():
        real_embs, fake_embs = [], []
        for data, _ in tqdm(dataloader, desc="Generating Submission"):
            reals, z = prepare_data_for_gan(data, nz, device)
            fakes = net_g(z)[0]
            reals = inception(prepare_data_for_inception(reals, device))
            fakes = inception(prepare_data_for_inception(fakes, device))
            real_embs.append(reals)
            fake_embs.append(fakes)
        real_embs = torch.cat(real_embs)
        fake_embs = torch.cat(fake_embs)
        embs = torch.stack((real_embs, fake_embs)).permute(1, 0, 2).cpu()

    torch.save(embs, path)


def eval(args):
    r"""
    Evaluates specified checkpoint.
    """

    # Set parameters
    ndf, ngf, nz, eval_size, num_workers = (
        64,
        64,
        256,
        4000 if args.submit else 10000,
        4,
    )
    
    if args.cond:
        net_g = CondFastGAN_Generator(num_classes=4, ngf=ngf, nz=nz, im_size=args.im_size)
        net_d = CondFastGAN_Discriminator(num_classes=4, ndf=ndf, im_size=args.im_size)
    else:
        net_g = FastGAN_Generator(ngf=ngf, nz=nz, im_size=args.im_size)
        net_d = FastGAN_Discriminator(ndf=ndf, im_size=args.im_size)

    # Loads checkpoint
    state_dict = torch.load(args.ckpt_path)
    net_g.load_state_dict(state_dict["net_g"])
    net_d.load_state_dict(state_dict["net_d"])

    # Configures eval dataloader
    _, eval_dataloader = util.get_dataloaders(
        args.data_dir, args.im_size, args.batch_size, eval_size, num_workers
    )

    if args.submit:
        # Generate leaderboard submission
        generate_submission(net_g, eval_dataloader, nz, args.device)

    else:
        # Evaluate models
        cond_fn = ConditionalContrastiveLoss(num_classes=4, temperature=1.0, master_rank=args.device)
        percept_fn = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=torch.cuda.is_available())
        metrics = evaluate_FastGAN(net_g, net_d, eval_dataloader, nz, args.device, percept_fn=percept_fn, cond_fn=cond_fn, cond=args.cond)
        pprint.pprint(metrics)


if __name__ == "__main__":
    eval(parse_args())
