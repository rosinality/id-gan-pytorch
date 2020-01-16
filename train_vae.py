import argparse
import os

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb

except ImportError:
    wandb = None

from vae import VAE
from scheduler import cycle_scheduler

from stylegan2.dataset import MultiResolutionDataset
from stylegan2.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def recon_loss(out, target):
    loss = F.mse_loss(out, target, reduction='sum')
    loss = loss / target.shape[0]

    return loss


def kl_loss(mean, logvar):
    kl_div = -0.5 * (1 + logvar - mean.pow(2) - torch.exp(logvar))
    kl_div = kl_div.sum(1).mean(0)

    return kl_div


@torch.no_grad()
def valid(args, epoch, loader, model, device):
    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader

    model.eval()

    recon_total = 0
    kl_total = 0
    n_imgs = 0

    for i, img in enumerate(pbar):
        img = img.to(device)

        out, mean, logvar = model(img, sample=False)
        recon = recon_loss(out, img)
        kl = kl_loss(mean, logvar)

        loss_dict = {'recon': recon, 'kl': kl}
        loss_reduced = reduce_loss_dict(loss_dict)

        if get_rank() == 0:
            batch = img.shape[0]
            recon_total += loss_reduced['recon'] * batch
            kl_total += loss_reduced['kl'] * batch
            n_imgs += batch
            recon = recon_total / n_imgs
            kl = kl_total / n_imgs

            pbar.set_description(
                f'valid; epoch: {epoch}; recon: {recon.item():.2f}; kl: {kl.item():.2f}'
            )

            if i == 0:
                utils.save_image(
                    torch.cat([img, out], 0),
                    f'sample_vae/{str(epoch).zfill(2)}.png',
                    nrow=8,
                    normalize=True,
                    range=(-1, 1),
                )

    if get_rank() == 0:
        if wandb and args.wandb:
            wandb.log(
                {
                    'Valid/Reconstruction': recon.item(),
                    'Valid/KL Divergence': kl.item(),
                },
                step=epoch,
            )


def train(args, epoch, loader, model, optimizer, scheduler, device):
    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader

    model.train()

    for img in pbar:
        img = img.to(device)

        out, mean, logvar = model(img)
        recon = recon_loss(out, img)
        kl = kl_loss(mean, logvar)
        loss = recon + args.beta * kl

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_dict = {'recon': recon, 'kl': kl}
        loss_reduced = reduce_loss_dict(loss_dict)

        if get_rank() == 0:
            recon = loss_reduced['recon']
            kl = loss_reduced['kl']
            lr = optimizer.param_groups[0]['lr']

            pbar.set_description(
                f'train; epoch: {epoch}; recon: {recon.item():.2f}; kl: {kl.item():.2f}; lr: {lr:.5f}'
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        'Train/Reconstruction': recon.item(),
                        'Train/KL Divergence': kl.item(),
                    }
                )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_size', type=int, default=128)
    parser.add_argument('--input_size', type=int, default=64)
    parser.add_argument('--n_latent', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--beta', type=float, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_train', type=int, default=60000)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--warmup', type=float, default=0.05)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('path', metavar='PATH', type=str)

    args = parser.parse_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    model = VAE(3, n_latent=args.n_latent, size=args.input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.Resize(args.input_size),
            transforms.CenterCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.data_size)
    train_set = data.Subset(dataset, list(range(args.n_train)))
    valid_set = data.Subset(dataset, list(range(args.n_train, len(dataset))))

    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch,
        sampler=data_sampler(train_set, shuffle=True, distributed=args.distributed),
    )

    valid_loader = data.DataLoader(
        valid_set,
        batch_size=args.batch,
        sampler=data_sampler(valid_set, shuffle=True, distributed=args.distributed),
    )

    if args.scheduler:
        scheduler = cycle_scheduler(
            optimizer, args.lr, len(train_loader) * args.epoch, warmup=args.warmup
        )

    else:
        scheduler = None

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project='id-gan')

    for i in range(args.epoch):
        train(args, i + 1, train_loader, model, optimizer, scheduler, device)
        valid(args, i + 1, valid_loader, model, device)

        if args.distributed:
            model_module = model.module

        else:
            model_module = model

        if get_rank() == 0:
            torch.save(
                {
                    'vae': model_module.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': args,
                },
                f'checkpoint/vae-{str(i + 1).zfill(2)}.pt',
            )

