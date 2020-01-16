import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(
        self, in_channels, channels=(32, 32, 64, 64), n_latent=20, n_hidden=256, size=64
    ):
        super().__init__()

        self.size = size

        in_ch = in_channels

        layers = []

        for ch in channels:
            layers.append(nn.Conv2d(in_ch, ch, 4, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_ch = ch

        self.conv = nn.Sequential(*layers)

        out_size = size // (len(channels) ** 2)

        self.linear = nn.Sequential(
            nn.Linear(out_size ** 2 * channels[-1], n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_latent * 2),
        )

    def sample(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mean + std * eps

    def forward(self, input):
        if input.shape[2] != self.size:
            input = F.interpolate(
                input, size=(self.size, self.size), mode='bilinear', align_corners=False
            )

        out = self.conv(input)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        mean, logvar = out.chunk(2, dim=1)

        return self.sample(mean, logvar), mean, logvar


class Decoder(nn.Module):
    def __init__(
        self, out_channels, channels=(64, 32, 32), n_latent=20, n_hidden=256, size=64
    ):
        super().__init__()

        start_size = size // (2 ** (len(channels) + 1))

        self.start_size = start_size

        self.linear = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, (start_size ** 2) * channels[0]),
            nn.ReLU(),
        )

        layers = []
        in_ch = channels[0]

        for ch in channels:
            layers.append(nn.ConvTranspose2d(in_ch, ch, 4, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_ch = ch

        layers.append(nn.ConvTranspose2d(in_ch, out_channels, 4, stride=2, padding=1))

        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        out = self.linear(input)
        out = out.view(out.shape[0], -1, self.start_size, self.start_size)
        out = self.conv(out)

        return out


class VAE(nn.Module):
    def __init__(
        self,
        in_channels,
        enc_channels=(32, 32, 64, 64),
        dec_channels=(64, 32, 32),
        n_latent=20,
        n_hidden=256,
        size=64,
    ):
        super().__init__()

        self.enc = Encoder(in_channels, enc_channels, n_latent, n_hidden, size)
        self.dec = Decoder(in_channels, dec_channels, n_latent, n_hidden, size)

    def forward(self, input, sample=True):
        latent, mean, logvar = self.enc(input)

        if not sample:
            latent = mean

        out = self.dec(latent)

        return out, mean, logvar
