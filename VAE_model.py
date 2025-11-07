import torch
import torch.nn as nn
from torch.nn import functional as F
import math



class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(16, channels)
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        n, c, h, w = x.shape
        x = self.groupnorm(x)
        x = x.view(n, c, h * w).transpose(1, 2)
        x, _ = self.attention(x, x, x)
        x = x.transpose(1, 2).view(n, c, h, w)
        return x + residue


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(16, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(16, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)


class VAE_Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, base_channel=16, downsample_factor=4):
        super().__init__()
        assert math.log2(downsample_factor).is_integer(), "downsample_factor must be a power of 2"
        num_downsamples = int(math.log2(downsample_factor))
        self.latent_channels = latent_channels

        layers = [nn.Conv2d(in_channels, base_channel, kernel_size=3, padding=1)]

        current_channels = base_channel
        for i in range(num_downsamples):
            out_channels = base_channel * (2 ** (i + 1))
            layers.extend([
                VAE_ResidualBlock(current_channels, out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=0)
            ])
            current_channels = out_channels

        layers.extend([
            VAE_ResidualBlock(current_channels, current_channels),
            VAE_AttentionBlock(current_channels),
            nn.GroupNorm(16, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, 2 * self.latent_channels, kernel_size=3, padding=1),
            nn.Conv2d(2 * self.latent_channels, 2 * self.latent_channels, kernel_size=1, padding=0)
        ])
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for module in self.layers:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        return mean, log_variance


class VAE_Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_channels=4, base_channel=16, downsample_factor=4):
        super().__init__()
        assert math.log2(downsample_factor).is_integer(), "downsample_factor must be a power of 2"
        num_upsamples = int(math.log2(downsample_factor))
        bottleneck_channels = base_channel * (2 ** num_upsamples)

        layers = [
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1, padding=0),
            nn.Conv2d(latent_channels, bottleneck_channels, kernel_size=3, padding=1),
            VAE_AttentionBlock(bottleneck_channels),
            VAE_ResidualBlock(bottleneck_channels, bottleneck_channels)
        ]

        current_channels = bottleneck_channels
        for i in range(num_upsamples):
            out_channels_loop = base_channel * (2 ** (num_upsamples - i - 1))
            layers.extend([
                nn.Upsample(scale_factor=2),
                nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
                VAE_ResidualBlock(current_channels, out_channels_loop)
            ])
            current_channels = out_channels_loop

        layers.extend([
            nn.GroupNorm(16, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)
        ])
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 0.18215
        for module in self.layers:
            x = module(x)
        return x


class VAE(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_channels=4, base_channel=16, downsample_factor=4):
        super().__init__()
        self.encoder = VAE_Encoder(in_channels, latent_channels, base_channel, downsample_factor)
        self.decoder = VAE_Decoder(out_channels, latent_channels, base_channel, downsample_factor)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encoder(x)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        epsilon = torch.randn_like(stdev)
        z = mean + stdev * epsilon
        scaled_z = z * 0.18215
        reconstructed_x = self.decoder(scaled_z)
        return reconstructed_x, mean, log_variance