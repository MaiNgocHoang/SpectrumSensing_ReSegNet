import numpy as np
from tqdm import tqdm
import os
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics import JaccardIndex
from copy import deepcopy
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from SUR_loss import *
from VAE_model import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = sorted(
            [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])
        self.label_paths = sorted(
            [os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir) if lbl.endswith(('.png', '.jpg', '.jpeg'))])
        # Map RGB colors to class indices
        self.class_colors = {
            (160, 160, 160): 0,  # Class 0
            (80, 80, 80): 1,  # Class 1
            (255, 255, 255): 2,  # Class 2
            (0, 0, 0): 3,  # Class 3 (Background)
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        label_rgb = cv2.cvtColor(cv2.imread(self.label_paths[idx]), cv2.COLOR_BGR2RGB)

        # Keep a copy of the clean RGB mask for VAE input
        clean_mask = label_rgb.copy()

        # Convert RGB label to indexed class mask
        label_mask_indexed = np.zeros(label_rgb.shape[:2], dtype=np.uint8)
        for rgb, class_idx in self.class_colors.items():
            label_mask_indexed[np.all(label_rgb == rgb, axis=-1)] = class_idx

        if self.transform:
            image = self.transform(image)
            clean_mask = self.transform(clean_mask)
            label_mask_indexed = torch.from_numpy(label_mask_indexed).long()

        return image, label_mask_indexed, clean_mask


class ParameterInferenceNetwork(nn.Module):
    """
    New PIN version: uses a small gating network to predict spatial importance,
    then performs a weighted sum.
    """

    def __init__(self, in_channels: int, theta_dim: int, hidden_mlp_dim: int = 128):
        super().__init__()

        # 1. Gating Network: A lightweight Conv network to create an attention map
        # Input: [B, C, H, W] -> Output: [B, 1, H, W]
        self.gating_network = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        )

        # 2. Final MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_mlp_dim),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dim, theta_dim)
        )

    def forward(self, condition_feature_map: torch.Tensor) -> torch.Tensor:
        b, c, h, w = condition_feature_map.shape

        # --- Step 1: Create and normalize attention weights ---
        # attention_logits shape: [B, 1, H, W]
        attention_logits = self.gating_network(condition_feature_map)

        # Flatten spatial dimensions to apply softmax
        # [B, 1, H, W] -> [B, H*W]
        attention_logits_flat = attention_logits.view(b, -1)

        # Apply softmax to turn logits into probability weights
        # attention_weights_flat shape: [B, H*W], each row sums to 1
        attention_weights_flat = F.softmax(attention_logits_flat, dim=-1)

        # Reshape weights back to a 2D map
        # [B, H*W] -> [B, 1, H, W]
        attention_map = attention_weights_flat.view(b, 1, h, w)

        # --- Step 2: Calculate weighted sum ---
        # Use broadcasting to multiply the original feature map by the attention map
        # (B, C, H, W) * (B, 1, H, W) -> (B, C, H, W)
        weighted_features = condition_feature_map * attention_map

        # Sum all feature vectors across the spatial dimensions
        # Result is pooled_vector with shape [B, C]
        pooled_vector = weighted_features.sum(dim=[2, 3])

        # --- Step 3: Infer theta ---
        theta_predicted = self.mlp(pooled_vector)
        return theta_predicted


class FiLMLayer(nn.Module):
    """
    This layer receives the theta vector and uses it to modulate (scale and shift)
    a feature map, allowing the PGN to "listen" to instructions from the PIN.
    """

    def __init__(self, channels: int, theta_dim: int):
        super().__init__()
        self.channels = channels
        # Generates 2*channels parameters (gamma and beta) from theta
        self.param_generator = nn.Linear(theta_dim, channels * 2)

    def forward(self, feature_map: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        params = self.param_generator(theta)

        # Split params into gamma (scale) and beta (shift)
        gamma = params[:, :self.channels].unsqueeze(-1).unsqueeze(-1)
        beta = params[:, self.channels:].unsqueeze(-1).unsqueeze(-1)

        # Apply FiLM: y = gamma * x + beta
        return gamma * feature_map + beta


class ParameterizedGenerationNetwork(nn.Module):
    """
    This is the PGN (formerly ConsistencyModel), upgraded to receive
    and use the "design" vector theta.
    """

    def __init__(self, in_channels, out_channels, base_channels, condition_dim, emb_channels=128, theta_dim=128):
        super().__init__()
        # Consistency model parameters
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.sigma_data = 0.5
        self.rho = 7

        self.modulation_scale = nn.Parameter(torch.tensor(0.01))
        noise_emb_dim = 32

        # Noise and condition mapping
        self.map_noise = PositionalEmbedding(num_channels=noise_emb_dim)
        self.map_label = nn.Linear(in_features=condition_dim, out_features=noise_emb_dim)
        self.map_layer0 = nn.Linear(in_features=noise_emb_dim, out_features=emb_channels)
        self.map_layer1 = nn.Linear(in_features=emb_channels, out_features=emb_channels)

        # The core U-Net is now a UNET_FiLM, upgraded to receive theta
        self.F_theta = UNET_FiLM(in_channels, out_channels, base_channels, emb_channels, theta_dim)

    # --- Consistency Model helper functions ---
    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return self.sigma_data * (sigma - self.sigma_min) / (
                self.sigma_data ** 2 + sigma ** 2).sqrt()

    def t_steps(self, N, device):
        return (
                self.sigma_min ** (1 / self.rho) + torch.arange(N, device=device) / (N - 1) * (
                self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

    def N(self, k, K=800000, s0=2, s1=150):
        return int(
            np.ceil(np.sqrt(k / K * ((s1 + 1) ** 2 - s0 ** 2) + s0 ** 2) - 1) + 1)

    def mu(self, k, mu0=0.9, s0=2):
        return np.exp(s0 * np.log(mu0) / self.N(k))

    def forward(self, x, sigma, conditions, feature_seg, theta):
        # Time and condition embedding
        sigma_for_unet = sigma.log() / 4
        emb = self.map_noise(sigma_for_unet.flatten())
        if conditions is not None:
            emb = emb + self.map_label(conditions)
        emb = F.silu(self.map_layer1(F.silu(self.map_layer0(emb))))

        # U-Net now receives theta
        model_output = self.F_theta(x, emb, feature_seg, theta)

        # Apply consistency model scaling
        return self.c_skip(sigma.view(-1, 1, 1, 1)) * x + self.c_out(sigma.view(-1, 1, 1, 1)) * model_output


class ReSegNet(nn.Module):
    """
    The overall model containing both PIN and PGN, coordinating their operation.
    This is the main model you will train.
    """

    def __init__(self, conditioner, pin, pgn):
        super().__init__()
        self.conditioner = conditioner
        self.pin = pin
        self.pgn = pgn

    def forward(self, x, sigma, base_spectrogram):
        # 1. Extract conditional feature map
        feature_seg = self.conditioner(base_spectrogram)

        # 2. "Engineer" (PIN) analyzes features to create the "design" (theta)
        theta_predicted = self.pin(feature_seg)
        conditions = None  # 'conditions' seems deprecated in this flow, but kept for PGN API

        # 3. "Artist" (PGN) generates the output using the "design"
        output = self.pgn(x, sigma, conditions, feature_seg, theta_predicted)
        return output, feature_seg


class SFE(nn.Module):
    """Simple Feature Extractor block."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SFE, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conditioner(nn.Module):
    """Extracts a feature map to condition the PGN and PIN."""

    def __init__(self, in_channels=3, base_channels=16, out_dim=32):
        super(Conditioner, self).__init__()
        self.encoder1 = SFE(in_channels, base_channels)
        self.encoder2 = SFE(base_channels, base_channels * 2)
        self.encoder3 = SFE(base_channels * 2, base_channels * 4)
        # self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(base_channels * 4, out_dim))

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        # conditions = self.flatten(e3) # Not used, returns the feature map e3
        return e3


class Upsampler(nn.Module):
    """Upsamples the feature map from the conditioner."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.model(x)


class SegmentationHead(nn.Module):
    """Final convolutional layers to produce segmentation logits."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)


class PositionalEmbedding(torch.nn.Module):
    """Standard sinusoidal positional embedding."""

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class DoubleConv_FiLM(nn.Module):
    """U-Net style DoubleConv block, modified to accept time embedding and theta."""

    def __init__(self, in_channels, out_channels, num_groups=8, emb_channels=128, theta_dim=128):
        super().__init__()
        # For time embedding
        self.affine = nn.Linear(in_features=emb_channels, out_features=in_channels)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        # Add FiLM layer for theta modulation
        self.film = FiLMLayer(out_channels, theta_dim)

    def forward(self, x, emb, theta):
        # Add time embedding
        params = self.affine(emb).unsqueeze(2).unsqueeze(3)
        x = x + params

        x = self.double_conv(x)

        # Apply FiLM modulation
        x = self.film(x, theta)
        return x


class CrossAttention(nn.Module):
    """Cross-Attention layer for InteractiveBridge"""

    def __init__(self, query_dim, context_dim=None, num_heads=8):
        super().__init__()
        context_dim = context_dim if context_dim is not None else query_dim
        head_dim = query_dim // num_heads
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        b, q_len, _ = q.shape
        _, kv_len, _ = k.shape
        q = q.reshape(b, q_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(b, kv_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(b, kv_len, self.num_heads, -1).permute(0, 2, 1, 3)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        hidden_states = torch.matmul(attention_probs, v)
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(b, q_len, -1)
        return self.to_out(hidden_states)


class CIB(nn.Module):
    """
    Conditional Interactive Bridge (CIB).
    This module combines feature maps in two ways:
    1. 'inter': An unconditional InteractiveBridge using cross-attention.
    2. 'gate': A gated interactive bridge controlled by 'feature_seg'.
    The final output is a processed concatenation of both results.
    """

    def __init__(self, channels: int, channels_of_feature_seg: int, num_groups: int = 16):
        super().__init__()

        # --- 1. Modules for 'InteractiveBridge' (inter branch) ---
        self.norm_skip = nn.BatchNorm2d(channels)
        self.norm_decoder = nn.BatchNorm2d(channels)
        self.attn_dec_to_skip = CrossAttention(query_dim=channels)
        self.attn_skip_to_dec = CrossAttention(query_dim=channels)

        # --- 2. Modules for 'GatedInteractiveBridge' (gate branch) ---
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels_of_feature_seg, channels * 2),
            nn.Sigmoid()
        )
        # Processing layer to combine outputs
        self.proc = nn.Conv2d(channels * 4, channels, kernel_size=1)

    def forward(self, x_decoder: torch.Tensor, x_encoder_skip: torch.Tensor, feature_seg: torch.Tensor) -> torch.Tensor:
        # --- Branch 1: 'InteractiveBridge' logic ---
        b, c, h, w = x_encoder_skip.shape  # Get shape for reshaping later

        # Normalize
        x_skip_norm = self.norm_skip(x_encoder_skip)
        x_decoder_norm = self.norm_decoder(x_decoder)

        # Reshape for attention: (B, C, H, W) -> (B, H*W, C)
        query_skip = x_skip_norm.view(b, c, -1).permute(0, 2, 1)
        query_decoder = x_decoder_norm.view(b, c, -1).permute(0, 2, 1)

        # Apply Cross-Attention
        dec_refined = self.attn_dec_to_skip(query_decoder, query_skip)
        skip_refined = self.attn_skip_to_dec(query_skip, query_decoder)

        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        dec_refined = dec_refined.permute(0, 2, 1).view(b, c, h, w)
        skip_refined = skip_refined.permute(0, 2, 1).view(b, c, h, w)

        # Combine with residual connection and concatenate
        out_inter = torch.cat([dec_refined + x_decoder, skip_refined + x_encoder_skip], dim=1)

        # --- Branch 2: 'GatedInteractiveBridge' logic ---
        # 2a: Generate gates from feature_seg
        gates = self.gate_generator(feature_seg)
        gate_decoder, gate_encoder = torch.chunk(gates, 2, dim=-1)

        # 2b: Reshape and apply gates
        # Reshape from [B, C] -> [B, C, 1, 1] for broadcasting
        gate_decoder_rs = gate_decoder.view(-1, c, 1, 1)
        gate_encoder_rs = gate_encoder.view(-1, c, 1, 1)

        # Apply gates (element-wise multiplication)
        x_decoder_gated = x_decoder * gate_decoder_rs
        x_encoder_skip_gated = x_encoder_skip * gate_encoder_rs

        # 2c: Combine gated features
        out_gate = torch.cat([x_decoder_gated, x_encoder_skip_gated], dim=1)

        # --- 3. Final Combination ---
        # Concatenate 'inter' (C*2) and 'gate' (C*2) results
        # Final output will have C*4 channels, then processed to C channels
        return self.proc(torch.cat([out_inter, out_gate], dim=1))


class Downsampler(nn.Module):
    """
    Downsampling module.
    - If num_down > 0: Uses a single Conv with a large stride to downsample.
    - If num_down = 0: Uses Conv 3x3 (stride=1, padding=1) to process
      and change channel count without changing resolution.
    Uses Batch Normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, num_down: int, num_groups: int = 16):
        super().__init__()

        if num_down > 0:
            # Case: Downsample
            stride = 2 ** num_down
            kernel_size = stride

            self.model = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif num_down == 0:
            # Case: No downsampling, just process with Conv 3x3
            # stride=1 and padding=1 preserves H, W
            self.model = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError("num_down must be a non-negative integer.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNET_FiLM(nn.Module):
    """The core U-Net architecture, modified to use FiLM modulation."""

    def __init__(self, in_channels=4, out_channels=4, base_channels=32, emb_channels=128, theta_dim=128):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # Upgrade DoubleConv blocks to accept theta
        self.enc1 = DoubleConv_FiLM(in_channels, c1, emb_channels=emb_channels, theta_dim=theta_dim)
        self.enc2 = DoubleConv_FiLM(c1, c2, emb_channels=emb_channels, theta_dim=theta_dim)
        self.enc3 = DoubleConv_FiLM(c2, c3, emb_channels=emb_channels, theta_dim=theta_dim)
        self.bottleneck = DoubleConv_FiLM(c3, c4, emb_channels=emb_channels, theta_dim=theta_dim)
        self.up_double_conv1 = DoubleConv_FiLM(c3, c3, emb_channels=emb_channels, theta_dim=theta_dim)
        self.up_double_conv2 = DoubleConv_FiLM(c2, c2, emb_channels=emb_channels, theta_dim=theta_dim)
        self.up_double_conv3 = DoubleConv_FiLM(c1, c1, emb_channels=emb_channels, theta_dim=theta_dim)

        # Standard U-Net components
        self.down1, self.down2, self.down3 = nn.MaxPool2d(2), nn.MaxPool2d(2), nn.MaxPool2d(2)

        # Components for conditioning
        self.down_cond1 = Downsampler(base_channels * 4, c3, 2, 16)
        self.down_cond2 = Downsampler(base_channels * 4, c2, 1, 16)
        self.down_cond3 = Downsampler(base_channels * 4, c1, 0, 16)

        self.intergrate_cond1 = CIB(c3, c3)
        self.intergrate_cond2 = CIB(c2, c2)
        self.intergrate_cond3 = CIB(c1, c1)

        self.up_conv1 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x, emb, feature_seg, theta):  # Add theta to signature
        x1 = self.enc1(x, emb, theta);
        p1 = self.down1(x1)
        x2 = self.enc2(p1, emb, theta);
        p2 = self.down2(x2)
        x3 = self.enc3(p2, emb, theta);
        p3 = self.down3(x3)

        b = self.bottleneck(p3, emb, theta)

        u1 = self.up_conv1(b);
        e1 = self.intergrate_cond1(x3, u1, self.down_cond1(feature_seg))
        d1 = self.up_double_conv1(e1, emb, theta)

        u2 = self.up_conv2(d1);
        e2 = self.intergrate_cond2(x2, u2, self.down_cond2(feature_seg))
        d2 = self.up_double_conv2(e2, emb, theta)

        u3 = self.up_conv3(d2);
        e3 = self.intergrate_cond3(x1, u3, self.down_cond3(feature_seg))
        d3 = self.up_double_conv3(e3, emb, theta)

        return self.final_conv(d3)


def evaluate_model(model, vae_model, seg_head, upsample_model, dataloader, device, config):
    """
    Calculates metrics (mIoU, mAcc, F1) and displays them on a progress bar.
    """
    model.eval()
    seg_head.eval()
    upsample_model.eval()
    vae_model.eval()

    # Initialize metric functions
    miou_fn = JaccardIndex("multiclass", num_classes=config["NUM_CLASSES"], average="macro").to(device)
    macc_fn = MulticlassAccuracy(num_classes=config["NUM_CLASSES"], average="macro").to(device)
    f1_fn = MulticlassF1Score(num_classes=config["NUM_CLASSES"], average="macro").to(device)

    # Assign tqdm to a variable to call set_postfix
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for i, (spectrograms, label_mask_indexed, _) in enumerate(progress_bar):
            spectrograms, label_mask_indexed = spectrograms.to(device), label_mask_indexed.to(device)

            # --- Run the generation and segmentation pipeline ---
            z = torch.randn((spectrograms.shape[0], config["LATENT_CHANNELS"], 32, 32), device=device)
            initial_noise = z * model.pgn.sigma_max
            sigma_max_tensor = torch.full((spectrograms.shape[0],), model.pgn.sigma_max, device=device)

            # Use the main model (ReSegNet)
            predicted_latents_eval, feature_seg_eval = model(initial_noise, sigma_max_tensor, spectrograms)

            # Decode and segment
            reconstructed_image_eval = vae_model.decoder(predicted_latents_eval)
            upsampled_feature_eval = upsample_model(feature_seg_eval)
            seg_head_input_eval = torch.cat([reconstructed_image_eval, upsampled_feature_eval], dim=1)
            seg_logits_eval = seg_head(seg_head_input_eval)

            # --- Calculate and update metrics ---
            preds = torch.argmax(seg_logits_eval, dim=1)

            miou_fn.update(preds, label_mask_indexed)
            macc_fn.update(preds, label_mask_indexed)
            f1_fn.update(preds, label_mask_indexed)

            # Calculate running metrics for live progress bar
            current_miou = miou_fn.compute().item()
            current_macc = macc_fn.compute().item()
            current_f1 = f1_fn.compute().item()

            progress_bar.set_postfix({
                'mIoU': f'{current_miou:.4f}',
                'mAcc': f'{current_macc:.4f}',
                'F1': f'{current_f1:.4f}'
            })

    # After iterating through the whole dataloader, compute final results
    final_miou = miou_fn.compute().item()
    final_macc = macc_fn.compute().item()
    final_f1 = f1_fn.compute().item()

    # Reset metrics for the next evaluation
    miou_fn.reset()
    macc_fn.reset()
    f1_fn.reset()

    return {"F1": final_f1, "mIoU": final_miou, "mAcc": final_macc}


class LossAwareSampler:
    """
    Manages Dual Adaptive Sampling, using a Moving Average with a
    Natural Decaying Update Rate (alpha = 1/N).
    This is a stable, dynamic smoothing without extra hyperparameters.
    """

    def __init__(self, num_steps, num_bins=10, min_jump=1, max_jump=15, exploration_factor=0.1):
        self.num_steps = num_steps
        self.num_bins = num_bins
        self.min_jump = min_jump
        self.max_jump = max_jump
        self.exploration_factor = exploration_factor
        self.bin_size = num_steps // num_bins

        # Initialize with zeros for both loss and counts.
        # Loss history will store the smoothed average value.
        self.loss_history = torch.zeros(num_bins)
        self.bin_counts = torch.zeros(num_bins)  # Number of times a bin has been updated

    def update_loss(self, t_indices, losses):
        """
        Updates the smoothed average loss.
        Formula: L_new_avg = (1 - alpha_N) * L_old_avg + alpha_N * L_batch_avg
        Where alpha_N = 1 / (N_real_old + N_batch)
        """
        losses = losses.cpu().detach()
        t_indices = t_indices.cpu()

        for bin_idx in range(self.num_bins):
            # Define the index range for the current bin
            low_bound = bin_idx * self.bin_size

            #  Modification: Ensure the last bin goes all the way to num_steps
            if bin_idx == self.num_bins - 1:
                high_bound = self.num_steps  # Go to the end
            else:
                high_bound = (bin_idx + 1) * self.bin_size

            # Identify samples belonging to the current bin
            mask = (t_indices >= low_bound) & (t_indices < high_bound)

            if mask.any():
                # 1. Calculate current batch Average Loss (L_batch_avg)
                L_batch_avg = losses[mask].mean()
                N_batch = mask.sum().item()

                # 2. Get old and new counts
                N_real_old = self.bin_counts[bin_idx].item()
                N_real_new = N_real_old + N_batch

                # 3. Calculate smoothing coefficient (alpha_N)
                # Alpha gets smaller as counts increase (more stable)
                alpha_N = 1.0 / (N_real_new + 1e-6)  # Avoid division by zero

                # 4. Apply Decaying Update Rate Formula
                L_old_avg = self.loss_history[bin_idx]
                L_new_avg = (1 - alpha_N) * L_old_avg + alpha_N * L_batch_avg

                # 5. Update
                self.loss_history[bin_idx] = L_new_avg
                self.bin_counts[bin_idx] = N_real_new

    def get_current_loss_weights(self):
        """
        Calculates sampling weights from the smoothed average loss.
        Loss for bins that have never been updated (N=0) is set to 1.0 (high weight)
        to encourage exploration.
        """
        # Return the smoothed loss_history if bin_counts > 0, else 1.0
        return torch.where(
            self.bin_counts > 0,
            self.loss_history,
            torch.tensor(1.0, dtype=self.loss_history.dtype)
        )

    def sample(self, batch_size, device):
        """Performs dual adaptive sampling."""

        current_losses = self.get_current_loss_weights()

        # --- Step 1: Importance Sampling (Choose "Weak Spots") ---
        # Convert losses to probability weights
        loss_weights = F.softmax(current_losses, dim=0)

        # Combine with a uniform probability for "exploration"
        uniform_weights = torch.ones_like(loss_weights) / self.num_bins
        sampling_weights = (1 - self.exploration_factor) * loss_weights + self.exploration_factor * uniform_weights

        # Select bins for the current batch based on weights
        chosen_bins = torch.multinomial(sampling_weights, batch_size, replacement=True)

        # From each chosen bin, randomly select a `t` index
        low = chosen_bins * self.bin_size

        #  Modification: Adjust 'high' for the last bin
        high = torch.zeros_like(low)  # Create 'high' tensor of the same size
        for i in range(batch_size):
            bin_idx = chosen_bins[i].item()
            if bin_idx == self.num_bins - 1:
                high[i] = self.num_steps  # Last bin goes to the end
            else:
                high[i] = (bin_idx + 1) * self.bin_size

        # Clamp the max index 'i' to ensure t_j doesn't exceed num_steps
        max_limit = self.num_steps - self.max_jump
        t_indices_i_raw = (low + torch.rand(batch_size) * (high - low)).long()
        t_indices_i = torch.clamp(t_indices_i_raw, 0, max_limit - 1).to(device)

        # Get the losses corresponding to the chosen bins
        selected_losses = current_losses[chosen_bins]

        # --- Step 2: Adaptive Jump Size (Decide "How to Learn") ---
        # Normalize the selected bin losses to [0, 1]
        min_loss = current_losses.min()
        max_loss = current_losses.max()
        if max_loss - min_loss < 1e-6:
            normalized_loss = torch.zeros_like(selected_losses)
        else:
            normalized_loss = (selected_losses - min_loss) / (max_loss - min_loss)

        # Calculate dynamic jump size: high loss -> small jump, low loss -> big jump
        jump_range = self.max_jump - self.min_jump
        dynamic_jump_size = self.max_jump - (normalized_loss * jump_range).round().long()
        dynamic_jump_size = dynamic_jump_size.to(device)

        # Calculate the second index
        t_indices_j = t_indices_i + dynamic_jump_size
        t_indices_j = torch.clamp(t_indices_j, max=self.num_steps - 1)

        return t_indices_i, t_indices_j


config = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-4,
    "EPOCHS": 60,
    "LATENT_CHANNELS": 3,
    "NUM_CLASSES": 4,
    "VAE_CHECKPOINT_PATH": "/kaggle/input/vae-best.pth",
    "SAVE_DIR": "results_physics_aware",
    "CHECKPOINT_DIR": "check_point_physics_aware",

    # Conditioner parameters
    "COND_BASE_CHANNELS": 16,
    "CONDITION_DIM": 256,  # Flattened output of conditioner (if used)
    "COND_FEATURE_CHANNELS": 16 * 4,  # Channel count of the feature map from conditioner
    "DOWN_SCALE_LATTENT": 8,

    # UNET parameters
    "UNET_BASE_CHANNELS": 16,

    # General parameters
    "EMB_CHANNELS": 128,

    # New architecture parameters
    "THETA_DIM": 128,  # Size of the "physics design" vector
    "EMA_MU": 0.95,  # Update coefficient for target network

    "NUM_STEPS": 50,  # Number of diffusion steps

    # --- DUAL ADAPTIVE SAMPLING PARAMETERS ---
    "SAMPLER_NUM_BINS": 8,  # Split NUM_STEPS into bins
    "SAMPLER_MOMENTUM": 0.9,  # "Memory" of loss history (Not used in current sampler)
    "SAMPLER_MIN_JUMP": 1,  # Smallest step
    "SAMPLER_MAX_JUMP": 5,  # Largest step
    "SAMPLER_EXPLORATION": 0.1,  # 10% chance of random exploration

    # --- LOSS PARAMETERS ---
    "LAMDA_SEG": 0.3,
    "lambda_alpha": 0.3,
    "lambda_mean": 1.5,
    "lambda_curvature": 1.1,
    "height_patch": 32,
    "width_patch": 1,

    # --- RESUME TRAINING ---
    "RESUME_CHECKPOINT_PATH": "/kaggle/input/model-4/check_point_physics_aware/epoch_checkpoints/checkpoint_epoch_10.pth",
    "START_EPOCH": 1,  # Default is 1 (start from scratch)
}

# --- 2. LOAD DATA ---
os.makedirs(config["SAVE_DIR"], exist_ok=True)
os.makedirs(config["CHECKPOINT_DIR"], exist_ok=True)
train_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

train_dataset = SemanticSegmentationDataset(
    image_dir="/kaggle/input/5g-lte-nr-j03/J03_spectrumm/train/data",
    label_dir="/kaggle/input/5g-lte-nr-j03/J03_spectrumm/train/label",
    transform=train_transform
)

val_dataset = SemanticSegmentationDataset(
    image_dir="/kaggle/input/5g-lte-nr-j03/J03_spectrumm/test/data",
    label_dir="/kaggle/input/5g-lte-nr-j03/J03_spectrumm/test/label",
    transform=train_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=4,
                              pin_memory=True)
test_dataloader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=4,
                             pin_memory=True)

# --- 3. INITIALIZE NEW ARCHITECTURE ---
print("--- Initializing Physics-Aware Models ---")
device = config["DEVICE"]

# VAE Model (frozen, only for encode/decode)
vae_model = VAE(in_channels=3, out_channels=3, latent_channels=config["LATENT_CHANNELS"],
                base_channel=config["COND_BASE_CHANNELS"],
                downsample_factor=config["DOWN_SCALE_LATTENT"]).to(device)
if os.path.exists(config["VAE_CHECKPOINT_PATH"]):
    vae_model.load_state_dict(torch.load(config["VAE_CHECKPOINT_PATH"], map_location=device))
    print("VAE checkpoint loaded successfully.")
else:
    print(f"Warning: VAE checkpoint not found. Using random weights.")
vae_model.eval()
for p in vae_model.parameters(): p.requires_grad = False

# Auxiliary models (trained)
conditioner = Conditioner(in_channels=3, base_channels=config["COND_BASE_CHANNELS"],
                          out_dim=config["CONDITION_DIM"]).to(device)
upsample_model = Upsampler(config["COND_FEATURE_CHANNELS"], config["COND_FEATURE_CHANNELS"]).to(device)
seg_head = SegmentationHead(3 + config["COND_FEATURE_CHANNELS"], config["NUM_CLASSES"]).to(device)

# Initialize the "Dual Brain" (PIN and PGN)
pin_model = ParameterInferenceNetwork(
    in_channels=config["COND_FEATURE_CHANNELS"],
    theta_dim=config["THETA_DIM"]
).to(device)

pgn_model = ParameterizedGenerationNetwork(
    in_channels=config["LATENT_CHANNELS"], out_channels=config["LATENT_CHANNELS"],
    base_channels=config["UNET_BASE_CHANNELS"], condition_dim=config["CONDITION_DIM"],
    emb_channels=config["EMB_CHANNELS"], theta_dim=config["THETA_DIM"]
).to(device)

# The main model (online_net) combines all trainable generation components
online_net = ReSegNet(conditioner, pin_model, pgn_model).to(device)
# The target_net is an EMA copy for consistency training
target_net = deepcopy(online_net)
target_net.eval()
for p in target_net.parameters(): p.requires_grad = False

# --- 4. INITIALIZE OPTIMIZER AND LOSS ---
# Optimizer now optimizes weights of online_net (PIN, PGN, Conditioner) and auxiliary models
optimizer = torch.optim.Adam(
    list(online_net.parameters()) + list(seg_head.parameters()) + list(upsample_model.parameters()),
    lr=config["LEARNING_RATE"]
)

# Loss functions
main_loss_fn = SUR(lambda_mean=config["lambda_mean"], lambda_curvature=config["lambda_curvature"],
                   lambda_alpha=config["lambda_alpha"], std_threshold=1e-6).to(device)
consistency_loss_fn = nn.MSELoss(reduction='none').to(device)  # Loss for comparing latents

# --- 5. INITIALIZE HISTORY LISTS ---
train_loss_history = []
val_miou_history = []
val_macc_history = []
val_f1_history = []

# --- 6. RESUME TRAINING LOGIC ---
print("\n--- Starting Training ---")
best_test_iou = 0.0

# Initialize sampler
sampler = LossAwareSampler(
    num_steps=config["NUM_STEPS"],
    num_bins=config["SAMPLER_NUM_BINS"],
    min_jump=config["SAMPLER_MIN_JUMP"],
    max_jump=config["SAMPLER_MAX_JUMP"],
    exploration_factor=config["SAMPLER_EXPLORATION"]
)
t_steps = online_net.pgn.t_steps(config['NUM_STEPS'], device)

# --- Load checkpoint to resume training ---
resume_path = config.get("RESUME_CHECKPOINT_PATH", None)
start_epoch = config.get("START_EPOCH", 1)  # Default is epoch 1

if resume_path and os.path.exists(resume_path):
    print(f"\n--- Resuming training from checkpoint: {resume_path} ---")
    checkpoint = torch.load(resume_path, map_location=device)

    # 1. Load model states
    online_net.load_state_dict(checkpoint['online_net_state_dict'])
    seg_head.load_state_dict(checkpoint['seg_head_state_dict'])
    upsample_model.load_state_dict(checkpoint['upsample_model_state_dict'])

    # 2. Load Optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 3. Update start epoch and best mIoU
    start_epoch = checkpoint['epoch'] + 1  # Start from the *next* epoch
    best_test_iou = checkpoint.get('best_test_iou', 0.0)

    print(f"✅ Successfully loaded checkpoint. Resuming from Epoch {start_epoch}.")
    print(f"   Current Best mIoU: {best_test_iou:.4f}")
else:
    print(f"\n--- Starting training from scratch (Epoch {start_epoch}) ---")

best_epoch = start_epoch
# --- END RESUME LOGIC ---


# Adjust main loop to use start_epoch
for epoch in range(start_epoch - 1, config["EPOCHS"]):  # Start from index (start_epoch - 1)

    online_net.train()
    seg_head.train()
    upsample_model.train()

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['EPOCHS']}")
    epoch_train_loss = 0.0

    for spectrograms, label_mask_indexed, clean_masks in progress_bar:
        spectrograms = spectrograms.to(device)
        label_mask_indexed = label_mask_indexed.to(device)
        clean_masks = clean_masks.to(device)

        optimizer.zero_grad()

        # Get clean latents from VAE
        with torch.no_grad():
            clean_latents = vae_model.encoder(clean_masks)[0].detach()  # Only use mean as latent

        # Sample time steps using the adaptive sampler
        indices_i, indices_j = sampler.sample(clean_latents.size(0), device)
        t_i = t_steps.gather(0, indices_i)
        t_j = t_steps.gather(0, indices_j)

        # Create noisy latents
        noise = torch.randn_like(clean_latents)
        noisy_latent_A = clean_latents + noise * t_j.view(-1, 1, 1, 1)  # Noisier
        noisy_latent_B = clean_latents + noise * t_i.view(-1, 1, 1, 1)  # Less noisy

        # --- Forward pass ---
        # Online network predicts from the noisier latent
        output_A, feature_seg = online_net(noisy_latent_A, t_j, spectrograms)
        # Target network predicts from the less noisy latent
        with torch.no_grad():
            output_B, _ = target_net(noisy_latent_B, t_i, spectrograms)

        # Decode latents back to image space
        reconstructed_A = vae_model.decoder(output_A)
        with torch.no_grad():
            reconstructed_B = vae_model.decoder(output_B)

        # --- Loss Calculation ---
        # 1. Consistency Loss (in image space)
        latent_loss = consistency_loss_fn(reconstructed_A, reconstructed_B).mean(dim=(1, 2, 3))
        # Update the sampler's loss history
        sampler.update_loss(indices_i, latent_loss)
        mean_latent_loss = latent_loss.mean()

        # 2. Segmentation Loss (from the online network's output)
        reconstructed_image = reconstructed_A
        upsampled_feature = upsample_model(feature_seg)
        seg_head_input = torch.cat([reconstructed_image, upsampled_feature], dim=1)
        seg_logits = seg_head(seg_head_input)

        # Convert to probabilities and one-hot for SUR loss
        seg_probs = F.softmax(seg_logits, dim=1)
        masks_one_hot = F.one_hot(label_mask_indexed.long(), num_classes=config["NUM_CLASSES"]).permute(0, 3, 1,
                                                                                                        2).float()

        # Calculate SUR loss
        segmentation_loss, avg_loss_c1, avg_loss_c2 = main_loss_fn(
            seg_probs, masks_one_hot,
            config["height_patch"], config["width_patch"],
            config["height_patch"], config["width_patch"]
        )

        # 3. Total Loss
        total_loss = mean_latent_loss + config["LAMDA_SEG"] * segmentation_loss
        epoch_train_loss += total_loss.item()

        # --- Backward and Step ---
        total_loss.backward()
        optimizer.step()

        # --- Update Target Network (EMA) ---
        with torch.no_grad():
            mu = config['EMA_MU']
            for online_param, target_param in zip(online_net.parameters(), target_net.parameters()):
                # Use out-of-place operation and .data assignment for safety
                target_param.data = mu * target_param.data + (1 - mu) * online_param.data

        progress_bar.set_postfix({
            'total_loss': f'{total_loss.item():.4f}',
            'latent_loss': f'{mean_latent_loss.item():.4f}',
            'seg_loss': f'{segmentation_loss.item():.4f}',
            'avg_loss_c1': f'{avg_loss_c1.item():.4f}',
            'avg_loss_c2': f'{avg_loss_c2.item():.4f}',
        })

    avg_epoch_train_loss = epoch_train_loss / len(train_dataloader)
    train_loss_history.append(avg_epoch_train_loss)

    # --- Evaluation at end of epoch ---
    eval_results = evaluate_model(
        online_net, vae_model, seg_head, upsample_model,
        test_dataloader, device, config
    )
    val_miou_history.append(eval_results['mIoU'])
    val_macc_history.append(eval_results['mAcc'])
    val_f1_history.append(eval_results['F1'])

    print(f"\n--- Epoch {epoch + 1} Evaluation ---")
    print(f"Test mIoU: {eval_results['mIoU']:.4f}| mAcc: {eval_results['mAcc']:.4f} | F1: {eval_results['F1']:.4f}")

    # Save best model based on mIoU
    if eval_results['mIoU'] > best_test_iou:
        best_test_iou = eval_results['mIoU']
        best_epoch = epoch + 1
        print(f"🎉 New best model found at epoch {epoch + 1} with mIoU: {best_test_iou:.4f}")
        torch.save(online_net.state_dict(), os.path.join(config["CHECKPOINT_DIR"], "best_model.pth"))

    # --- Save full checkpoint for resuming ---
    epoch_save_dir = os.path.join(config["CHECKPOINT_DIR"], "epoch_checkpoints")
    os.makedirs(epoch_save_dir, exist_ok=True)
    epoch_save_path = os.path.join(epoch_save_dir, f"checkpoint_epoch_{epoch + 1}.pth")

    checkpoint = {
        'epoch': epoch + 1,
        'online_net_state_dict': online_net.state_dict(),
        'seg_head_state_dict': seg_head.state_dict(),
        'upsample_model_state_dict': upsample_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_iou': best_test_iou,
        'current_miou': eval_results['mIoU'],
        # Note: vae_model state is not saved as it's frozen and loaded from a separate file
    }

    torch.save(checkpoint, epoch_save_path)
    print(f"✅ Saved full checkpoint for epoch {epoch + 1} at: {epoch_save_path}")

print(f"\nTraining finished. Best model mIoU: {best_test_iou:.4f} (from Epoch {best_epoch})")