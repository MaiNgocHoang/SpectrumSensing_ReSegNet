import torch
import torch.nn as nn
import torch.nn.functional as F


class SUR(nn.Module):
    """
    A unified hybrid loss version using a scaling method for all components.
    """

    def __init__(self, lambda_mean=1.0, std_threshold=1e-6, lambda_curvature=1, lambda_alpha=1.0):
        super().__init__()
        self.lambda_mean = lambda_mean
        self.std_threshold = std_threshold
        self.lambda_curvature = lambda_curvature
        self.lambda_alpha = lambda_alpha

        # Define the Laplacian kernel for curvature calculation
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        # Register as a buffer so it moves to the device with the model (e.g., .to(device) or .cuda())
        self.register_buffer('laplacian_kernel', laplacian_kernel.view(1, 1, 3, 3))

    def forward(self, y_pred, y_true, patch_height, patch_width, stride_h, stride_w):
        B, C, H, W = y_true.shape
        device = y_true.device

        # Reshape for patch-based processing
        y_true_reshaped = y_true.reshape(B * C, 1, H, W)
        y_pred_reshaped = y_pred.reshape(B * C, 1, H, W)

        # --- Calculate patch statistics and create masks ---
        avg_pool = nn.AvgPool2d(kernel_size=(patch_height, patch_width), stride=(stride_h, stride_w), padding=0)

        mean_true_map = avg_pool(y_true_reshaped)
        var_true_map = avg_pool(y_true_reshaped.pow(2)) - mean_true_map.pow(2)
        std_true_map = torch.sqrt(var_true_map.clamp(min=0))  # clamp(min=0) for numerical stability

        # Split patches into two cases based on the standard deviation of the ground truth
        low_res_mask_case2 = std_true_map > self.std_threshold  # Case 2: High variance (texture/edges)
        low_res_mask_case1 = ~low_res_mask_case2  # Case 1: Low variance (flat regions)

        num_patches_c1 = low_res_mask_case1.sum().item()
        num_patches_c2 = low_res_mask_case2.sum().item()

        # === CASE 1: L_mean + λ * L_var (for low-variance patches) ===
        loss_sum_c1 = torch.tensor(0.0, device=device)
        if num_patches_c1 > 0:
            mean_pred_map = avg_pool(y_pred_reshaped)
            var_pred_map = avg_pool(y_pred_reshaped.pow(2)) - mean_pred_map.pow(2)

            loss_mean = (mean_pred_map - mean_true_map).pow(2)
            loss_var = var_pred_map.clamp(min=0)

            case1_loss_map = loss_var + self.lambda_mean * loss_mean
            loss_sum_c1 = case1_loss_map[low_res_mask_case1].sum()

        # === CASE 2: L_WL1 + λ * L_curv (for high-variance patches) ===
        loss_sum_c2 = torch.tensor(0.0, device=device)
        if num_patches_c2 > 0:
            # Unfold patches for pixel-level loss calculation
            y_true_unfolded = F.unfold(y_true_reshaped, kernel_size=(patch_height, patch_width),
                                       stride=(stride_h, stride_w))
            y_pred_unfolded = F.unfold(y_pred_reshaped, kernel_size=(patch_height, patch_width),
                                       stride=(stride_h, stride_w))

            y_true_flat = y_true_unfolded.transpose(1, 2).reshape(-1, patch_height * patch_width)
            y_pred_flat = y_pred_unfolded.transpose(1, 2).reshape(-1, patch_height * patch_width)

            mask_flat = low_res_mask_case2.view(-1)

            # Select only the patches corresponding to Case 2
            true_patches_case2 = y_true_flat[mask_flat]
            pred_patches_case2 = y_pred_flat[mask_flat]

            # --- L_WL1 (Weighted L1) Calculation ---
            val_min = torch.min(true_patches_case2, dim=1, keepdim=True)[0]
            val_max = torch.max(true_patches_case2, dim=1, keepdim=True)[0]

            count_max = (true_patches_case2 == val_max).sum(dim=1).float()
            total_pixels = true_patches_case2.shape[1]
            count_min = total_pixels - count_max

            # Calculate weights inversely proportional to pixel count
            weight_max = total_pixels / (count_max + 1e-8)  # Add epsilon for numerical stability
            weight_min = total_pixels / (count_min + 1e-8)

            is_max_mask = (true_patches_case2 == val_max)
            weight_map = torch.where(is_max_mask, weight_max.unsqueeze(1), weight_min.unsqueeze(1))

            abs_error = torch.abs(pred_patches_case2 - true_patches_case2)
            weighted_error = weight_map * abs_error
            original_case2_losses = torch.mean(weighted_error, dim=1)  # This is L_WL1

            # --- L_curv (Curvature) Calculation ---
            num_c2_patches = true_patches_case2.shape[0]
            patches_true_2d = true_patches_case2.view(num_c2_patches, 1, patch_height, patch_width)
            patches_pred_2d = pred_patches_case2.view(num_c2_patches, 1, patch_height, patch_width)

            laplacian_true = F.conv2d(patches_true_2d, self.laplacian_kernel, padding=1)
            laplacian_pred = F.conv2d(patches_pred_2d, self.laplacian_kernel, padding=1)

            curvature_diff = torch.abs(laplacian_pred - laplacian_true)
            curvature_losses = torch.mean(curvature_diff.view(num_c2_patches, -1), dim=1)  # This is L_curv

            # Combine the two losses for Case 2
            combined_case2_losses = original_case2_losses + self.lambda_curvature * curvature_losses
            loss_sum_c2 = combined_case2_losses.sum()

        # === TOTAL LOSS CALCULATION ===
        # Calculate the average loss for each case to normalize by the number of patches
        avg_loss_c1 = loss_sum_c1 / num_patches_c1 if num_patches_c1 > 0 else torch.tensor(0.0, device=device)
        avg_loss_c2 = loss_sum_c2 / num_patches_c2 if num_patches_c2 > 0 else torch.tensor(0.0, device=device)

        # Combine the average losses, scaling Case 2 by lambda_alpha
        total_loss = avg_loss_c1 + self.lambda_alpha * avg_loss_c2

        return total_loss, avg_loss_c1, avg_loss_c2