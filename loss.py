import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionOverlapLoss(nn.Module):
    def __init__(self, target_size=(224, 224)):
        """
        Args:
            target_size (tuple): Size (H, W) to which attention maps and bboxes are expected.
            penalize_outside (bool): If True, subtracts attention outside the bbox.
        """
        super().__init__()
        self.H, self.W = target_size

        


    def forward(self, attn_map, bboxes):
        """
        Args:
            attn_map (Tensor): shape (B, H, W), attention maps per image.
            bboxes (Tensor): shape (B, 4), bounding boxes in format [x1, y1, x2, y2] in attention map coordinates.
        Returns:
            overlap_loss (Tensor): scalar loss (mean over batch).
        """
        B, H, W = attn_map.shape
        
        assert H == self.H and W == self.W, f"Attention map size mismatch: got {H}x{W}, expected {self.H}x{self.W}"

        # Normalize attention maps
        attn_map = attn_map.view(B, -1)
        attn_map = (attn_map - attn_map.min(dim=1, keepdim=True)[0]) / \
                   (attn_map.max(dim=1, keepdim=True)[0] - attn_map.min(dim=1, keepdim=True)[0] + 1e-8)
        attn_map = attn_map / (attn_map.sum(dim=1, keepdim=True) + 1e-8)
        attn_map = attn_map.view(B, H, W)

        masks = torch.zeros_like(attn_map)

        for i in range(B):
            x1, y1, x2, y2 = bboxes[i]
            x1 = int(torch.clamp(x1, 0, W - 1).item())
            x2 = int(torch.clamp(x2, 0, W - 1).item())
            y1 = int(torch.clamp(y1, 0, H - 1).item())
            y2 = int(torch.clamp(y2, 0, H - 1).item())
            masks[i, y1:y2+1, x1:x2+1] = 1.0

        mask_outside = 1.0 - masks

        # Compute attention inside and outside the bbox
        attn_inside = (attn_map * masks).sum(dim=(1, 2))  # shape (B,)

        attn_outside = (attn_map * mask_outside).sum(dim=(1, 2))
        loss = attn_outside


        # encourage total mass inside bbox

        return loss.mean()
