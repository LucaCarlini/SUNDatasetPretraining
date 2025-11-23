import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


def one_hot(x, num_classes, on_value=1.0, off_value=0.0, device="cuda"):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(
        1, x, on_value
    )


def mixup_target(target, num_classes, lam=1.0, smoothing=0.0, device="cuda"):
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y1 = one_hot(
        target, num_classes, on_value=on_value, off_value=off_value, device=device
    )
    y2 = one_hot(
        target.flip(0),
        num_classes,
        on_value=on_value,
        off_value=off_value,
        device=device,
    )
    return y1 * lam + y2 * (1.0 - lam)


def rand_bbox(img_shape, lam, margin=0.0, count=None):
    """Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(
        int(img_h * minmax[0]), int(img_h * minmax[1]), size=count
    )
    cut_w = np.random.randint(
        int(img_w * minmax[0]), int(img_w * minmax[1]), size=count
    )
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(
    img_shape, lam, ratio_minmax=None, correct_lam=True, count=None
):
    """Generate bbox and apply lambda correction."""
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam




class SoftTargetCrossEntropy(torch.nn.Module):
    def __init__(self) -> None:
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

import torch.nn.functional as F
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionOverlapLoss(nn.Module):
    def __init__(self, target_size=(224, 224), penalize_outside=False, inside_weight=1.0, outside_weight=1.0):
        """
        Args:
            target_size (tuple): Size (H, W) to which attention maps and bboxes are expected.
            penalize_outside (bool): If True, subtracts attention outside the bbox.
        """
        super().__init__()
        self.H, self.W = target_size
        self.penalize_outside = penalize_outside
        self.inside_weight = inside_weight
        self.outside_weight = outside_weight

        # if not penalize_outside and outside_weight != 1.0 raise warning
        if not penalize_outside and outside_weight != 1.0:
            print("Warning: outside_weight is ignored when penalize_outside is False.")


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
        if self.penalize_outside:
            attn_outside = (attn_map * mask_outside).sum(dim=(1, 2))
            loss = (1 - attn_inside) * self.inside_weight + attn_outside * self.outside_weight
        else:
            loss = (1.0 - attn_inside) * self.inside_weight

        # encourage total mass inside bbox

        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha) if isinstance(alpha, (float, int)) else alpha
        self.gamma = gamma
        self.reduction = reduction
        

    def forward(self, inputs, targets):
        # Ensure alpha is on the same device as inputs
        if isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.to(inputs.device)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # Probabilities of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DINOLoss(nn.Module):
    def __init__(self, num_prototypes, teacher_temp=0.1, student_temp=0.5,
                 center_momentum=0.9, use_sinkhorn=True):
        """
        Implements the DINO loss function.
        
        Args:
            num_prototypes (int): Number of prototypes.
            teacher_temp (float): Temperature for the teacher network.
            student_temp (float): Temperature for the student network.
            center_momentum (float): Momentum for centering.
            use_sinkhorn (bool): Whether to use Sinkhorn-Knopp normalization.
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.use_sinkhorn = use_sinkhorn

        # Initialize center
        self.register_buffer("center", torch.zeros(1, num_prototypes))

    def forward(self, student_output, teacher_output):
        """
        Compute the DINO loss.
        
        Args:
            student_output (torch.Tensor): Output from the student network.
            teacher_output (torch.Tensor): Output from the teacher network.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        # Apply softmax with temperature scaling
        student_probs = F.softmax(student_output / self.student_temp, dim=-1)
        center = self.center.to(teacher_output.device)
        teacher_probs = F.softmax((teacher_output - center) / self.teacher_temp, dim=-1)
        # Apply Sinkhorn-Knopp normalization if required
        if self.use_sinkhorn:
            teacher_probs = self.sinkhorn_knopp(teacher_probs)

        # Compute cross-entropy loss
        loss = -torch.sum(teacher_probs * torch.log(student_probs + 1e-9), dim=-1)
        loss = loss.mean()

        # Update center with exponential moving average
        self.update_center(teacher_output)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Updates the center based on the teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center.to(batch_center.device)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def sinkhorn_knopp(self, teacher_probs, num_iters=1):
        """
        Applies the Sinkhorn-Knopp normalization.

        Args:
            teacher_probs (torch.Tensor): The teacher probabilities.
            num_iters (int): Number of iterations for normalization.
        
        Returns:
            torch.Tensor: Normalized probabilities.
        """
        Q = torch.exp(teacher_probs).clone()
        Q /= Q.sum(dim=1, keepdim=True)
        for _ in range(num_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
        return Q
    

class IBOTLoss(nn.Module):
    def __init__(self, num_prototypes, teacher_temp=0.04, student_temp=0.3,
                 center_momentum=0.9, use_sinkhorn=False, patch_loss_weight=1.0):
        """
        Implements the iBOT loss function.
        
        Args:
            num_prototypes (int): Number of prototypes.
            teacher_temp (float): Temperature for the teacher network.
            student_temp (float): Temperature for the student network.
            center_momentum (float): Momentum for centering.
            use_sinkhorn (bool): Whether to use Sinkhorn-Knopp normalization.
            patch_loss_weight (float): Weight for patch-level loss.
        """
        super().__init__()
        self.dino_loss = DINOLoss(num_prototypes, teacher_temp, student_temp, 
                                  center_momentum, use_sinkhorn)
        self.patch_loss_weight = patch_loss_weight

    def forward(self, student_cls_output, teacher_cls_output, student_patch_output, teacher_patch_output):
        """
        Compute the iBOT loss.
        
        Args:
            student_cls_output (torch.Tensor): Output from the student network (CLS token).
            teacher_cls_output (torch.Tensor): Output from the teacher network (CLS token).
            student_patch_output (torch.Tensor): Output from the student network (patch tokens).
            teacher_patch_output (torch.Tensor): Output from the teacher network (patch tokens).
        
        Returns:
            torch.Tensor: Computed loss.
        """
        # Compute DINO loss for class tokens
        cls_loss = self.dino_loss(student_cls_output, teacher_cls_output)

        # Compute iBOT loss for patch tokens
        patch_loss = self.dino_loss(student_patch_output, teacher_patch_output)

        # Combine the two losses
        # Combine the two losses
        total_loss = cls_loss + self.patch_loss_weight * patch_loss
        return total_loss

