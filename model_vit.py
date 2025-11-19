from typing import List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.stat_scores import StatScores
from transformers import AutoConfig, AutoModelForImageClassification
from transformers.optimization import get_cosine_schedule_with_warmup
from loss import SoftTargetCrossEntropy, Mixup


MODEL_DICT = {
    "vit-b16-224-in21k": "google/vit-base-patch16-224-in21k",
    "vit-b32-224-in21k": "google/vit-base-patch32-224-in21k",
    "vit-l32-224-in21k": "google/vit-large-patch32-224-in21k",
    "vit-l16-224-in21k": "google/vit-large-patch16-224-in21k",
    "vit-h14-224-in21k": "google/vit-huge-patch14-224-in21k",
    # timm/vit_small_patch16_224.augreg_in1k
    "vit-s16-224-in21k": "timm/vit_small_patch16_224.augreg_in21k",
    "vit-b16-224": "google/vit-base-patch16-224",
    "vit-l16-224": "google/vit-large-patch16-224",
    "vit-b16-384": "google/vit-base-patch16-384",
    "vit-b32-384": "google/vit-base-patch32-384",
    "vit-l16-384": "google/vit-large-patch16-384",
    "vit-l32-384": "google/vit-large-patch32-384",
    "vit-b16-224-dino": "facebook/dino-vitb16",
    "vit-b8-224-dino": "facebook/dino-vitb8",
    "vit-s16-224-dino": "facebook/dino-vits16",
    "vit-s8-224-dino": "facebook/dino-vits8",

    
    'dinov2-large': 'facebook/dinov2-large',
    'dinov2-base': 'facebook/dinov2-base',
    'dinov2-small': 'facebook/dinov2-small',

    "beit-b16-224-in21k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-l16-224-in21k": "microsoft/beit-large-patch16-224-pt22k-ft22k",
    'vit_tiny': 'WinKawaks/vit-tiny-patch16-224',
}


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vit-s16-224-dino",
        optimizer: str = "sgd",
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        warmup_steps: int = 0,
        n_classes: int = 10,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        mix_prob: float = 1.0,
        label_smoothing: float = 0.0,
        image_size: int = 224,
        weights: Optional[str] = None,
        training_mode: str = "full",
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_target_modules: List[str] = ["query", "value"],
        lora_dropout: float = 0.0,
        lora_bias: str = "none",
        from_scratch: bool = False,
        split: int = 1,
        device: str = "cuda",
        pretrained_model = None,
    ):
        """Classification Model

        Args:
            model_name: Name of model checkpoint. List found in src/model.py
            optimizer: Name of optimizer. One of [adam, adamw, sgd]
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler. One of [cosine, none]
            warmup_steps: Number of warmup steps
            n_classes: Number of target class
            mixup_alpha: Mixup alpha value
            cutmix_alpha: Cutmix alpha value
            mix_prob: Probability of applying mixup or cutmix (applies when mixup_alpha and/or
                cutmix_alpha are >0)
            label_smoothing: Amount of label smoothing
            image_size: Size of input images
            weights: Path of checkpoint to load weights from (e.g when resuming after linear probing)
            training_mode: Fine-tuning mode. One of ["full", "linear", "lora"]
            lora_r: Dimension of LoRA update matrices
            lora_alpha: LoRA scaling factor
            lora_target_modules: Names of the modules to apply LoRA to
            lora_dropout: Dropout probability for LoRA layers
            lora_bias: Whether to train biases during LoRA. One of ['none', 'all' or 'lora_only']
            from_scratch: Initialize network with random weights instead of a pretrained checkpoint
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.n_classes = n_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.label_smoothing = label_smoothing
        self.image_size = image_size
        self.weights = weights
        self.training_mode = training_mode
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.from_scratch = from_scratch

        self._already_pretrained = False if pretrained_model is None else True


        # Initialize network
        try:
            model_path = MODEL_DICT[self.model_name]
        except:
            raise ValueError(
                f"{model_name} is not an available model. Should be one of {[k for k in MODEL_DICT.keys()]}"
            )

        if self.from_scratch:
            # Initialize with random weights
            config = AutoConfig.from_pretrained(model_path)
            config.image_size = self.image_size
            self.net = AutoModelForImageClassification.from_config(config)
            self.net.classifier = torch.nn.Linear(config.hidden_size, self.n_classes)
        else:
            # Initialize with pretrained weights
            self.net = AutoModelForImageClassification.from_pretrained(
                model_path,
                num_labels=self.n_classes,
                ignore_mismatched_sizes=True,
                #image_size=self.image_size,
            )

        # If pretrained model is provided, load it
        if pretrained_model is not None:
            print(f"Loaded pretrained model from {pretrained_model}")
            self.load_state_dict(torch.load(pretrained_model), strict=True)

        # Move the model to the specified device
        self.net.to(device)
            

        # Load checkpoint weights
        if self.weights:
            print(f"Loaded weights from {self.weights}")
            ckpt = torch.load(self.weights)["state_dict"]

            # Remove prefix from key names
            new_state_dict = {}
            for k, v in ckpt.items():
                if k.startswith("net"):
                    k = k.replace("net" + ".", "")
                    new_state_dict[k] = v

            self.net.load_state_dict(new_state_dict, strict=True)

        # Prepare model depending on fine-tuning mode
        if self.training_mode == "linear":
            # Freeze transformer layers and keep classifier unfrozen
            for name, param in self.net.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        elif self.training_mode == "lora":
            # Wrap in LoRA model
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias=self.lora_bias,
                modules_to_save=["classifier"],
            )
            self.net = get_peft_model(self.net, config)
        elif self.training_mode == "block":
            
            config = AutoConfig.from_pretrained(model_path)
            config.image_size = self.image_size

            print('Number of Layers: ', config.num_hidden_layers)

            ckpt = self.net.state_dict()
            output, selected_layers = block_expansion(ckpt,
                                                      split,
                                                      config.num_hidden_layers)
            
            print('Selected Layers: ', selected_layers)

            config.num_hidden_layers += len(selected_layers)
            config.num_labels = self.n_classes

            self.net = AutoModelForImageClassification.from_config(config)
            self.net.load_state_dict(output)

            self.net.requires_grad_(False)
            for n, p in self.net.named_parameters():
                for idx in selected_layers:
                    if 'layer.' + str(idx) + '.' in n:
                        p.requires_grad_(True)

            self.net.classifier = torch.nn.Linear(config.hidden_size, self.n_classes)

        elif self.training_mode == "full":
            pass  # Keep all layers unfrozen
        else:
            raise ValueError(
                f"{self.training_mode} is not an available fine-tuning mode. Should be one of ['full', 'linear', 'lora']"
            )

        # Define metrics
        self.train_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                "acc_top5": Accuracy(
                    num_classes=self.n_classes,
                    task="multiclass",
                    top_k=min(5, self.n_classes),
                ),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                "acc_top5": Accuracy(
                    num_classes=self.n_classes,
                    task="multiclass",
                    top_k=min(5, self.n_classes),
                ),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                "acc_top5": Accuracy(
                    num_classes=self.n_classes,
                    task="multiclass",
                    top_k=min(5, self.n_classes),
                ),
                "stats": StatScores(
                    task="multiclass", average=None, num_classes=self.n_classes
                ),
            }
        )

        

        # Define loss
        self.loss_fn = SoftTargetCrossEntropy()

        # Define regularizers
        self.mixup = Mixup(
            mixup_alpha=self.mixup_alpha,
            cutmix_alpha=self.cutmix_alpha,
            prob=self.mix_prob,
            label_smoothing=self.label_smoothing,
            num_classes=self.n_classes,
        )

        self.test_metric_outputs = []

    #def forward(self, x):
    #    return self.net(pixel_values=x).logits

    def forward(self, x, return_cls_token: bool = False, return_hidden_states: bool = False, return_attention_rollout: bool = False) -> torch.Tensor:
            if return_cls_token or return_hidden_states:
        
                outputs = self.net(pixel_values=x, output_hidden_states=return_cls_token or return_hidden_states)
                
                if return_cls_token:
                    # Get the last hidden states (sequence of token embeddings)
                    # Most ViT models return these in `hidden_states`, including the CLS token and patch tokens
                    # Return the last layer's CLS token
                    return outputs.logits, outputs.hidden_states[-1][:, 0]
                
                if return_hidden_states:
                    # Get the last hidden states (sequence of token embeddings)
                    # Most ViT models return these in `hidden_states`, including the CLS token and patch tokens
                    # Return the last layer's CLS token and patch tokens
                    return outputs.logits, outputs.hidden_states[-1][:, 0], outputs.hidden_states[-1][:, 1:]
                
                
                
                return outputs.logits
            else:
                # If no additional outputs are needed, just return the logits
                return self.net(pixel_values=x).logits
            
    def get_attention_rollout(self,
                            x: torch.Tensor,
                            layer_start: int = 0,
                            layer_end: Optional[int] = None) -> torch.Tensor:
        """
        Compute Attention Rollout from CLS to patches.

        Args:
            x:        Input image tensor, shape [B, C, H, W].
            layer_start: First layer index to include (default: 0).
            layer_end:   Last layer index (inclusive). If None, use all layers.

        Returns:
            rollout_map: Tensor [B, H_patch, W_patch] giving CLS→patch attention.
        """
        # 1. Forward with attentions
        outputs = self.net.vit(pixel_values=x, output_attentions=True, return_dict=True) \
                if hasattr(self.net, 'vit') \
                else self.net.dinov2(pixel_values=x, output_attentions=True, return_dict=True)
        all_attn = outputs.attentions  # List of length L: [B, heads, tokens, tokens]

        # 2. Determine layers to include
        L = len(all_attn)
        end = layer_end + 1 if layer_end is not None else L
        attn_layers = all_attn[layer_start:end]

        # 3. Build joint attention
        joint = None
        for A in attn_layers:
            # A: [B, H, T, T]
            # average over heads → [B, T, T]
            A_mean = A.mean(dim=1)
            # add identity for residual connection
            I = torch.eye(A_mean.size(-1), device=A_mean.device).unsqueeze(0)
            A_res = A_mean + I
            # normalize rows
            A_norm = A_res / A_res.sum(dim=-1, keepdim=True)
            # accumulate
            joint = A_norm if joint is None else A_norm.bmm(joint)

        # 4. Extract CLS→patch rollout
        B, T, _ = joint.size()
        # first token is CLS, rest are P patches; P = T-1
        cls_to_patches = joint[:, 0, 1:]  # [B, P]

        # 5. Reshape into spatial map
        P = cls_to_patches.size(-1)
        S = int(P**0.5)
        rollout_map = cls_to_patches.reshape(B, S, S)

        return rollout_map

            
    def get_cls_attention_map(self, x: torch.Tensor, layer: int = -1, head: int = None) -> torch.Tensor:
        """
        Returns the self-attention from the [CLS] token to all patches from a given layer.

        Args:
            x: Input tensor [B, C, H, W]
            layer: Transformer layer to extract attention from (default: last)
            head: Optional head index to select one head (averaged if None)

        Returns:
            attention map [B, num_patches_h, num_patches_w]
        """
        #with torch.no_grad():

        
        if hasattr(self.net, 'vit'):
            outputs = self.net.vit(
                pixel_values=x,
                output_attentions=True,
                return_dict=True
            )
        elif hasattr(self.net, 'dinov2'):
            outputs = self.net.dinov2(
                pixel_values=x,
                output_attentions=True,
                return_dict=True
            )
        elif hasattr(self.net, 'timm_model'):
            outputs = self.net.timm_model(
                
                output_attentions=True,
                return_dict=True
            )
        else:
            raise ValueError("Model does not have a 'vit' or 'dinov2' attribute for attention extraction.")
        attentions = outputs.attentions  # List of [B, heads, tokens, tokens]
        cls_attn = attentions[layer][:, :, 0, 1:]  # [B, heads, patches]
        if head is None:
            attn = cls_attn.mean(1)  # [B, patches]
        else:
            attn = cls_attn[:, head, :]  # [B, patches]

        # Reshape to spatial grid
        B, P = attn.shape
        H = W = int(P**0.5)  # assumes square grid
        
        attn_map = attn.reshape(B, H, W)

        

        return attn_map
    
    



    def shared_step(self, batch, mode="train"):
        x, y = batch

        if mode == "train":
            # Only converts targets to one-hot if no label smoothing, mixup or cutmix is set
            x, y = self.mixup(x, y)
        else:
            y = F.one_hot(y, num_classes=self.n_classes).float()

        # Pass through network
        pred = self(x)
        loss = self.loss_fn(pred, y)

        # Get accuracy
        metrics = getattr(self, f"{mode}_metrics")(pred, y.argmax(1))

        # Log
        self.log(f"{mode}_loss", loss, on_epoch=True)
        for k, v in metrics.items():
            if len(v.size()) == 0:
                self.log(f"{mode}_{k.lower()}", v, on_epoch=True)

        if mode == "test":
            self.test_metric_outputs.append(metrics["stats"])

        return loss

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return self.shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self.shared_step(batch, "val")

    def test_step(self, batch, _):
        return self.shared_step(batch, "test")

    def on_test_epoch_end(self):
        """Save per-class accuracies to csv"""
        # Aggregate all batch stats
        combined_stats = torch.sum(
            torch.stack(self.test_metric_outputs, dim=-1), dim=-1
        )

        # Calculate accuracy per class
        per_class_acc = []
        for tp, _, _, _, sup in combined_stats:
            acc = tp / sup
            per_class_acc.append((acc.item(), sup.item()))

        # Save to csv
        df = pd.DataFrame(per_class_acc, columns=["acc", "n"])
        df.to_csv("per-class-acc-test.csv")
        print("Saved per-class results in per-class-acc-test.csv")

    def configure_optimizers(self):
        # Initialize optimizer
        if self.optimizer == "adam":
            optimizer = Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # Initialize learning rate scheduler
        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                num_warmup_steps=self.warmup_steps,
            )
        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 1)
        else:
            raise ValueError(
                f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }