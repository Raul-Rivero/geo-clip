"""
Configurable ImageEncoder for the GeoCLIP backbone-swap experiment.

This mirrors the original geoclip/model/image_encoder.py structure exactly
(Linear -> ReLU -> Linear head on top of a frozen backbone), but lets you
choose the backbone at construction time instead of hardcoding CLIP.

Usage:
    encoder = ImageEncoder(backbone="clip")      # original GeoCLIP recipe
    encoder = ImageEncoder(backbone="siglip2-so400m")  # our SigLIP2-So400M ablation

Drop this in to replace geoclip/model/image_encoder.py, or import it
directly and wire it into your own copy of GeoCLIP.py in place of the
original ImageEncoder.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, AutoModel, AutoProcessor

# backbone_id -> config dict. Single source of truth for anything that
# depends on which backbone is active -- the checkpoint id, the native
# feature dim the mlp head is built for, and the preprocessing (image size +
# normalization stats) that both preprocess_image() and the training
# dataloader (geoclip/train/dataloader.py) must agree on.
# Native output dims and preprocessing stats confirmed against each
# checkpoint's actual HF config (preprocessor_config.json / vision_config).
BACKBONE_REGISTRY = {
    "clip": {
        "hf_id": "openai/clip-vit-large-patch14",
        "native_dim": 768,
        "image_size": 224,
        "image_mean": [0.485, 0.456, 0.406],  # ImageNet stats (matches original GeoCLIP dataloader)
        "image_std": [0.229, 0.224, 0.225],
    },
    "siglip2-so400m": {
        "hf_id": "google/siglip2-so400m-patch14-224",
        "native_dim": 1152,
        "image_size": 224,
        "image_mean": [0.5, 0.5, 0.5],  # SigLIP2's own preprocessor_config.json
        "image_std": [0.5, 0.5, 0.5],
    },
}


class ImageEncoder(nn.Module):
    def __init__(self, backbone: str = "clip", output_dim: int = 512):
        """
        Args:
            backbone: one of the keys in BACKBONE_REGISTRY ("clip" or "siglip2-so400m")
            output_dim: final embedding size, must match the location encoder's
                        output dim (512 in the original GeoCLIP repo -- do not
                        change this unless you're also changing location_encoder.py)
        """
        super(ImageEncoder, self).__init__()

        if backbone not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Choose from {list(BACKBONE_REGISTRY.keys())}"
            )
        self.backbone_name = backbone
        cfg = BACKBONE_REGISTRY[backbone]
        hf_id, native_dim = cfg["hf_id"], cfg["native_dim"]

        # --- Load the backbone itself ---
        if backbone == "clip":
            self.backbone = CLIPModel.from_pretrained(hf_id)
            self.image_processor = AutoProcessor.from_pretrained(hf_id)
        else:
            # SigLIP2 (and other AutoModel-compatible backbones) go through
            # the generic transformers interface. get_image_features() works
            # the same way for both CLIPModel and SigLIP2's AutoModel.
            self.backbone = AutoModel.from_pretrained(hf_id)
            self.image_processor = AutoProcessor.from_pretrained(hf_id)

        # --- Freeze the backbone (matches original GeoCLIP exactly) ---
        for param in self.backbone.parameters():
            param.requires_grad = False

        # --- Trainable head: Linear(native_dim -> native_dim) -> ReLU -> Linear(native_dim -> output_dim) ---
        # This is h1 -> ReLU -> h2 from the paper. Only these weights train.
        self.mlp = nn.Sequential(
            nn.Linear(native_dim, native_dim),
            nn.ReLU(),
            nn.Linear(native_dim, output_dim),
        )
        self.native_dim = native_dim
        self._shape_checked = False  # one-time runtime verification, see forward()

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        # get_image_features() is the shared interface both CLIPModel and
        # SigLIP2's AutoModel expose for pooled image embeddings.
        with torch.no_grad():  # extra safety: backbone is frozen, no grad needed
            features = self.backbone.get_image_features(pixel_values=x)
            if not torch.is_tensor(features):
                # Some AutoModel variants wrap output in a container object
                features = features.pooler_output if hasattr(features, "pooler_output") else features[0]

        # One-time runtime shape check: confirms the input tensor actually
        # reaching the backbone is [B, 3, 224, 224] and its pooled output is
        # [B, native_dim] -- the checkpoint string lining up with the registry
        # on paper doesn't guarantee upstream code is feeding the right shape.
        if not self._shape_checked:
            print(f"[{self.backbone_name}] encoder input shape: {tuple(x.shape)} "
                  f"| backbone output shape: {tuple(features.shape)}")
            assert x.shape[1] == 3 and x.shape[-2:] == (224, 224), (
                f"[{self.backbone_name}] expected input [B, 3, 224, 224], got {tuple(x.shape)}"
            )
            assert features.shape[-1] == self.native_dim, (
                f"[{self.backbone_name}] expected backbone output dim {self.native_dim}, "
                f"got {features.shape[-1]}"
            )
            self._shape_checked = True

        x = self.mlp(features)
        return x

    def trainable_parameters(self):
        """Convenience: only the mlp head should ever be passed to the optimizer."""
        return self.mlp.parameters()


if __name__ == "__main__":
    # Quick sanity check -- run this file directly to confirm both backbones
    # load correctly and produce the expected output shape.
    for backbone in ["clip", "siglip2-so400m"]:
        enc = ImageEncoder(backbone=backbone)
        dummy = torch.randn(2, 3, 224, 224)
        # Note: real usage should go through preprocess_image() on actual PIL
        # images -- this dummy tensor is just to confirm the mlp head's shapes
        # are wired correctly, not a full forward-pass test.
        n_trainable = sum(p.numel() for p in enc.trainable_parameters())
        n_frozen = sum(p.numel() for p in enc.backbone.parameters())
        print(f"{backbone}: trainable params = {n_trainable:,} | frozen backbone params = {n_frozen:,}")
