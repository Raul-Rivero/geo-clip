"""
Driver script for the GeoCLIP backbone-swap experiment (CLIP vs SigLIP2).
Verified directly against the real geo-clip repo source
(https://github.com/VicenteVivan/geo-clip) -- not guessed.

REQUIRED PATCH to your local clone's geoclip/model/GeoCLIP.py:
    Change:
        def __init__(self, from_pretrained=True, queue_size=4096):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.image_encoder = ImageEncoder()
            self.location_encoder = LocationEncoder()
    To:
        def __init__(self, from_pretrained=True, queue_size=4096, image_encoder=None):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.image_encoder = image_encoder if image_encoder is not None else ImageEncoder()
            self.location_encoder = LocationEncoder(from_pretrained=from_pretrained)

    Why: the original code always builds the default CLIP-based ImageEncoder with
    no way to inject a different one, AND always loads pretrained LocationEncoder
    weights regardless of the outer from_pretrained flag (a real bug for anyone
    trying to train from scratch). This patch fixes both -- verified against the
    actual source, not guessed.

Usage:
    python train_driver.py --backbone clip --data_csv train.csv --image_dir ./images --val_csv val.csv --epochs 5
    python train_driver.py --backbone siglip2-so400m --data_csv train.csv --image_dir ./images --val_csv val.csv --epochs 5
"""

import argparse
import os
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from geoclip.model.GeoCLIP import GeoCLIP
from geoclip.train.dataloader import GeoDataLoader, img_train_transform, img_val_transform
from geoclip.train.train import train as train_one_epoch
from geoclip.train.eval import eval_images

from image_encoder_configurable import ImageEncoder as ConfigurableImageEncoder


def build_model(backbone: str, device: str):
    """
    from_pretrained=False is critical here -- we want a genuinely from-scratch
    location encoder + heads for a fair, controlled backbone comparison, not
    fine-tuning on top of the paper's already-converged checkpoint.
    """
    custom_encoder = ConfigurableImageEncoder(backbone=backbone, output_dim=512)
    model = GeoCLIP(from_pretrained=False, image_encoder=custom_encoder)
    model = model.to(device)
    return model


def get_optimizer_and_scheduler(model):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Optimizer will update {n_trainable:,} trainable parameters")

    optimizer = optim.Adam(trainable_params, lr=3e-5, weight_decay=1e-6)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.87)
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["clip", "siglip2-so400m"], required=True)
    parser.add_argument("--data_csv", required=True,
                         help="Training CSV with IMG_FILE/LAT/LON columns")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--val_csv", default=None,
                         help="Optional validation CSV; if omitted, skips eval between epochs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256,
                         help="NOTE: GeoCLIP's dequeue_and_enqueue requires "
                              "queue_size % batch_size == 0 -- default queue_size "
                              "is 4096, so batch sizes like 512, 256, 128, 64 all "
                              "divide evenly. Avoid odd batch sizes.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Backbone: {args.backbone} | Device: {args.device} | Batch size: {args.batch_size}")

    # --- Data: GeoDataLoader is a Dataset, must be wrapped in a real DataLoader ---
    train_dataset = GeoDataLoader(
        dataset_file=args.data_csv,
        dataset_folder=args.image_dir,
        transform=img_train_transform(args.backbone),
    )
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,  # avoid a final partial batch breaking the queue math
        generator=g,
    )
    print(f"Train set: {len(train_dataset):,} images, {len(train_loader):,} batches/epoch")

    val_loader = None
    if args.val_csv:
        val_dataset = GeoDataLoader(
            dataset_file=args.val_csv,
            dataset_folder=args.image_dir,
            transform=img_val_transform(args.backbone),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print(f"Val set: {len(val_dataset):,} images")

    # --- Model, optimizer, scheduler ---
    model = build_model(args.backbone, args.device)
    optimizer, scheduler = get_optimizer_and_scheduler(model)

    # --- Training loop ---
    # NOTE: the repo's own train() function calls scheduler.step() internally
    # at the end of each epoch when a scheduler is passed in -- do NOT also
    # call scheduler.step() out here, or the LR will decay twice as fast.
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_one_epoch(
            train_dataloader=train_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            batch_size=args.batch_size,
            device=args.device,
            scheduler=scheduler,
        )
        elapsed = time.time() - start
        current_lr = scheduler.get_last_lr()[0]
        print(f"[epoch {epoch}/{args.epochs}] lr={current_lr:.2e} time={elapsed:.1f}s")

        if val_loader is not None:
            eval_images(val_loader, model, device=args.device)

        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.backbone}_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "backbone": args.backbone,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        print(f"  saved checkpoint: {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
