import os
import sys
import time
import torch
from torch.utils.data import DataLoader

from data.culane_dataset import CULaneDataset
from models.hybrid_vit_unet import HybridViTUNet
from utils.losses import SegLoss


# -------------------------------------------------
# Device picker (safe for RTX 5060)
# -------------------------------------------------
def _pick_device(requested: str | None = None) -> tuple[str, bool]:
    requested = (requested or "").strip().lower() or None
    device = requested or ("cuda" if torch.cuda.is_available() else "cpu")

    if device != "cuda" or not torch.cuda.is_available():
        return "cpu", False

    try:
        major, minor = torch.cuda.get_device_capability(0)
        gpu_arch = f"sm_{major}{minor}"
        arch_list = torch.cuda.get_arch_list()
        if arch_list and gpu_arch not in arch_list:
            print(f"[WARN] CUDA arch {gpu_arch} unsupported → CPU fallback")
            return "cpu", False
    except Exception:
        pass

    try:
        x = torch.randn(1, 3, 16, 16, device="cuda")
        w = torch.randn(8, 3, 3, 3, device="cuda")
        torch.nn.functional.conv2d(x, w)
    except RuntimeError as e:
        if "no kernel image is available" in str(e):
            print("[WARN] CUDA kernel missing → CPU fallback")
            return "cpu", False
        raise

    return "cuda", True


# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
def seed_everything(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------
# Training
# -------------------------------------------------
def train(
    root,
    train_list,
    val_list,
    img_size=(288, 800),
    epochs=30,
    bs=6,
    lr=3e-4,
    num_workers=4,
    device=None,
):
    seed_everything(42)

    device, use_cuda = _pick_device(device)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Image size: {img_size}")
    print(f"[INFO] Batch size: {bs}")

    # ---------------- Dataset ----------------
    tr_ds = CULaneDataset(root, train_list, img_size=img_size, augment=True)
    va_ds = CULaneDataset(root, val_list, img_size=img_size, augment=False)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=True,
    )

    va_loader = DataLoader(
        va_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    # ---------------- Model ----------------
    model = HybridViTUNet(
        num_classes=2,          # BINARY
        img_size_hw=img_size,
        base=48
    ).to(device)

    # ---------------- Loss (BINARY + WEIGHTED) ----------------
    loss_fn = SegLoss(
        ce_weight=1.0,
        dice_weight=1.0,
        ignore_index=255,
        class_weights=(0.3, 3.0)   # background, lane
    )
    loss_fn = loss_fn.to(device)

    # ---------------- Optimizer ----------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-2
    )

    # ---------------- Scheduler ----------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    # ---------------- AMP ----------------
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    # ---------------- Checkpointing ----------------
    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ---------- TRAIN ----------
        model.train()
        train_loss = 0.0

        for imgs, masks in tr_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
                logits = model(imgs)
                loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()

            # Gradient clipping (important with ViT)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(tr_loader)

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, masks in va_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
                    logits = model(imgs)
                    loss = loss_fn(logits, masks)

                val_loss += loss.item()

        val_loss /= len(va_loader)

        scheduler.step()
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"lr={lr_now:.2e} | "
            f"time={dt:.1f}s"
        )

        # ---------- SAVE BEST ----------
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "img_size": img_size,
                    "num_classes": 2,
                },
                "checkpoints/best.pt",
            )
            print("  ✔ saved checkpoints/best.pt")

    print("[DONE] Training finished.")


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    data_root = os.path.join(script_dir, "data")

    train_list = os.path.join(data_root, "list", "train_gt - Copy.txt")
    val_list   = os.path.join(data_root, "list", "val_gt - Copy.txt")

    train(
        root=data_root,
        train_list=train_list,
        val_list=val_list,
        epochs=30,
        bs=6,
        lr=3e-4,
    )
