import torch


@torch.no_grad()
def photometric_augment(
    imgs: torch.Tensor,
    *,
    p: float = 0.9,
    brightness: float = 0.20,
    contrast: float = 0.25,
    channel_scale: float = 0.20,
    grayscale_mix: float = 0.25,
    gamma: float = 0.35,
    noise: float = 0.03,
) -> torch.Tensor:
    """GPU-only photometric augmentation for lane segmentation.

    Applies only appearance changes (no geometry), so labels remain valid.

    Assumptions:
    - imgs is a float tensor shaped (B,3,H,W)
    - values are in [0,1]

    Notes on performance:
    - All randomness is generated on the same device as imgs.
    - Avoids CPU↔GPU sync (no .item(), no Python branching per sample).
    """
    if p <= 0.0:
        return imgs

    if imgs.ndim != 4 or imgs.size(1) != 3:
        raise ValueError(f"Expected imgs of shape (B,3,H,W), got {tuple(imgs.shape)}")

    device = imgs.device
    dtype = imgs.dtype

    b = imgs.size(0)

    # Decide which samples are augmented at all (per-sample gate).
    gate = (torch.rand((b, 1, 1, 1), device=device) < p).to(dtype=dtype)

    x = imgs

    # 1) Brightness (additive)
    if brightness > 0:
        delta = (torch.rand((b, 1, 1, 1), device=device, dtype=dtype) * 2.0 - 1.0) * brightness
        x = x + gate * delta

    # 2) Contrast (around per-image mean)
    if contrast > 0:
        c = 1.0 + (torch.rand((b, 1, 1, 1), device=device, dtype=dtype) * 2.0 - 1.0) * contrast
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * (gate * c + (1.0 - gate)) + mean

    # 3) Channel scaling (breaks color shortcuts while preserving structure)
    if channel_scale > 0:
        s = 1.0 + (torch.rand((b, 3, 1, 1), device=device, dtype=dtype) * 2.0 - 1.0) * channel_scale
        x = x * (gate * s + (1.0 - gate))

    # 4) Grayscale mix (simulates low-chroma / washed lighting)
    if grayscale_mix > 0:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=device, dtype=dtype).view(1, 3, 1, 1)
        gray = (x * w).sum(dim=1, keepdim=True)
        # With probability `grayscale_mix`, mix toward grayscale by a random strength.
        # This yields better coverage: sometimes none, sometimes partial, sometimes near-full grayscale.
        apply = (torch.rand((b, 1, 1, 1), device=device) < grayscale_mix).to(dtype=dtype)
        alpha = torch.rand((b, 1, 1, 1), device=device, dtype=dtype)
        mix = gate * apply * alpha
        x = x * (1.0 - mix) + gray * mix

    # 5) Gamma (camera response / exposure curve)
    if gamma > 0:
        x = torch.clamp(x, 0.0, 1.0)
        # sample gamma in [1-gamma, 1+gamma], clipped to a sane positive range
        g = 1.0 + (torch.rand((b, 1, 1, 1), device=device, dtype=dtype) * 2.0 - 1.0) * gamma
        g = torch.clamp(g, 0.5, 2.0)
        eps = torch.tensor(1e-6, device=device, dtype=dtype)
        x = torch.pow(torch.clamp(x, min=eps), gate * g + (1.0 - gate) * 1.0)

    # 6) Gaussian noise (sensor / compression noise)
    if noise > 0:
        sigma = torch.rand((b, 1, 1, 1), device=device, dtype=dtype) * noise
        x = x + gate * (torch.randn_like(x) * sigma)

    return torch.clamp(x, 0.0, 1.0)
