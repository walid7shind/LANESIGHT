import torch
import torch.nn.functional as F


def _homography_from_4pts(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Compute homography H such that dst ~ H @ src.

    src: (B,4,2) float (x,y) in pixels
    dst: (B,4,2) float (u,v) in pixels
    returns: (B,3,3) float

    Uses DLT with h33 fixed to 1.
    """
    if src.ndim != 3 or dst.ndim != 3 or src.shape != dst.shape or src.shape[1:] != (4, 2):
        raise ValueError(f"Expected src/dst of shape (B,4,2); got {tuple(src.shape)} and {tuple(dst.shape)}")

    B = src.shape[0]
    device = src.device
    dtype = src.dtype

    x = src[:, :, 0]
    y = src[:, :, 1]
    u = dst[:, :, 0]
    v = dst[:, :, 1]

    A = torch.zeros((B, 8, 8), device=device, dtype=dtype)
    b = torch.zeros((B, 8, 1), device=device, dtype=dtype)

    # For each correspondence i, add two rows.
    # [ x y 1 0 0 0 -u*x -u*y ] [h11 h12 h13 h21 h22 h23 h31 h32]^T = u
    # [ 0 0 0 x y 1 -v*x -v*y ] [...] = v
    for i in range(4):
        xi = x[:, i]
        yi = y[:, i]
        ui = u[:, i]
        vi = v[:, i]

        r0 = 2 * i
        r1 = r0 + 1

        A[:, r0, 0] = xi
        A[:, r0, 1] = yi
        A[:, r0, 2] = 1
        A[:, r0, 6] = -ui * xi
        A[:, r0, 7] = -ui * yi
        b[:, r0, 0] = ui

        A[:, r1, 3] = xi
        A[:, r1, 4] = yi
        A[:, r1, 5] = 1
        A[:, r1, 6] = -vi * xi
        A[:, r1, 7] = -vi * yi
        b[:, r1, 0] = vi

    h = torch.linalg.solve(A, b).squeeze(-1)  # (B,8)

    H = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    H[:, 0, 0] = h[:, 0]
    H[:, 0, 1] = h[:, 1]
    H[:, 0, 2] = h[:, 2]
    H[:, 1, 0] = h[:, 3]
    H[:, 1, 1] = h[:, 4]
    H[:, 1, 2] = h[:, 5]
    H[:, 2, 0] = h[:, 6]
    H[:, 2, 1] = h[:, 7]
    H[:, 2, 2] = 1

    return H


@torch.no_grad()
def mild_perspective_warp(
    imgs: torch.Tensor,
    masks: torch.Tensor,
    *,
    p: float = 0.25,
    max_dx_frac: float = 0.02,
    max_dy_frac: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a mild random perspective warp to BOTH imgs and masks.

    This is a GEOMETRIC augmentation, so it must be applied identically to the image and mask.

    imgs: (B,3,H,W) float in [0,1]
    masks: (B,H,W) long/int (e.g., {0,1} or ignore_index)

    Returns (imgs_warped, masks_warped).

    Notes:
    - Uses bilinear sampling for imgs, nearest for masks.
    - Uses zero padding for both so labels stay conservative near borders.
    - Designed to be mild; avoid topology changes.
    """
    if p <= 0.0:
        return imgs, masks

    if imgs.ndim != 4 or imgs.size(1) != 3:
        raise ValueError(f"Expected imgs of shape (B,3,H,W), got {tuple(imgs.shape)}")
    if masks.ndim != 3:
        raise ValueError(f"Expected masks of shape (B,H,W), got {tuple(masks.shape)}")
    if imgs.shape[0] != masks.shape[0] or imgs.shape[-2:] != masks.shape[-2:]:
        raise ValueError(
            f"imgs/masks batch or spatial mismatch: imgs={tuple(imgs.shape)}, masks={tuple(masks.shape)}"
        )

    B, _, H, W = imgs.shape
    device = imgs.device

    # Per-sample gate: when false, dst==src (identity warp).
    gate = (torch.rand((B, 1, 1), device=device) < p)

    # Base corners in pixel coordinates.
    src = torch.tensor(
        [[0.0, 0.0], [W - 1.0, 0.0], [W - 1.0, H - 1.0], [0.0, H - 1.0]],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0).repeat(B, 1, 1)

    max_dx = float(max_dx_frac) * float(W)
    max_dy = float(max_dy_frac) * float(H)

    jitter = torch.empty((B, 4, 2), device=device, dtype=torch.float32)
    jitter[:, :, 0].uniform_(-max_dx, max_dx)
    jitter[:, :, 1].uniform_(-max_dy, max_dy)

    # Apply jitter only where gated on.
    jitter = jitter * gate.to(dtype=torch.float32)

    dst = src + jitter
    dst[:, :, 0].clamp_(0.0, W - 1.0)
    dst[:, :, 1].clamp_(0.0, H - 1.0)

    H_mat = _homography_from_4pts(src, dst)  # input -> output
    H_inv = torch.linalg.inv(H_mat)          # output -> input

    # Build output grid in pixel coords.
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)

    p_out = torch.stack([xs, ys, ones], dim=-1).view(-1, 3).t()  # (3, HW)
    p_out = p_out.unsqueeze(0).expand(B, -1, -1)                 # (B, 3, HW)

    p_in = H_inv @ p_out
    x_in = p_in[:, 0, :] / (p_in[:, 2, :].clamp_min(1e-6))
    y_in = p_in[:, 1, :] / (p_in[:, 2, :].clamp_min(1e-6))

    # Normalize for grid_sample with align_corners=True.
    x_norm = (x_in / (W - 1.0)) * 2.0 - 1.0
    y_norm = (y_in / (H - 1.0)) * 2.0 - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1).view(B, H, W, 2)

    imgs_w = F.grid_sample(
        imgs,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    masks_f = masks.to(dtype=torch.float32).unsqueeze(1)  # (B,1,H,W)
    masks_w = F.grid_sample(
        masks_f,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(1)

    # Nearest sampling preserves discrete labels; cast back.
    masks_w = masks_w.to(dtype=masks.dtype)

    return imgs_w, masks_w
