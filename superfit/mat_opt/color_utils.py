
import torch.nn.functional as F
import torch as th

from sysl.shader.utils.texture import gather_textures
from superfit.utils.config import AlgorithmConfig as AlgConf

def srgb_to_linear(x):
    print("srgb_to_linear")
    return th.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


# --- sRGB (gamma) -> linear RGB ---
def _srgb_to_linear(x: th.Tensor) -> th.Tensor:
    a = 0.055
    return th.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)).clamp_min(0)**2.4)

def old_tv_loss_flat(cur_expr, reduction='mean'):
    """
    textures_flat: (B, 3, N)  where N = H*W
    Returns total variation loss (scalar)
    """
    textures = gather_textures(cur_expr)
    textures = th.stack(textures, dim=0)

    # Horizontal and vertical gradients
    dh = th.abs(textures[:, 1:, :, :] - textures[:, :-1, :, :])
    dw = th.abs(textures[:, :, 1:, :] - textures[:, :, :-1, :])

    if reduction == 'mean':
        return dh.mean() + dw.mean()
    else:
        return dh.sum() + dw.sum()

def tv_loss_flat(cur_expr, reduction="mean", eps=1e-6):
    """
    Toroidal TV-min using 8 neighbors (4 dirs: down/right + 2 diagonals),
    isotropic per-pixel L2 across channels.

    Returns a scalar.
    """
    textures = gather_textures(cur_expr)          # list of (H,W,C) or (B?,H,W,C)
    textures = th.stack(textures, dim=0)          # (B,H,W,C)

    # neighbors (toroidal wrap)
    dh  = textures - th.roll(textures, shifts=-1,        dims=1)       # down
    dw  = textures - th.roll(textures, shifts=-1,        dims=2)       # right
    dd1 = textures - th.roll(textures, shifts=(-1, -1),  dims=(1, 2))  # down-right
    dd2 = textures - th.roll(textures, shifts=(-1,  1),  dims=(1, 2))  # down-left

    # sum over channels (C) -> (B,H,W)
    sq = (dh * dh).sum(dim=3) + (dw * dw).sum(dim=3) + (dd1 * dd1).sum(dim=3) + (dd2 * dd2).sum(dim=3)

    tv = th.sqrt(sq + eps)  # eps prevents infinite gradients at 0

    if reduction == "mean":
        return tv.mean()
    elif reduction == "sum":
        return tv.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
        import torch as th
import torch.nn.functional as F

def laplacian_loss_flat(cur_expr, reduction="mean", circular=True, mode="charbonnier", eps=1e-6):
    """
    Laplacian smoothness (second-derivative penalty) per channel.

    textures from gather_textures assumed (H,W,C) each; stacked -> (B,H,W,C).
    mode: "l2" | "l1" | "charbonnier"
    circular: toroidal wrap like your TV (no boundary artifacts).
    """
    textures = th.stack(gather_textures(cur_expr), dim=0)  # (B,H,W,C)

    # to (B,C,H,W)
    x = textures.permute(0, 3, 1, 2).contiguous()
    B, C, H, W = x.shape

    # 3x3 Laplacian kernel
    k = th.tensor([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]], dtype=x.dtype, device=x.device)
    k = k.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # (C,1,3,3) depthwise

    if circular:
        x = F.pad(x, (1, 1, 1, 1), mode="circular")
        lap = F.conv2d(x, k, padding=0, groups=C)  # (B,C,H,W)
    else:
        lap = F.conv2d(x, k, padding=1, groups=C)

    if mode == "l2":
        loss = (lap * lap).mean()
    elif mode == "l1":
        loss = lap.abs().mean()
    elif mode == "charbonnier":
        loss = th.sqrt(lap * lap + eps).mean()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if reduction == "mean":
        return loss
    elif reduction == "sum":
        # sum over all elements (roughly scales with image size)
        return loss * lap.numel()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def diff_tv_loss_flat(cur_expr, reduction='mean'):
    """
    Compute total variation loss with toroidal (wrap-around) neighbors.

    textures: (B, H, W, C)
    Returns scalar (mean or sum)
    """
    textures = gather_textures(cur_expr)     # list of tensors
    textures = th.stack(textures, dim=0)     # (B, H, W, C)

    # --- Toroidal (periodic) differences ---
    # Vertical: difference between row i and row (i+1 mod H)
    dh = th.abs(textures[:, 1:, :, :] - textures[:, :-1, :, :])
    dh_wrap = th.abs(textures[:, :, 0:1, :] - textures[:, :, -1:, :])   # last row → first row
    dh = th.cat([dh, dh_wrap], dim=1)

    # Horizontal: difference between col j and col (j+1 mod W)
    dw = th.abs(textures[:, :, 1:, :] - textures[:, :, :-1, :])
    dw_wrap = th.abs(textures[:, :, 0:1, :] - textures[:, :, -1:, :])    # last col → first col
    dw = th.cat([dw, dw_wrap], dim=3)

    if reduction == 'mean':
        return dh.mean() + dw.mean()
    else:
        return dh.sum() + dw.sum()

# --- linear RGB -> OKLab (perceptual) ---
# From Björn Ottosson’s OKLab reference

def _linear_rgb_to_oklab(rgb: th.Tensor) -> th.Tensor:
    # rgb: (..., 3), linear
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    l_ = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
    m_ = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
    s_ = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b

    l_c = l_.clamp_min(1e-10).pow(1/3)
    m_c = m_.clamp_min(1e-10).pow(1/3)
    s_c = s_.clamp_min(1e-10).pow(1/3)

    L = 0.2104542553*l_c + 0.7936177850*m_c - 0.0040720468*s_c
    a = 1.9779984951*l_c - 2.4285922050*m_c + 0.4505937099*s_c
    b = 0.0259040371*l_c + 0.7827717662*m_c - 0.8086757660*s_c
    return th.stack([L, a, b], dim=-1)

def _rgb_to_oklab_srgb(rgb_01: th.Tensor) -> th.Tensor:
    return _linear_rgb_to_oklab(_srgb_to_linear(rgb_01.clamp(0,1)))

def material_loss_rgb_oklab_huber(
    pred: th.Tensor,            # (N, 5) in [0,1], channels [R,G,B,M,Rgh]
    target: th.Tensor,          # (N, 5) in [0,1]
    *,
    w_rgb: float = 1.0,
    w_m: float = 1.0,
    w_rgh: float = 1.0,
    huber_delta_m: float = 0.05,
    huber_delta_r: float = 0.05,
    use_charbonnier_rgb: bool = False,
    eps_charb: float = 1e-3
) -> th.Tensor:
    assert pred.shape[-1] == 5 and target.shape[-1] == 5, "Expect (N,5) RGBMR"

    # Split
    pred_rgb = pred[..., :3]
    tgt_rgb  = target[..., :3]
    pred_m   = pred[..., 3:4]
    tgt_m    = target[..., 3:4]
    pred_r   = pred[..., 4:5]
    tgt_r    = target[..., 4:5]

    # Color loss in OKLab (perceptual)
    pred_ok = _rgb_to_oklab_srgb(pred_rgb)
    tgt_ok  = _rgb_to_oklab_srgb(tgt_rgb)
    if use_charbonnier_rgb:
        # Charbonnier (smooth L1) in OKLab
        diff = pred_ok - tgt_ok
        rgb_loss = th.sqrt(diff.pow(2).sum(dim=-1) + eps_charb**2).mean()
    else:
        # L1 in OKLab
        rgb_loss = F.l1_loss(pred_ok, tgt_ok)

    # Metallic & Roughness: Huber / smooth-L1
    m_loss = F.huber_loss(pred_m, tgt_m, delta=huber_delta_m)
    r_loss = F.huber_loss(pred_r, tgt_r, delta=huber_delta_r)

    return w_rgb * rgb_loss + w_m * m_loss + w_rgh * r_loss
