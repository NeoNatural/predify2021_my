# -*- coding: utf-8 -*-
import random
import torch
import torch.nn.functional as F


DEGRADATION_MODES = {"none", "predify_noise", "fractal_occlusion"}


def adjust_contrast_per_image(img: torch.Tensor, contrast_scale: float):
    mean = img.mean(dim=(1, 2), keepdim=True)
    std = img.std(dim=(1, 2), keepdim=True) + 1e-6
    return mean + (img - mean) * contrast_scale


def gaussian_blur(img: torch.Tensor, sigma: float):
    if sigma <= 0:
        return img
    radius = int(3 * sigma)
    size = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=img.device)
    kernel1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel1d /= kernel1d.sum()
    kernel2d = kernel1d[:, None] * kernel1d[None, :]
    kernel2d = kernel2d.expand(img.shape[0], 1, size, size)
    return F.conv2d(img.unsqueeze(0), kernel2d, padding=radius, groups=img.shape[0]).squeeze(0)


def add_predify_noise(img: torch.Tensor, sigma: float, seed: int = 0):
    """Reproduce Predify's additive N(0, sigma) noise on a normalized tensor."""
    if sigma <= 0:
        return img
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.normal(
        mean=0.0,
        std=float(sigma),
        size=tuple(img.shape),
        generator=generator,
    )
    return img + noise.to(device=img.device, dtype=img.dtype)


def add_fractal_occlusion(
    img: torch.Tensor,
    intensity: float,
    *,
    octaves: int = 3,
    seed: int = 0,
):
    """Apply edge-guided fractal black occlusion without additive noise."""
    if intensity <= 0:
        return img
    if img.dim() not in {3, 4}:
        raise ValueError("fractal occlusion expects a CHW or BCHW tensor")
    if octaves < 1:
        raise ValueError("fractal octaves must be at least 1")

    batched = img.unsqueeze(0) if img.dim() == 3 else img
    batch_size, channels, height, width = batched.shape
    if channels != 3:
        raise ValueError("fractal occlusion expects three ImageNet-normalized channels")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    fractal_mask = torch.zeros(
        (batch_size, 1, height, width), device=img.device, dtype=img.dtype
    )
    for octave in range(octaves):
        coarse_size = 2 ** (octave + 2)
        amplitude = 0.5 ** octave
        layer = torch.rand(
            (batch_size, 1, coarse_size, coarse_size),
            generator=generator,
        ).to(device=img.device, dtype=img.dtype)
        layer = F.interpolate(
            layer,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        fractal_mask += layer * amplitude
    fractal_mask /= 1.5

    edge_kernel = torch.tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
        device=img.device,
        dtype=img.dtype,
    ).view(1, 1, 3, 3)
    edge_map = F.conv2d(batched.mean(dim=1, keepdim=True), edge_kernel, padding=1).abs()
    edge_min = edge_map.amin(dim=(-2, -1), keepdim=True)
    edge_max = edge_map.amax(dim=(-2, -1), keepdim=True)
    edge_map = (edge_map - edge_min) / (edge_max - edge_min + 1e-5)

    combined_signal = fractal_mask * (1.0 + edge_map * 2.5)
    threshold = 1.4 - (float(intensity) ** 0.7 * 1.25)
    binary_mask = (combined_signal > threshold).to(dtype=img.dtype)
    blur_size = 5 if intensity < 0.5 else 9
    soft_mask = F.avg_pool2d(
        binary_mask,
        kernel_size=blur_size,
        stride=1,
        padding=blur_size // 2,
    )

    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype)
    normalized_black = (-mean / std).view(1, 3, 1, 1)
    occluded = (1.0 - soft_mask) * batched + soft_mask * normalized_black
    return occluded.squeeze(0) if img.dim() == 3 else occluded


def apply_input_degradation(
    img: torch.Tensor,
    mode: str = "none",
    level: float = 0.0,
    *,
    seed: int = 0,
    fractal_octaves: int = 3,
):
    if mode not in DEGRADATION_MODES:
        raise ValueError(f"unknown degradation mode: {mode}")
    if mode == "none":
        return img
    if mode == "predify_noise":
        return add_predify_noise(img, sigma=level, seed=seed)
    return add_fractal_occlusion(
        img,
        intensity=level,
        octaves=fractal_octaves,
        seed=seed,
    )


def make_half_field_canvas(
    img_priming: torch.Tensor,
    img_other: torch.Tensor,
    priming_on_left: bool = None,
    canvas_scale: float = 2.0,
    contrast_scale: float = 0.5,
    degradation_mode: str = "none",
    degradation_level: float = 0.0,
    degradation_seed: int = 0,
    fractal_octaves: int = 3,
    blur_sigma: float = 1.0,
    downsample_after_blur: bool = True,
    dtype=torch.float32,
):
    device = img_priming.device
    if priming_on_left is None:
        priming_on_left = random.random() > 0.5

    img_priming = adjust_contrast_per_image(img_priming.to(device), contrast_scale)
    img_other = adjust_contrast_per_image(img_other.to(device), contrast_scale)

    C, H, W = img_priming.shape
    new_width = int(round(W * canvas_scale))
    half_w = new_width // 2
    canvas = torch.zeros((C, H, new_width), dtype=dtype, device=device)

    def _paste_horiz(dst, img_src, x0, x1):
        dst_w = x1 - x0
        src_w = img_src.shape[2]
        if dst_w >= src_w:
            dst_start = x0 + (dst_w - src_w) // 2
            src_start = 0
            copy_w = src_w
        else:
            src_start = (src_w - dst_w) // 2
            dst_start = x0
            copy_w = dst_w
        dst[:, :, dst_start:dst_start + copy_w] = img_src[:, :, src_start:src_start + copy_w]

    if priming_on_left:
        _paste_horiz(canvas, img_priming, 0, half_w)
        _paste_horiz(canvas, img_other, new_width - half_w, new_width)
    else:
        _paste_horiz(canvas, img_other, 0, half_w)
        _paste_horiz(canvas, img_priming, new_width - half_w, new_width)

    # Input degradation and the historical prime blur are deliberately separate.
    # The blur is applied second so the two effects can be stacked when requested.
    canvas = apply_input_degradation(
        canvas,
        mode=degradation_mode,
        level=degradation_level,
        seed=degradation_seed,
        fractal_octaves=fractal_octaves,
    )
    canvas = gaussian_blur(canvas, sigma=blur_sigma)

    if downsample_after_blur:
        canvas = F.interpolate(
            canvas.unsqueeze(0),
            scale_factor=0.5,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    meta = {
        "priming_on_left": priming_on_left,
        "canvas_size": canvas.shape[1:],
        "mode": "half-field",
        "degradation_mode": degradation_mode,
        "degradation_level": float(degradation_level),
    }
    return canvas, meta


def make_half_field_stimulus_v5(
    dataset,
    idx,
    trial_idx,
    priming=True,
    idx_log_list=None,
    canvas_scale: float = 2.0,
    contrast_scale: float = 0.5,
    degradation_mode: str = "none",
    degradation_level: float = 0.0,
    degradation_seed: int = 0,
    fractal_octaves: int = 3,
    blur_sigma: float = 1.0,
    downsample_after_blur: bool = True,
    dtype=torch.float32,
    device=None,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if trial_idx % 2 == 1:
        img, y = dataset[idx]
        img = img.to(device)
        img = apply_input_degradation(
            img,
            mode=degradation_mode,
            level=degradation_level,
            seed=degradation_seed,
            fractal_octaves=fractal_octaves,
        )
        return img, {
            "y": int(y),
            "mode": "single",
            "degradation_mode": degradation_mode,
            "degradation_level": float(degradation_level),
        }

    if priming:
        priming_idx = idx
    else:
        if idx_log_list is None:
            raise ValueError("idx_log_list is required when priming=False.")
        priming_idx = idx_log_list[(trial_idx + 2) % len(idx_log_list)]

    if idx_log_list is not None:
        other_idx = idx_log_list[(trial_idx + 4) % len(idx_log_list)]
    else:
        other_idx = random.randint(0, len(dataset) - 1)
        if other_idx == priming_idx:
            other_idx = (other_idx + 1) % len(dataset)

    img_priming, y_priming = dataset[priming_idx]
    img_other, y_other = dataset[other_idx]

    canvas, meta = make_half_field_canvas(
        img_priming.to(device),
        img_other.to(device),
        priming_on_left=None,
        canvas_scale=canvas_scale,
        contrast_scale=contrast_scale,
        degradation_mode=degradation_mode,
        degradation_level=degradation_level,
        degradation_seed=degradation_seed,
        fractal_octaves=fractal_octaves,
        blur_sigma=blur_sigma,
        downsample_after_blur=downsample_after_blur,
        dtype=dtype,
    )
    meta.update({
        "y_priming": int(y_priming),
        "y_other": int(y_other),
    })
    return canvas, meta


def make_foveated_full_stimulus(
    dataset,
    idx,
    trial_idx,
    priming=True,
    idx_log_list=None,
    degradation_mode: str = "none",
    degradation_level: float = 0.0,
    degradation_seed: int = 0,
    fractal_octaves: int = 3,
    device=None,
    **_unused,
):
    """Build a full-resolution centered prime or the unchanged odd target.

    No half-field canvas, contrast reduction, blur, or downsampling is applied.
    The same input degradation code and seed convention as the historical path
    are used so odd targets remain exactly paired across layouts.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if trial_idx % 2 == 1:
        image, label = dataset[idx]
        image = apply_input_degradation(
            image.to(device),
            mode=degradation_mode,
            level=degradation_level,
            seed=degradation_seed,
            fractal_octaves=fractal_octaves,
        )
        return image, {
            "y": int(label),
            "mode": "single",
            "prime_layout": "foveated_full",
            "degradation_mode": degradation_mode,
            "degradation_level": float(degradation_level),
        }

    if priming:
        priming_idx = idx
    else:
        if idx_log_list is None:
            raise ValueError("idx_log_list is required when priming=False.")
        priming_idx = idx_log_list[(trial_idx + 2) % len(idx_log_list)]

    image, label = dataset[priming_idx]
    image = apply_input_degradation(
        image.to(device),
        mode=degradation_mode,
        level=degradation_level,
        seed=degradation_seed,
        fractal_octaves=fractal_octaves,
    )
    return image, {
        "y_priming": int(label),
        "mode": "foveated-full",
        "prime_layout": "foveated_full",
        "degradation_mode": degradation_mode,
        "degradation_level": float(degradation_level),
    }


def make_priming_stimulus(*args, prime_layout="parafoveal_half_field", **kwargs):
    if prime_layout == "parafoveal_half_field":
        image, metadata = make_half_field_stimulus_v5(*args, **kwargs)
        metadata.setdefault("prime_layout", "parafoveal_half_field")
        return image, metadata
    if prime_layout == "foveated_full":
        return make_foveated_full_stimulus(*args, **kwargs)
    raise ValueError(f"unknown prime layout: {prime_layout}")
