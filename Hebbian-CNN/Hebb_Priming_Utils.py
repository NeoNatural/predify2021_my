# -*- coding: utf-8 -*-
import random
import torch
import torch.nn.functional as F


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


def make_half_field_canvas(
    img_priming: torch.Tensor,
    img_other: torch.Tensor,
    priming_on_left: bool = None,
    canvas_scale: float = 2.0,
    contrast_scale: float = 0.5,
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
        return img, {"y": int(y), "mode": "single"}

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
        blur_sigma=blur_sigma,
        downsample_after_blur=downsample_after_blur,
        dtype=dtype,
    )
    meta.update({
        "y_priming": int(y_priming),
        "y_other": int(y_other),
    })
    return canvas, meta
