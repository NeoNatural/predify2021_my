from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable, Sequence


SUPER_CATEGORIES = ("animals", "clothing", "food", "plants", "vehicles")

# ImageNet-1K is not organized into the five superordinate categories used by
# Wynn et al. Animals and food occupy contiguous portions of the canonical
# label order; clothing and vehicles require an explicit, audited subset.
_CLOTHING_LABELS = frozenset(
    {
        399, 400, 411, 433, 439, 443, 445, 451, 452, 457, 459, 461,
        465, 474, 490, 496, 501, 502, 514, 515, 518, 552, 560, 568,
        578, 601, 608, 610, 614, 617, 630, 638, 639, 643, 652, 655,
        658, 667, 676, 678, 689, 691, 697, 715, 735, 770, 774, 775,
        793, 796, 806, 808, 824, 834, 836, 837, 841, 842, 869, 885,
        887, 903, 906,
    }
)
_VEHICLE_LABELS = frozenset(
    {
        403, 404, 405, 407, 408, 436, 444, 450, 466, 468, 472, 484,
        510, 511, 537, 547, 554, 555, 561, 565, 569, 573, 575, 576,
        586, 603, 609, 612, 625, 627, 628, 654, 656, 660, 661, 665,
        670, 671, 675, 690, 694, 705, 717, 734, 751, 757, 779, 780,
        802, 803, 812, 814, 817, 820, 829, 833, 847, 864, 866, 867,
        870, 871, 874, 880, 895, 914,
    }
)


def imagenet_supercategory(label: int) -> str | None:
    label = int(label)
    if 0 <= label <= 397:
        return "animals"
    if label in _CLOTHING_LABELS:
        return "clothing"
    if 924 <= label <= 969:
        return "food"
    if 984 <= label <= 990:
        return "plants"
    if label in _VEHICLE_LABELS:
        return "vehicles"
    return None


@dataclass(frozen=True)
class PaperTarget:
    dataset_index: int
    imagenet_label: int
    supercategory: str
    category_index: int


@dataclass(frozen=True)
class PaperTrialPlan:
    target: PaperTarget
    preview_indices: tuple[int, int, int, int]
    preview_labels: tuple[int, int, int, int]
    preview_categories: tuple[str, str, str, str]
    one_back_slot: int
    two_back_slot: int


def _category_pools(dataset_targets: Sequence[int]) -> dict[str, dict[int, list[int]]]:
    pools: dict[str, dict[int, list[int]]] = {name: {} for name in SUPER_CATEGORIES}
    for dataset_index, raw_label in enumerate(dataset_targets):
        label = int(raw_label)
        category = imagenet_supercategory(label)
        if category is not None:
            pools[category].setdefault(label, []).append(dataset_index)
    return pools


def select_balanced_paper_targets(
    dataset_targets: Sequence[int],
    *,
    targets_per_category: int,
    seed: int,
    max_targets: int | None = None,
) -> list[PaperTarget]:
    """Select balanced targets while spreading samples across ImageNet classes."""
    if targets_per_category < 1:
        raise ValueError("targets_per_category must be positive")
    pools = _category_pools(dataset_targets)
    rng = random.Random(seed)

    requested_total = targets_per_category * len(SUPER_CATEGORIES)
    total = requested_total if max_targets is None else min(requested_total, max_targets)
    if total < len(SUPER_CATEGORIES):
        raise ValueError("paper experiments require at least one target per category")
    quotas = {name: total // len(SUPER_CATEGORIES) for name in SUPER_CATEGORIES}
    for name in SUPER_CATEGORIES[: total % len(SUPER_CATEGORIES)]:
        quotas[name] += 1

    selected_by_category: dict[str, list[PaperTarget]] = {
        name: [] for name in SUPER_CATEGORIES
    }
    for category_index, category in enumerate(SUPER_CATEGORIES):
        label_to_indices = pools[category]
        if not label_to_indices:
            raise ValueError(f"ImageNet contains no mapped samples for {category}")
        labels = list(label_to_indices)
        rng.shuffle(labels)
        shuffled_indices: dict[int, list[int]] = {}
        for label in labels:
            values = list(label_to_indices[label])
            rng.shuffle(values)
            shuffled_indices[label] = values

        count = quotas[category]
        for item_index in range(count):
            label = labels[item_index % len(labels)]
            cycle = item_index // len(labels)
            values = shuffled_indices[label]
            dataset_index = values[cycle % len(values)]
            selected_by_category[category].append(
                PaperTarget(dataset_index, label, category, category_index)
            )

    selected: list[PaperTarget] = []
    for within_category_index in range(max(quotas.values())):
        for category in SUPER_CATEGORIES:
            values = selected_by_category[category]
            if within_category_index < len(values):
                selected.append(values[within_category_index])
    return selected


def build_paper_trial_plans(
    targets: Sequence[PaperTarget],
    dataset_targets: Sequence[int],
    *,
    seed: int,
) -> list[PaperTrialPlan]:
    pools = _category_pools(dataset_targets)
    category_records: dict[str, list[tuple[int, int]]] = {}
    for category, label_map in pools.items():
        category_records[category] = [
            (dataset_index, label)
            for label, indices in label_map.items()
            for dataset_index in indices
        ]

    plans: list[PaperTrialPlan] = []
    for ordinal, target in enumerate(targets):
        rng = random.Random(seed + ordinal * 1009 + target.dataset_index)
        distractors: list[tuple[int, int, str]] = []
        for category in SUPER_CATEGORIES:
            if category == target.supercategory:
                continue
            dataset_index, label = rng.choice(category_records[category])
            distractors.append((dataset_index, label, category))
        rng.shuffle(distractors)
        # The exact prime is balanced left/right within each temporal position.
        two_back_slot = ordinal % 2
        one_back_slot = 2 + (ordinal % 2)
        plans.append(
            PaperTrialPlan(
                target=target,
                preview_indices=tuple(value[0] for value in distractors),
                preview_labels=tuple(value[1] for value in distractors),
                preview_categories=tuple(value[2] for value in distractors),
                one_back_slot=one_back_slot,
                two_back_slot=two_back_slot,
            )
        )
    return plans


def materialize_preview_indices(
    plan: PaperTrialPlan,
    *,
    condition: str,
    prime_identity: str,
) -> tuple[tuple[int, int], tuple[int, int], dict[str, Any]]:
    if condition not in {"no_prime", "one_back", "two_back"}:
        raise ValueError(f"unknown paper condition: {condition}")
    if prime_identity not in {"same", "different"}:
        raise ValueError(f"unknown prime identity: {prime_identity}")

    values = list(plan.preview_indices)
    prime_slot: int | None = None
    if condition == "one_back":
        prime_slot = plan.one_back_slot
    elif condition == "two_back":
        prime_slot = plan.two_back_slot
    control_index = None if prime_slot is None else values[prime_slot]
    if prime_slot is not None and prime_identity == "same":
        values[prime_slot] = plan.target.dataset_index

    metadata = {
        "condition": condition,
        "prime_identity": prime_identity,
        "prime_slot": prime_slot,
        "prime_display": None if prime_slot is None else 1 + prime_slot // 2,
        "prime_side": None if prime_slot is None else ("left" if prime_slot % 2 == 0 else "right"),
        "prime_dataset_index": None if prime_slot is None else values[prime_slot],
        "different_control_index": control_index,
    }
    return (values[0], values[1]), (values[2], values[3]), metadata


def make_paper_canvas(
    images: Iterable[tuple[Any, str]],
    *,
    object_scale: float,
    horizontal_offset: float,
    background_value: float,
):
    """Place normalized ImageNet tensors on a screen-like 224×224 canvas."""
    import torch
    import torch.nn.functional as F

    values = list(images)
    if not values:
        raise ValueError("at least one image is required")
    first = values[0][0]
    if first.dim() != 3 or first.shape[0] != 3:
        raise ValueError("paper canvases require normalized CHW RGB tensors")
    _, height, width = first.shape
    tile_size = max(1, int(round(height * object_scale)))
    mean = torch.tensor([0.485, 0.456, 0.406], device=first.device, dtype=first.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=first.device, dtype=first.dtype)
    normalized_background = ((background_value - mean) / std).view(3, 1, 1)
    canvas = normalized_background.expand(3, height, width).clone()

    centers = {
        "left": (width / 2 - horizontal_offset * width, height / 2),
        "center": (width / 2, height / 2),
        "right": (width / 2 + horizontal_offset * width, height / 2),
    }
    for image, position in values:
        if position not in centers:
            raise ValueError(f"unknown paper canvas position: {position}")
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(tile_size, tile_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        center_x, center_y = centers[position]
        x0 = int(round(center_x - tile_size / 2))
        y0 = int(round(center_y - tile_size / 2))
        x1 = x0 + tile_size
        y1 = y0 + tile_size
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise ValueError("paper object placement exceeds the canvas")
        canvas[:, y0:y1, x0:x1] = resized
    return canvas
