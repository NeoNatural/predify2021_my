from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Sequence

from .config import ExperimentConfig, load_experiment_config
from .hebb_vgg_priming import _degradation_suffix, _resolve_device, _seed_everything


VANILLA_SCHEMA_VERSION = 1
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "vanilla_vgg16_degradation_benchmark.toml"
)


def vanilla_result_filename(mode: str, level: float) -> str:
    suffix = _degradation_suffix(mode, level) or "_clean"
    return f"Vanilla_VGG16{suffix}.pckl"


def topk_correct(logits: Any, labels: Any, k: int) -> Any:
    import numpy as np

    logits = np.asarray(logits)
    labels = np.asarray(labels)
    k = min(k, logits.shape[-1])
    top_indices = np.argpartition(logits, -k, axis=-1)[:, -k:]
    return np.any(top_indices == labels[:, None], axis=1)


def _add_legacy_module_path(project_root: Path) -> None:
    legacy_dir = str(project_root / "Hebbian-CNN")
    if legacy_dir not in sys.path:
        sys.path.insert(0, legacy_dir)


def run_vanilla_benchmark(
    config: ExperimentConfig,
    *,
    batch_size: int = 32,
    overwrite: bool = False,
) -> list[Path]:
    config.validate_for_run()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    _add_legacy_module_path(config.paths.project_root)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.datasets import ImageNet
    from torchvision.models import VGG16_Weights, vgg16

    from Hebb_Priming_Utils import apply_input_degradation

    device = _resolve_device(config.runtime.device, torch)
    _seed_everything(config.runtime.seed, np, torch)
    weights = VGG16_Weights.IMAGENET1K_V1
    dataset = ImageNet(
        str(config.paths.imagenet_root), "val", transform=weights.transforms()
    )
    model = vgg16(weights=weights).eval().to(device)

    with config.paths.sample_file.open("rb") as handle:
        sample_dict = pickle.load(handle)
    all_indices = list(sample_dict["out_list"])
    schedule_count = len(all_indices)
    if config.runtime.max_samples is not None:
        schedule_count = min(schedule_count, config.runtime.max_samples)
    target_trials = list(range(1, schedule_count, 2))

    class TargetDataset(Dataset):
        def __init__(self, mode: str, level: float) -> None:
            self.mode = mode
            self.level = level

        def __len__(self) -> int:
            return len(target_trials)

        def __getitem__(self, position: int):
            trial_index = target_trials[position]
            dataset_index = all_indices[trial_index]
            image, label = dataset[dataset_index]
            degradation_seed = config.stimulus.degradation_seed
            if self.mode == "fractal_occlusion":
                degradation_seed += trial_index
            image = apply_input_degradation(
                image,
                mode=self.mode,
                level=self.level,
                seed=degradation_seed,
                fractal_octaves=config.stimulus.fractal_octaves,
            )
            return image, int(label), int(dataset_index), int(trial_index)

    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    started = time.time()
    for mode, level in config.stimulus.degradation_conditions():
        output_file = config.paths.output_dir / vanilla_result_filename(mode, level)
        if output_file.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing benchmark result: {output_file}"
            )
        loader = DataLoader(
            TargetDataset(mode, level),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        logits_parts: list[Any] = []
        labels_parts: list[Any] = []
        indices_parts: list[Any] = []
        trials_parts: list[Any] = []
        condition_started = time.time()
        with torch.inference_mode():
            for images, labels, dataset_indices, trial_indices in loader:
                logits_parts.append(model(images.to(device, non_blocking=True)).cpu())
                labels_parts.append(labels)
                indices_parts.append(dataset_indices)
                trials_parts.append(trial_indices)

        logits = torch.cat(logits_parts).numpy().astype(np.float32, copy=False)
        labels = torch.cat(labels_parts).numpy().astype(np.int64, copy=False)
        dataset_indices = torch.cat(indices_parts).numpy().astype(np.int64, copy=False)
        trial_indices = torch.cat(trials_parts).numpy().astype(np.int64, copy=False)
        top1 = topk_correct(logits, labels, 1)
        top5 = topk_correct(logits, labels, 5)
        elapsed = time.time() - condition_started
        result = {
            "schema_version": VANILLA_SCHEMA_VERSION,
            "benchmark_type": "torchvision_vgg16_imagenet1k_v1",
            "run_metadata": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_file": str(config.source_file),
                "device": str(device),
                "seed": config.runtime.seed,
                "degradation_mode": mode,
                "degradation_level": level,
                "degradation_seed": config.stimulus.degradation_seed,
                "schedule_parity": "odd_targets_only",
                "elapsed_seconds": elapsed,
                "config": asdict(config),
            },
            "idx_log_list": dataset_indices,
            "trial_indices": trial_indices,
            "y_list": labels,
            "logits": logits,
            "acc_top_1": top1,
            "acc_top_5": top5,
            "acc_result": (float(top1.mean()), float(top5.mean())),
        }
        with output_file.open("wb") as handle:
            pickle.dump(result, handle)
        output_paths.append(output_file)
        print(
            f"[vanilla] degradation={mode} level={level:g} "
            f"accuracy={result['acc_result']} elapsed={elapsed:.1f}s"
        )
        print(f"[vanilla] wrote {output_file}")

    print(f"[vanilla] completed in {time.time() - started:.1f}s")
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark torchvision VGG16 on the scheduled odd targets."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_experiment_config(args.config).with_overrides(
            imagenet_root=args.imagenet_root or os.environ.get("PREDIFY_IMAGENET_ROOT"),
            output_dir=args.output_dir,
            device=args.device,
            max_samples=args.max_samples,
        )
        config.validate_for_run()
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    if args.check_config:
        print(config)
        return 0
    run_vanilla_benchmark(
        config,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
