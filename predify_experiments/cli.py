from __future__ import annotations

import argparse
import os
from pathlib import Path
from pprint import pprint
from typing import Sequence

from .config import load_experiment_config


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "hebb_vgg_priming.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the configurable Predify + Hebbian VGG priming experiment."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--fc-weight-source", choices=("torchvision", "finetuned"))
    parser.add_argument("--fc-checkpoint", type=Path)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate paths and print the resolved configuration without loading PyTorch",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_experiment_config(args.config).with_overrides(
            imagenet_root=args.imagenet_root or os.environ.get("PREDIFY_IMAGENET_ROOT"),
            output_dir=args.output_dir or os.environ.get("PREDIFY_OUTPUT_DIR"),
            fc_checkpoint=args.fc_checkpoint,
            fc_weight_source=args.fc_weight_source,
            device=args.device,
            max_samples=args.max_samples,
        )
        config.validate_for_run()
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    if args.check_config:
        pprint(config)
        return 0

    # Heavy scientific dependencies are intentionally imported only for a run.
    from .hebb_vgg_priming import run_experiment

    run_experiment(config)
    return 0
