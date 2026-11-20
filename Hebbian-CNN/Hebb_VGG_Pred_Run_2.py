"""Compatibility entry point using the previously fine-tuned FC checkpoint."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predify_experiments.cli import main


if __name__ == "__main__":
    default_args = [
        "--config",
        str(PROJECT_ROOT / "configs" / "hebb_vgg_priming.toml"),
        "--fc-weight-source",
        "finetuned",
        "--fc-checkpoint",
        str(PROJECT_ROOT / "Hebbian-CNN" / "Weights" / "fc_finetune_20260307_035854.pth"),
    ]
    raise SystemExit(main([*default_args, *sys.argv[1:]]))
