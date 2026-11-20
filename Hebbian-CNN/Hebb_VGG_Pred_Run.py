"""Compatibility entry point for the configurable Hebbian VGG experiment.

Prefer running::

    python -m predify_experiments --config configs/hebb_vgg_predify_noise.toml
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predify_experiments.cli import main


if __name__ == "__main__":
    default_args = [
        "--config",
        str(PROJECT_ROOT / "configs" / "hebb_vgg_predify_noise.toml"),
    ]
    raise SystemExit(main([*default_args, *sys.argv[1:]]))
