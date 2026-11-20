from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import pickle
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

from .config import ExperimentConfig
from .paper_stimuli import (
    SUPER_CATEGORIES,
    build_paper_trial_plans,
    make_paper_canvas,
    materialize_preview_indices,
    select_balanced_paper_targets,
)
from .sequence import reset_hebbian_state, run_recurrent_sequence


PAPER_RESULT_SCHEMA_VERSION = 1
RECOVERY_METRIC_NAMES = ("cosine", "normalized_l2", "mse")
REPRESENTATION_LAYER_NAMES = ("conv4", "conv5", "fc1", "fc2")


def _add_legacy_module_path(project_root: Path) -> None:
    legacy_dir = project_root / "Hebbian-CNN"
    legacy_str = str(legacy_dir)
    if legacy_str not in sys.path:
        sys.path.insert(0, legacy_str)


def _resolve_device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _seed_everything(seed: int, np: Any, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _degradation_suffix(mode: str, level: float) -> str:
    if mode == "none":
        return "clean"
    value = format(level, ".12g").replace("-", "m").replace(".", "p")
    label = "sigma" if mode == "predify_noise" else "intensity"
    return f"{mode}_{label}{value}"


def paper_degradation_seed(
    base_seed: int,
    degradation_mode: str,
    *,
    target_offset: int,
    ordinal: int,
) -> int:
    """Keep stochastic degradation paired across disjoint target subsets."""
    if degradation_mode == "fractal_occlusion":
        return int(base_seed) + int(target_offset) + int(ordinal)
    return int(base_seed)


def paper_result_filename(
    *,
    protocol: str,
    condition: str,
    prime_identity: str,
    memory_carrier: str,
    degradation_mode: str,
    degradation_level: float,
) -> str:
    return (
        f"paper_{protocol}_{condition}_{prime_identity}_{memory_carrier}_"
        f"{_degradation_suffix(degradation_mode, degradation_level)}.pckl"
    )


class _FeatureRecorder:
    def __init__(self, model: Any, hebbian_layers: list[Any], torch: Any) -> None:
        self.torch = torch
        self.cache: dict[str, Any] = {}
        modules = {
            "fc2": hebbian_layers[0],
            "fc1": hebbian_layers[1],
            "conv5": hebbian_layers[2],
            "conv4": hebbian_layers[3],
        }
        self.handles = [
            module.register_forward_hook(self._hook(name))
            for name, module in modules.items()
        ]

    def _hook(self, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = output.detach()
            if value.dim() > 2:
                value = value.mean(dim=tuple(range(2, value.dim())))
            if value.dim() == 1:
                value = value.unsqueeze(0)
            self.cache[name] = value

        return hook

    def snapshot(self) -> dict[str, Any]:
        missing = set(REPRESENTATION_LAYER_NAMES) - set(self.cache)
        if missing:
            raise RuntimeError(f"missing representation hooks: {sorted(missing)}")
        return {
            name: self.cache[name].squeeze(0).to(dtype=self.torch.float16).cpu().numpy()
            for name in REPRESENTATION_LAYER_NAMES
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@contextmanager
def _defer_hebbian_updates(layers: Iterable[Any]):
    states: list[tuple[Any, bool]] = []
    for layer in layers:
        if hasattr(layer, "update_enabled"):
            states.append((layer, bool(layer.update_enabled)))
            layer.update_enabled = False
    try:
        yield
    finally:
        for layer, state in states:
            layer.update_enabled = state


def _recovery_metrics(value: Any, reference: Any, torch: Any) -> Any:
    value = value.detach().float().flatten(start_dim=1)
    reference = reference.detach().float().flatten(start_dim=1)
    eps = 1e-12
    value_norm = torch.linalg.vector_norm(value, dim=1, keepdim=True).clamp_min(eps)
    reference_norm = torch.linalg.vector_norm(reference, dim=1, keepdim=True).clamp_min(eps)
    cosine = ((value * reference).sum(dim=1) / (value_norm[:, 0] * reference_norm[:, 0]))
    normalized_l2 = torch.linalg.vector_norm(
        value / value_norm - reference / reference_norm, dim=1
    )
    mse = torch.mean((value - reference) ** 2, dim=1)
    return torch.stack((cosine, normalized_l2, mse), dim=1).squeeze(0)


def _capture_pcoder_recovery(
    model: Any,
    reference_model: Any,
    clean_input: Any,
    torch: Any,
) -> tuple[Any, Any, Any]:
    rep_values = []
    prediction_values = []
    internal_errors = []
    for index in range(1, 6):
        pcoder = getattr(model, f"pcoder{index}")
        reference_pcoder = getattr(reference_model, f"pcoder{index}")
        rep_values.append(_recovery_metrics(pcoder.rep, reference_pcoder.rep, torch))
        clean_lower_reference = (
            clean_input
            if index == 1
            else getattr(reference_model, f"pcoder{index - 1}").rep
        )
        prediction_values.append(
            _recovery_metrics(pcoder.prd, clean_lower_reference, torch)
        )
        internal_errors.append(pcoder.prediction_error.detach().float())
    return (
        torch.stack(rep_values).cpu().numpy(),
        torch.stack(prediction_values).cpu().numpy(),
        torch.stack(internal_errors).cpu().numpy(),
    )


def _cuda_rng_state(torch: Any, device: Any) -> Any:
    if device.type != "cuda":
        return None
    return torch.cuda.get_rng_state(device)


def _restore_rng_state(torch: Any, device: Any, cpu_state: Any, cuda_state: Any) -> None:
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _run_target(
    *,
    model: Any,
    reference_model: Any | None,
    image: Any,
    clean_image: Any,
    steps: int,
    preserve_pcoder_state: bool,
    hebbian_layers: list[Any],
    reference_hebbian_layers: list[Any] | None,
    recorder: _FeatureRecorder,
    torch: Any,
    device: Any,
) -> tuple[list[Any], dict[str, list[Any]], list[Any], list[Any], list[Any]]:
    logits: list[Any] = []
    features: dict[str, list[Any]] = {name: [] for name in REPRESENTATION_LAYER_NAMES}
    representation_recovery: list[Any] = []
    prediction_recovery: list[Any] = []
    internal_errors: list[Any] = []
    batched = image.unsqueeze(0).to(device)
    clean_batched = clean_image.unsqueeze(0).to(device)

    if reference_model is None:
        def after_step(_timestep: int, output: Any) -> None:
            logits.append(output.detach().cpu())
            snapshot = recorder.snapshot()
            for name, value in snapshot.items():
                features[name].append(value)

        run_recurrent_sequence(
            model,
            batched,
            steps,
            hebbian_layers=hebbian_layers,
            commit_update=False,
            preserve_model_state=preserve_pcoder_state,
            after_step=after_step,
        )
        return logits, features, representation_recovery, prediction_recovery, internal_errors

    if reference_hebbian_layers is None:
        raise RuntimeError("reference Hebbian layers are required with a reference model")
    reference_model.reset()
    if preserve_pcoder_state:
        model.input_mem = batched

    with _defer_hebbian_updates(hebbian_layers), _defer_hebbian_updates(
        reference_hebbian_layers
    ):
        for timestep in range(steps):
            if timestep == 0:
                cpu_rng = torch.get_rng_state()
                cuda_rng = _cuda_rng_state(torch, device)
                reference_output = reference_model(clean_batched)
                if not preserve_pcoder_state:
                    _restore_rng_state(torch, device, cpu_rng, cuda_rng)
                    output = model(batched)
                else:
                    output = model(None)
            else:
                reference_output = reference_model(None)
                output = model(None)
            del reference_output
            logits.append(output.detach().cpu())
            snapshot = recorder.snapshot()
            for name, value in snapshot.items():
                features[name].append(value)
            rep, prediction, errors = _capture_pcoder_recovery(
                model, reference_model, clean_batched, torch
            )
            representation_recovery.append(rep)
            prediction_recovery.append(prediction)
            internal_errors.append(errors)

    return logits, features, representation_recovery, prediction_recovery, internal_errors


def _topk_correct(logits: Any, labels: Any, k: int, np: Any) -> Any:
    top = np.argsort(logits, axis=-1)[..., -k:]
    return np.any(top == labels[..., None], axis=-1)


def run_paper_experiment(config: ExperimentConfig) -> list[Path]:
    if not config.paper.enabled:
        raise ValueError("paper experiment runner requires [paper] enabled = true")
    config.validate_for_run()
    _add_legacy_module_path(config.paths.project_root)

    import numpy as np
    import torch
    from torchvision.datasets import ImageNet
    from torchvision.models import VGG16_Weights

    from Hebb_Priming_Utils import apply_input_degradation
    from .hebb_vgg_priming import _build_model

    device = _resolve_device(config.runtime.device, torch)
    _seed_everything(config.runtime.seed, np, torch)
    dataset = ImageNet(
        str(config.paths.imagenet_root), "val", transform=VGG16_Weights.IMAGENET1K_V1.transforms()
    )
    max_targets = (
        None
        if config.runtime.max_samples is None
        else config.paper.target_offset + config.runtime.max_samples
    )
    targets = select_balanced_paper_targets(
        dataset.targets,
        targets_per_category=config.paper.targets_per_category,
        seed=config.runtime.seed,
        max_targets=max_targets,
    )
    targets = targets[config.paper.target_offset :]
    plans = build_paper_trial_plans(targets, dataset.targets, seed=config.runtime.seed)

    model, hebbian_layers, _ = _build_model(config, device, torch)
    reference_model = None
    reference_hebbian_layers = None
    if config.paper.clean_reference_enabled:
        reference_model, reference_hebbian_layers, _ = _build_model(config, device, torch)
    recorder = _FeatureRecorder(model, hebbian_layers, torch)

    conditions = (
        ("target_only",)
        if config.paper.protocol == "target_only"
        else config.paper.conditions
    )
    runs = []
    for degradation_mode, degradation_level in config.stimulus.degradation_conditions():
        for memory_carrier in config.paper.memory_carriers:
            for condition in conditions:
                identities = (
                    ("none",)
                    if condition in {"no_prime", "target_only"}
                    else ("same",)
                    if memory_carrier == "none"
                    else config.paper.prime_identities
                )
                for prime_identity in identities:
                    runs.append(
                        (
                            degradation_mode,
                            degradation_level,
                            memory_carrier,
                            condition,
                            prime_identity,
                        )
                    )

    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    started = time.time()
    try:
        for degradation_mode, degradation_level, carrier, condition, prime_identity in runs:
            _seed_everything(config.runtime.seed, np, torch)
            carrier_has_hebb = carrier in {"hebb_only", "hebb_and_pcoder"}
            carrier_has_pcoder = carrier in {"pcoder_carry", "hebb_and_pcoder"}
            all_logits: list[Any] = []
            all_features: dict[str, list[Any]] = {
                name: [] for name in REPRESENTATION_LAYER_NAMES
            }
            all_rep_recovery: list[Any] = []
            all_prediction_recovery: list[Any] = []
            all_internal_errors: list[Any] = []
            trial_metadata: list[dict[str, Any]] = []

            print(
                f"[paper] condition={condition} identity={prime_identity} "
                f"carrier={carrier} degradation={degradation_mode}:{degradation_level:g} "
                f"targets={len(plans)}"
            )
            for ordinal, plan in enumerate(plans):
                reset_hebbian_state(hebbian_layers)
                if reference_hebbian_layers is not None:
                    reset_hebbian_state(reference_hebbian_layers)

                sequence_metadata: dict[str, Any] = {}
                actual_preview_indices: list[int] = []
                if condition != "target_only":
                    preview1_indices, preview2_indices, sequence_metadata = (
                        materialize_preview_indices(
                            plan,
                            condition=condition,
                            prime_identity=(
                                "different" if prime_identity == "different" else "same"
                            ),
                        )
                    )
                    actual_preview_indices = [
                        *preview1_indices,
                        *preview2_indices,
                    ]
                    preview_canvases = []
                    for pair in (preview1_indices, preview2_indices):
                        left = dataset[pair[0]][0].to(device)
                        right = dataset[pair[1]][0].to(device)
                        preview_canvases.append(
                            make_paper_canvas(
                                ((left, "left"), (right, "right")),
                                object_scale=config.paper.object_scale,
                                horizontal_offset=config.paper.horizontal_offset,
                                background_value=config.paper.background_value,
                            )
                        )
                    for preview_index, preview in enumerate(preview_canvases):
                        run_recurrent_sequence(
                            model,
                            preview.unsqueeze(0).to(device),
                            config.paper.preview_time_steps,
                            hebbian_layers=hebbian_layers,
                            commit_update=carrier_has_hebb,
                            preserve_model_state=(carrier_has_pcoder and preview_index > 0),
                        )

                target_image, target_label = dataset[plan.target.dataset_index]
                target_image = target_image.to(device)
                clean_target = make_paper_canvas(
                    ((target_image, "center"),),
                    object_scale=config.paper.object_scale,
                    horizontal_offset=config.paper.horizontal_offset,
                    background_value=config.paper.background_value,
                )
                degradation_seed = paper_degradation_seed(
                    config.stimulus.degradation_seed,
                    degradation_mode,
                    target_offset=config.paper.target_offset,
                    ordinal=ordinal,
                )
                degraded_image = apply_input_degradation(
                    target_image,
                    mode=degradation_mode,
                    level=degradation_level,
                    seed=degradation_seed,
                    fractal_octaves=config.stimulus.fractal_octaves,
                )
                degraded_target = make_paper_canvas(
                    ((degraded_image, "center"),),
                    object_scale=config.paper.object_scale,
                    horizontal_offset=config.paper.horizontal_offset,
                    background_value=config.paper.background_value,
                )

                logits, features, rep_recovery, prediction_recovery, errors = _run_target(
                    model=model,
                    reference_model=reference_model,
                    image=degraded_target,
                    clean_image=clean_target,
                    steps=config.paper.target_time_steps,
                    preserve_pcoder_state=(carrier_has_pcoder and condition != "target_only"),
                    hebbian_layers=hebbian_layers,
                    reference_hebbian_layers=reference_hebbian_layers,
                    recorder=recorder,
                    torch=torch,
                    device=device,
                )
                all_logits.append(torch.cat(logits, dim=0).numpy())
                for name in REPRESENTATION_LAYER_NAMES:
                    all_features[name].append(np.stack(features[name], axis=0))
                if rep_recovery:
                    all_rep_recovery.append(np.stack(rep_recovery, axis=0))
                    all_prediction_recovery.append(np.stack(prediction_recovery, axis=0))
                    all_internal_errors.append(np.stack(errors, axis=0))
                trial_metadata.append(
                    {
                        "dataset_index": plan.target.dataset_index,
                        "imagenet_label": int(target_label),
                        "object_identity": plan.target.imagenet_label,
                        "supercategory": plan.target.supercategory,
                        "category_index": plan.target.category_index,
                        "degradation_seed": degradation_seed,
                        "base_distractor_indices": list(plan.preview_indices),
                        "actual_preview_indices": actual_preview_indices,
                        "preview_indices": actual_preview_indices,
                        **sequence_metadata,
                    }
                )

            logits_array = np.stack(all_logits, axis=0)
            labels = np.asarray([plan.target.imagenet_label for plan in plans], dtype=np.int64)
            category_labels = np.asarray(
                [plan.target.category_index for plan in plans], dtype=np.int64
            )
            top1_timeline = _topk_correct(logits_array, labels[:, None], 1, np)
            top5_timeline = _topk_correct(logits_array, labels[:, None], 5, np)
            result = {
                "schema_version": 4,
                "paper_schema_version": PAPER_RESULT_SCHEMA_VERSION,
                "run_metadata": {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "paper_reference_doi": "10.1101/2025.05.16.654463",
                    "config_file": str(config.source_file),
                    "device": str(device),
                    "seed": config.runtime.seed,
                    "protocol": config.paper.protocol,
                    "condition": condition,
                    "prime_identity": prime_identity,
                    "memory_carrier": carrier,
                    "degradation_mode": degradation_mode,
                    "degradation_level": degradation_level,
                    "hebbian_condition": "nohebb" if config.model.original_mode else "hebb",
                    "config": asdict(config),
                },
                "supercategory_names": SUPER_CATEGORIES,
                "trial_metadata": trial_metadata,
                "dataset_indices": np.asarray(
                    [plan.target.dataset_index for plan in plans], dtype=np.int64
                ),
                "imagenet_labels": labels,
                "object_identity_labels": labels.copy(),
                "category_labels": category_labels,
                "target_logits": logits_array,
                "target_top1_correct": top1_timeline,
                "target_top5_correct": top5_timeline,
                "target_top1_accuracy_timeline": top1_timeline.mean(axis=0),
                "target_top5_accuracy_timeline": top5_timeline.mean(axis=0),
                "representation_layer_names": REPRESENTATION_LAYER_NAMES,
                "target_representations": {
                    name: np.stack(values, axis=0)
                    for name, values in all_features.items()
                },
                "recovery_metric_names": RECOVERY_METRIC_NAMES,
                "pcoder_indices": (1, 2, 3, 4, 5),
                "representation_recovery": (
                    np.stack(all_rep_recovery, axis=0)
                    if all_rep_recovery
                    else None
                ),
                "prediction_recovery": (
                    np.stack(all_prediction_recovery, axis=0)
                    if all_prediction_recovery
                    else None
                ),
                "pcoder_internal_error": (
                    np.stack(all_internal_errors, axis=0)
                    if all_internal_errors
                    else None
                ),
            }
            output_file = config.paths.output_dir / paper_result_filename(
                protocol=config.paper.protocol,
                condition=condition,
                prime_identity=prime_identity,
                memory_carrier=carrier,
                degradation_mode=degradation_mode,
                degradation_level=degradation_level,
            )
            with output_file.open("wb") as handle:
                pickle.dump(result, handle)
            output_paths.append(output_file)
            print(
                f"[paper] wrote {output_file} "
                f"t0={result['target_top1_accuracy_timeline'][0]:.3f} "
                f"t9={result['target_top1_accuracy_timeline'][-1]:.3f}"
            )
    finally:
        recorder.close()

    print(f"[paper] completed {len(output_paths)} files in {time.time() - started:.1f}s")
    return output_paths


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run paper-matched parafoveal experiments")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--check-config", action="store_true")
    return parser


def main(argv=None) -> int:
    import os
    from pprint import pprint

    from .config import load_experiment_config

    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config).with_overrides(
        imagenet_root=args.imagenet_root or os.environ.get("PREDIFY_IMAGENET_ROOT"),
        output_dir=args.output_dir or os.environ.get("PREDIFY_OUTPUT_DIR"),
        device=args.device,
        max_samples=args.max_samples,
    )
    config.validate_for_run()
    if args.check_config:
        pprint(config)
        return 0
    run_paper_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
