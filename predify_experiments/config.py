from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping


_UNEXPANDED_ENV = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)|%[^%]+%")


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    imagenet_root: Path
    sample_file: Path
    output_dir: Path
    pcoder_weight_dir: Path
    fc_checkpoint: Path | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    seed: int = 114514
    max_time_steps: int = 10
    reduced_time_steps: int = 2
    max_samples: int | None = 2000


@dataclass(frozen=True)
class ModelConfig:
    name: str = "pvgg"
    pretrained_pcoders: bool = True
    random_init: bool = True
    fc_weight_source: str = "torchvision"
    fc_hebb_compute_enabled: bool = True
    original_mode: bool = False
    hyperparameters: tuple[Mapping[str, float], ...] = ()


@dataclass(frozen=True)
class DegradationSpec:
    mode: str = "none"
    levels: tuple[float, ...] = (0.0,)


@dataclass(frozen=True)
class StimulusConfig:
    canvas_scale: float = 2.0
    contrast_scale: float = 0.5
    blur_sigma: float = 1.0
    downsample_after_blur: bool = True
    prime_layout: str = "parafoveal_half_field"
    degradation_mode: str = "none"
    degradation_levels: tuple[float, ...] = (0.0,)
    degradations: tuple[DegradationSpec, ...] = ()
    degradation_seed: int = 0
    fractal_octaves: int = 3

    def degradation_conditions(self) -> tuple[tuple[str, float], ...]:
        specs = self.degradations or (
            DegradationSpec(self.degradation_mode, self.degradation_levels),
        )
        return tuple((spec.mode, level) for spec in specs for level in spec.levels)


@dataclass(frozen=True)
class ExperimentSettings:
    task_name: str = "N_Back-"
    gap: int = 1
    run_priming: bool = True
    run_nonpriming: bool = True
    representation_class_items: int = 400


@dataclass(frozen=True)
class PaperExperimentSettings:
    enabled: bool = False
    protocol: str = "paper_sequence"
    conditions: tuple[str, ...] = ("no_prime", "one_back", "two_back")
    prime_identities: tuple[str, ...] = ("same",)
    memory_carriers: tuple[str, ...] = ("hebb_only",)
    targets_per_category: int = 40
    target_offset: int = 0
    preview_time_steps: int = 2
    target_time_steps: int = 10
    object_scale: float = 0.15
    horizontal_offset: float = 0.175
    background_value: float = 0.5
    clean_reference_enabled: bool = False
    diagnostic_tensor_samples: int = 0
    decoder_folds: int = 5
    decoder_seed: int = 114514


@dataclass(frozen=True)
class ExperimentConfig:
    source_file: Path
    paths: ProjectPaths
    runtime: RuntimeConfig
    model: ModelConfig
    stimulus: StimulusConfig
    experiment: ExperimentSettings
    paper: PaperExperimentSettings = PaperExperimentSettings()

    def with_overrides(
        self,
        *,
        imagenet_root: str | Path | None = None,
        output_dir: str | Path | None = None,
        fc_checkpoint: str | Path | None = None,
        fc_weight_source: str | None = None,
        device: str | None = None,
        max_samples: int | None = None,
    ) -> "ExperimentConfig":
        paths = self.paths
        model = self.model
        runtime = self.runtime
        if imagenet_root is not None:
            paths = replace(paths, imagenet_root=_resolve_path(str(imagenet_root), paths.project_root))
        if output_dir is not None:
            paths = replace(paths, output_dir=_resolve_path(str(output_dir), paths.project_root))
        if fc_checkpoint is not None:
            paths = replace(paths, fc_checkpoint=_resolve_path(str(fc_checkpoint), paths.project_root))
        if fc_weight_source is not None:
            model = replace(model, fc_weight_source=fc_weight_source)
        if device is not None:
            runtime = replace(runtime, device=device)
        if max_samples is not None:
            runtime = replace(runtime, max_samples=max_samples)
        return replace(self, paths=paths, model=model, runtime=runtime)

    def validate_for_run(self) -> None:
        errors: list[str] = []
        if not self.paths.imagenet_root.is_dir():
            errors.append(f"ImageNet root does not exist: {self.paths.imagenet_root}")
        if not self.paths.sample_file.is_file():
            errors.append(f"sample file does not exist: {self.paths.sample_file}")
        if self.model.pretrained_pcoders and not self.paths.pcoder_weight_dir.is_dir():
            errors.append(f"PCoder weight directory does not exist: {self.paths.pcoder_weight_dir}")
        elif self.model.pretrained_pcoders and self.model.name.lower() == "pvgg":
            for index in range(1, 6):
                checkpoint = self.paths.pcoder_weight_dir / (
                    f"pvgg16_imagenet_pretrained_pc{index}_pmodule.pth"
                )
                if not checkpoint.is_file():
                    errors.append(f"PCoder checkpoint does not exist: {checkpoint}")
        if self.model.fc_weight_source not in {"torchvision", "finetuned"}:
            errors.append("model.fc_weight_source must be 'torchvision' or 'finetuned'")
        if self.model.name.lower() == "pvgg" and len(self.model.hyperparameters) not in {0, 5}:
            errors.append("PVGG requires exactly five hyperparameter tables")
        for index, hyperparameters in enumerate(self.model.hyperparameters, start=1):
            ffm = float(hyperparameters["ffm"])
            fbm = float(hyperparameters["fbm"])
            erm = float(hyperparameters["erm"])
            if ffm < 0 or fbm < 0 or ffm + fbm > 1:
                errors.append(
                    f"PCoder{index} requires ffm >= 0, fbm >= 0, and ffm + fbm <= 1"
                )
            if erm < 0:
                errors.append(f"PCoder{index} requires erm >= 0")
        if self.model.fc_weight_source == "finetuned":
            if self.paths.fc_checkpoint is None or not self.paths.fc_checkpoint.is_file():
                errors.append(f"FC checkpoint does not exist: {self.paths.fc_checkpoint}")
        if self.runtime.max_time_steps < 1:
            errors.append("runtime.max_time_steps must be at least 1")
        if not 1 <= self.runtime.reduced_time_steps <= self.runtime.max_time_steps:
            errors.append("runtime.reduced_time_steps must be between 1 and max_time_steps")
        if self.runtime.max_samples is not None and self.runtime.max_samples < 1:
            errors.append("runtime.max_samples must be positive or omitted")
        if self.stimulus.prime_layout not in {
            "parafoveal_half_field",
            "foveated_full",
            "paper_parafoveal",
        }:
            errors.append(
                "stimulus.prime_layout must be 'parafoveal_half_field' or "
                "'foveated_full', or 'paper_parafoveal'"
            )
        conditions = self.stimulus.degradation_conditions()
        if not conditions:
            errors.append("stimulus degradation settings must contain at least one value")
        if len(set(conditions)) != len(conditions):
            errors.append("stimulus degradation mode/level pairs must be unique")
        for mode, level in conditions:
            if mode not in {"none", "predify_noise", "fractal_occlusion"}:
                errors.append(f"unknown stimulus degradation mode: {mode}")
            if level < 0:
                errors.append("stimulus degradation levels must be non-negative")
            if mode == "none" and level != 0.0:
                errors.append("degradation mode 'none' requires level 0.0")
            if mode == "fractal_occlusion" and level > 1:
                errors.append("fractal occlusion levels must be between 0 and 1")
        if self.stimulus.fractal_octaves < 1:
            errors.append("stimulus.fractal_octaves must be at least 1")
        if not (self.experiment.run_priming or self.experiment.run_nonpriming):
            errors.append("at least one experimental condition must be enabled")
        if self.paper.enabled:
            valid_conditions = {"no_prime", "one_back", "two_back"}
            invalid_conditions = set(self.paper.conditions) - valid_conditions
            if invalid_conditions:
                errors.append(
                    "paper.conditions contains unknown values: "
                    + ", ".join(sorted(invalid_conditions))
                )
            if not self.paper.conditions:
                errors.append("paper.conditions must not be empty")
            valid_identities = {"same", "different"}
            if set(self.paper.prime_identities) - valid_identities:
                errors.append("paper.prime_identities must contain same/different")
            valid_carriers = {"none", "hebb_only", "pcoder_carry", "hebb_and_pcoder"}
            if set(self.paper.memory_carriers) - valid_carriers:
                errors.append("paper.memory_carriers contains an unknown carrier")
            if self.paper.protocol not in {"paper_sequence", "target_only"}:
                errors.append("paper.protocol must be 'paper_sequence' or 'target_only'")
            if self.paper.targets_per_category < 1:
                errors.append("paper.targets_per_category must be positive")
            paper_target_total = self.paper.targets_per_category * 5
            if self.paper.target_offset < 0:
                errors.append("paper.target_offset must be non-negative")
            elif self.paper.target_offset >= paper_target_total:
                errors.append("paper.target_offset must leave at least one target")
            if (
                self.runtime.max_samples is not None
                and self.paper.target_offset + self.runtime.max_samples
                > paper_target_total
            ):
                errors.append(
                    "paper.target_offset + runtime.max_samples exceeds the "
                    "balanced paper target pool"
                )
            if self.paper.preview_time_steps < 1 or self.paper.target_time_steps < 1:
                errors.append("paper timestep counts must be positive")
            if not 0 < self.paper.object_scale <= 1:
                errors.append("paper.object_scale must be in (0, 1]")
            if not 0 <= self.paper.horizontal_offset < 0.5:
                errors.append("paper.horizontal_offset must be in [0, 0.5)")
            if self.paper.object_scale / 2 + self.paper.horizontal_offset > 0.5:
                errors.append("paper parafoveal objects do not fit on the canvas")
            if not 0 <= self.paper.background_value <= 1:
                errors.append("paper.background_value must be in [0, 1]")
            if self.paper.diagnostic_tensor_samples < 0:
                errors.append("paper.diagnostic_tensor_samples must be non-negative")
            if self.paper.decoder_folds < 2:
                errors.append("paper.decoder_folds must be at least 2")
        if errors:
            raise ValueError("Invalid experiment configuration:\n- " + "\n- ".join(errors))


def _required_table(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing TOML table: [{key}]")
    return value


def _expand(raw: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(raw))
    if _UNEXPANDED_ENV.search(expanded):
        raise ValueError(f"Unresolved environment variable in path: {raw}")
    return expanded


def _resolve_path(raw: str, base: Path) -> Path:
    expanded = _expand(raw)
    path = Path(expanded)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _optional_path(raw: Any, base: Path) -> Path | None:
    if raw is None or str(raw).strip() == "":
        return None
    return _resolve_path(str(raw), base)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source_file = Path(path).resolve()
    with source_file.open("rb") as handle:
        data = tomllib.load(handle)

    project = _required_table(data, "project")
    project_root = _resolve_path(str(project.get("root", "..")), source_file.parent)
    raw_paths = _required_table(data, "paths")
    raw_runtime = _required_table(data, "runtime")
    raw_model = _required_table(data, "model")
    raw_stimulus = _required_table(data, "stimulus")
    raw_experiment = _required_table(data, "experiment")
    raw_paper = data.get("paper", {})
    if not isinstance(raw_paper, Mapping):
        raise ValueError("[paper] must be a TOML table")

    paths = ProjectPaths(
        project_root=project_root,
        imagenet_root=_resolve_path(str(raw_paths.get("imagenet_root", "")), project_root),
        sample_file=_resolve_path(str(raw_paths["sample_file"]), project_root),
        output_dir=_resolve_path(str(raw_paths["output_dir"]), project_root),
        pcoder_weight_dir=_resolve_path(str(raw_paths["pcoder_weight_dir"]), project_root),
        fc_checkpoint=_optional_path(raw_paths.get("fc_checkpoint"), project_root),
    )
    runtime = RuntimeConfig(
        device=str(raw_runtime.get("device", "auto")),
        seed=int(raw_runtime.get("seed", 114514)),
        max_time_steps=int(raw_runtime.get("max_time_steps", 10)),
        reduced_time_steps=int(raw_runtime.get("reduced_time_steps", 2)),
        max_samples=(
            None if raw_runtime.get("max_samples") is None else int(raw_runtime["max_samples"])
        ),
    )
    hps = tuple(
        {
            "ffm": float(item["ffm"]),
            "fbm": float(item["fbm"]),
            "erm": float(item["erm"]),
        }
        for item in raw_model.get("hyperparameters", [])
    )
    model = ModelConfig(
        name=str(raw_model.get("name", "pvgg")),
        pretrained_pcoders=bool(raw_model.get("pretrained_pcoders", True)),
        random_init=bool(raw_model.get("random_init", True)),
        fc_weight_source=str(raw_model.get("fc_weight_source", "torchvision")),
        fc_hebb_compute_enabled=bool(raw_model.get("fc_hebb_compute_enabled", True)),
        original_mode=bool(raw_model.get("original_mode", False)),
        hyperparameters=hps,
    )
    raw_degradations = raw_stimulus.get("degradations", [])
    if raw_degradations and (
        "degradation_mode" in raw_stimulus or "degradation_levels" in raw_stimulus
    ):
        raise ValueError(
            "Use either stimulus.degradations or the legacy degradation_mode/"
            "degradation_levels keys, not both"
        )
    if raw_degradations and not isinstance(raw_degradations, list):
        raise ValueError("stimulus.degradations must be an array of TOML tables")

    degradation_mode = str(raw_stimulus.get("degradation_mode", "none"))
    raw_degradation_levels = raw_stimulus.get("degradation_levels")
    if raw_degradation_levels is None:
        if degradation_mode == "predify_noise":
            degradation_levels = (0.25, 0.5, 0.75, 1.0, 2.0)
        elif degradation_mode == "fractal_occlusion":
            degradation_levels = tuple(index * 0.5 / 29 for index in range(30))
        else:
            degradation_levels = (0.0,)
    elif not isinstance(raw_degradation_levels, list):
        raise ValueError("stimulus.degradation_levels must be a TOML array")
    else:
        degradation_levels = tuple(float(level) for level in raw_degradation_levels)

    degradations: list[DegradationSpec] = []
    for item in raw_degradations:
        if not isinstance(item, Mapping):
            raise ValueError("each stimulus.degradations entry must be a TOML table")
        levels = item.get("levels")
        if not isinstance(levels, list):
            raise ValueError("stimulus.degradations levels must be a TOML array")
        degradations.append(
            DegradationSpec(
                mode=str(item.get("mode", "none")),
                levels=tuple(float(level) for level in levels),
            )
        )

    stimulus = StimulusConfig(
        canvas_scale=float(raw_stimulus.get("canvas_scale", 2.0)),
        contrast_scale=float(raw_stimulus.get("contrast_scale", 0.5)),
        blur_sigma=float(raw_stimulus.get("blur_sigma", 1.0)),
        downsample_after_blur=bool(raw_stimulus.get("downsample_after_blur", True)),
        prime_layout=str(raw_stimulus.get("prime_layout", "parafoveal_half_field")),
        degradation_mode=degradation_mode,
        degradation_levels=degradation_levels,
        degradations=tuple(degradations),
        degradation_seed=int(raw_stimulus.get("degradation_seed", 0)),
        fractal_octaves=int(raw_stimulus.get("fractal_octaves", 3)),
    )
    experiment = ExperimentSettings(
        task_name=str(raw_experiment.get("task_name", "N_Back-")),
        gap=int(raw_experiment.get("gap", 1)),
        run_priming=bool(raw_experiment.get("run_priming", True)),
        run_nonpriming=bool(raw_experiment.get("run_nonpriming", True)),
        representation_class_items=int(raw_experiment.get("representation_class_items", 400)),
    )
    paper = PaperExperimentSettings(
        enabled=bool(raw_paper.get("enabled", False)),
        protocol=str(raw_paper.get("protocol", "paper_sequence")),
        conditions=tuple(str(value) for value in raw_paper.get(
            "conditions", ["no_prime", "one_back", "two_back"]
        )),
        prime_identities=tuple(str(value) for value in raw_paper.get(
            "prime_identities", ["same"]
        )),
        memory_carriers=tuple(str(value) for value in raw_paper.get(
            "memory_carriers", ["hebb_only"]
        )),
        targets_per_category=int(raw_paper.get("targets_per_category", 40)),
        target_offset=int(raw_paper.get("target_offset", 0)),
        preview_time_steps=int(raw_paper.get("preview_time_steps", 2)),
        target_time_steps=int(raw_paper.get("target_time_steps", runtime.max_time_steps)),
        object_scale=float(raw_paper.get("object_scale", 0.15)),
        horizontal_offset=float(raw_paper.get("horizontal_offset", 0.175)),
        background_value=float(raw_paper.get("background_value", 0.5)),
        clean_reference_enabled=bool(raw_paper.get("clean_reference_enabled", False)),
        diagnostic_tensor_samples=int(raw_paper.get("diagnostic_tensor_samples", 0)),
        decoder_folds=int(raw_paper.get("decoder_folds", 5)),
        decoder_seed=int(raw_paper.get("decoder_seed", runtime.seed)),
    )
    return ExperimentConfig(source_file, paths, runtime, model, stimulus, experiment, paper)
