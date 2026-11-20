import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from predify_experiments.config import load_experiment_config


class ExperimentConfigTests(unittest.TestCase):
    def _write_minimal_config(self, root: Path, stimulus_lines: str) -> Path:
        (root / "data").mkdir()
        (root / "weights").mkdir()
        (root / "sample.pckl").touch()
        config_file = root / "experiment.toml"
        config_file.write_text(
            textwrap.dedent(
                f"""
                [project]
                root = "."
                [paths]
                imagenet_root = "data"
                sample_file = "sample.pckl"
                output_dir = "outputs"
                pcoder_weight_dir = "weights"
                [runtime]
                [model]
                pretrained_pcoders = false
                [stimulus]
                {stimulus_lines}
                [experiment]
                """
            ),
            encoding="utf-8",
        )
        return config_file

    def test_paths_are_resolved_from_project_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            config_dir = project / "configs"
            imagenet = project / "data" / "imagenet"
            sample = project / "samples" / "sample.pckl"
            weights = project / "weights"
            for directory in (config_dir, imagenet, sample.parent, weights):
                directory.mkdir(parents=True, exist_ok=True)
            sample.touch()
            config_file = config_dir / "experiment.toml"
            config_file.write_text(
                textwrap.dedent(
                    """
                    [project]
                    root = ".."
                    [paths]
                    imagenet_root = "data/imagenet"
                    sample_file = "samples/sample.pckl"
                    output_dir = "outputs"
                    pcoder_weight_dir = "weights"
                    fc_checkpoint = ""
                    [runtime]
                    max_time_steps = 4
                    reduced_time_steps = 2
                    max_samples = 8
                    [model]
                    pretrained_pcoders = false
                    [stimulus]
                    [experiment]
                    """
                ),
                encoding="utf-8",
            )

            config = load_experiment_config(config_file)
            self.assertEqual(config.paths.project_root, project.resolve())
            self.assertEqual(config.paths.imagenet_root, imagenet.resolve())
            self.assertEqual(config.paths.sample_file, sample.resolve())
            config.validate_for_run()

    def test_environment_variables_expand_in_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data").mkdir()
            (root / "weights").mkdir()
            (root / "sample.pckl").touch()
            config_file = root / "experiment.toml"
            config_file.write_text(
                textwrap.dedent(
                    """
                    [project]
                    root = "."
                    [paths]
                    imagenet_root = "${TEST_IMAGE_ROOT}"
                    sample_file = "sample.pckl"
                    output_dir = "outputs"
                    pcoder_weight_dir = "weights"
                    [runtime]
                    [model]
                    pretrained_pcoders = false
                    [stimulus]
                    [experiment]
                    """
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TEST_IMAGE_ROOT": str(root / "data")}):
                config = load_experiment_config(config_file)
            self.assertEqual(config.paths.imagenet_root, (root / "data").resolve())

    def test_finetuned_classifier_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data").mkdir()
            (root / "weights").mkdir()
            (root / "sample.pckl").touch()
            config_file = root / "experiment.toml"
            config_file.write_text(
                textwrap.dedent(
                    """
                    [project]
                    root = "."
                    [paths]
                    imagenet_root = "data"
                    sample_file = "sample.pckl"
                    output_dir = "outputs"
                    pcoder_weight_dir = "weights"
                    [runtime]
                    [model]
                    pretrained_pcoders = false
                    fc_weight_source = "finetuned"
                    [stimulus]
                    [experiment]
                    """
                ),
                encoding="utf-8",
            )
            config = load_experiment_config(config_file)
            with self.assertRaisesRegex(ValueError, "FC checkpoint"):
                config.validate_for_run()

    def test_predify_noise_uses_canonical_levels_when_omitted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = self._write_minimal_config(
                Path(temp_dir), 'degradation_mode = "predify_noise"'
            )
            config = load_experiment_config(config_file)
            self.assertEqual(
                config.stimulus.degradation_levels,
                (0.25, 0.5, 0.75, 1.0, 2.0),
            )
            config.validate_for_run()

    def test_fractal_occlusion_uses_thirty_levels_without_noise(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = self._write_minimal_config(
                Path(temp_dir), 'degradation_mode = "fractal_occlusion"'
            )
            config = load_experiment_config(config_file)
            self.assertEqual(len(config.stimulus.degradation_levels), 30)
            self.assertEqual(config.stimulus.degradation_levels[0], 0.0)
            self.assertEqual(config.stimulus.degradation_levels[-1], 0.5)
            config.validate_for_run()

    def test_predify_noise_experiment_uses_torchvision_fc_and_all_conditions(self):
        config_file = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "hebb_vgg_predify_noise.toml"
        )
        config = load_experiment_config(config_file)

        self.assertEqual(config.model.fc_weight_source, "torchvision")
        self.assertTrue(config.model.fc_hebb_compute_enabled)
        self.assertFalse(config.model.original_mode)
        self.assertEqual(config.stimulus.degradation_mode, "predify_noise")
        self.assertEqual(
            config.stimulus.degradation_levels,
            (0.25, 0.5, 0.75, 1.0, 2.0),
        )
        self.assertTrue(config.experiment.run_priming)
        self.assertTrue(config.experiment.run_nonpriming)
        self.assertEqual(config.runtime.max_samples, 2000)
        self.assertEqual(config.runtime.max_time_steps, 10)
        self.assertEqual(config.runtime.reduced_time_steps, 2)

    def test_paper_target_offset_is_parsed_for_disjoint_calibration_split(self):
        config_file = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "paper_clean_reference_diagnostic.toml"
        )
        text = config_file.read_text(encoding="utf-8").replace(
            "targets_per_category = 40",
            "targets_per_category = 40\ntarget_offset = 100",
        ).replace("max_samples = 200", "max_samples = 100")
        with tempfile.TemporaryDirectory() as temp_dir:
            modified = Path(temp_dir) / "offset.toml"
            modified.write_text(text, encoding="utf-8")
            # Preserve the original project-relative root after moving TOML.
            text = modified.read_text(encoding="utf-8").replace(
                'root = ".."',
                f'root = "{Path(__file__).resolve().parents[1].as_posix()}"',
            )
            modified.write_text(text, encoding="utf-8")
            config = load_experiment_config(modified)
            self.assertEqual(config.paper.target_offset, 100)

    def test_fractal_experiment_uses_torchvision_fc_and_five_nonzero_levels(self):
        config_file = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "hebb_vgg_fractal_occlusion.toml"
        )
        config = load_experiment_config(config_file)

        self.assertEqual(config.model.fc_weight_source, "torchvision")
        self.assertTrue(config.model.fc_hebb_compute_enabled)
        self.assertFalse(config.model.original_mode)
        self.assertEqual(config.stimulus.degradation_mode, "fractal_occlusion")
        self.assertEqual(
            config.stimulus.degradation_levels,
            (0.1, 0.2, 0.3, 0.4, 0.5),
        )
        self.assertTrue(config.experiment.run_priming)
        self.assertTrue(config.experiment.run_nonpriming)
        self.assertEqual(config.runtime.max_samples, 2000)

    def test_nohebb_configs_preserve_predify_and_disable_hebbian_modules(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        for filename, mode in (
            ("hebb_vgg_predify_noise_nohebb.toml", "predify_noise"),
            ("hebb_vgg_fractal_occlusion_nohebb.toml", "fractal_occlusion"),
        ):
            with self.subTest(filename=filename):
                config = load_experiment_config(config_dir / filename)
                self.assertEqual(config.model.name, "pvgg")
                self.assertTrue(config.model.pretrained_pcoders)
                self.assertTrue(config.model.original_mode)
                self.assertEqual(config.model.fc_weight_source, "torchvision")
                self.assertEqual(config.stimulus.degradation_mode, mode)
                self.assertEqual(
                    config.paths.output_dir.name,
                    "nohebb",
                )

    def test_multi_degradation_foveated_config_is_parsed_and_deduplicated(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        for filename, original_mode in (
            ("hebb_vgg_foveated_representative.toml", False),
            ("hebb_vgg_foveated_representative_nohebb.toml", True),
        ):
            with self.subTest(filename=filename):
                config = load_experiment_config(config_dir / filename)
                self.assertEqual(config.stimulus.prime_layout, "foveated_full")
                self.assertEqual(
                    config.stimulus.degradation_conditions(),
                    (
                        ("none", 0.0),
                        ("predify_noise", 0.5),
                        ("fractal_occlusion", 0.3),
                    ),
                )
                self.assertEqual(config.model.original_mode, original_mode)
                self.assertTrue(config.model.random_init)
                self.assertEqual(config.model.fc_weight_source, "torchvision")

    def test_multi_degradation_rejects_duplicate_mode_level_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = self._write_minimal_config(
                Path(temp_dir),
                textwrap.dedent(
                    """
                    [[stimulus.degradations]]
                    mode = "predify_noise"
                    levels = [0.5]
                    [[stimulus.degradations]]
                    mode = "predify_noise"
                    levels = [0.5]
                    """
                ),
            )
            config = load_experiment_config(config_file)
            with self.assertRaisesRegex(ValueError, "must be unique"):
                config.validate_for_run()

    def test_paper_configs_lock_protocol_conditions_and_recovery(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        clean = load_experiment_config(config_dir / "paper_parafoveal_clean_hebb.toml")
        self.assertTrue(clean.paper.enabled)
        self.assertEqual(clean.stimulus.prime_layout, "paper_parafoveal")
        self.assertEqual(
            clean.paper.conditions,
            ("no_prime", "one_back", "two_back"),
        )
        self.assertEqual(clean.paper.memory_carriers, ("hebb_only",))
        self.assertFalse(clean.paper.clean_reference_enabled)

        diagnostic = load_experiment_config(
            config_dir / "paper_clean_reference_diagnostic.toml"
        )
        self.assertEqual(diagnostic.paper.protocol, "target_only")
        self.assertTrue(diagnostic.paper.clean_reference_enabled)
        self.assertEqual(
            diagnostic.stimulus.degradation_conditions(),
            (("predify_noise", 0.5), ("fractal_occlusion", 0.3)),
        )

        pilot = load_experiment_config(config_dir / "paper_memory_carrier_pilot.toml")
        self.assertEqual(pilot.paper.conditions, ("one_back",))
        self.assertEqual(pilot.paper.prime_identities, ("same", "different"))
        self.assertEqual(
            pilot.paper.memory_carriers,
            ("none", "hebb_only", "pcoder_carry", "hebb_and_pcoder"),
        )


if __name__ == "__main__":
    unittest.main()
