from pathlib import Path
import random
import sys
import unittest


try:
    import torch

    LEGACY_DIR = Path(__file__).resolve().parents[1] / "Hebbian-CNN"
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    from Hebb_Priming_Utils import (
        add_fractal_occlusion,
        add_predify_noise,
        apply_input_degradation,
        gaussian_blur,
        make_half_field_canvas,
        make_half_field_stimulus_v5,
        make_priming_stimulus,
    )

    TORCH_IMPORT_ERROR = None
except Exception as error:
    torch = None
    TORCH_IMPORT_ERROR = error


@unittest.skipIf(torch is None, f"PyTorch environment unavailable: {TORCH_IMPORT_ERROR}")
class StimulusDegradationTests(unittest.TestCase):
    class _Dataset:
        def __init__(self):
            self.items = [
                (torch.full((3, 8, 8), float(index)), 100 + index)
                for index in range(8)
            ]

        def __getitem__(self, index):
            image, label = self.items[index]
            return image.clone(), label

        def __len__(self):
            return len(self.items)

    def test_zero_degradation_is_shared_identity_baseline(self):
        image = torch.randn((3, 9, 11))
        predify_zero = apply_input_degradation(
            image,
            mode="predify_noise",
            level=0.0,
            seed=0,
        )
        fractal_zero = apply_input_degradation(
            image,
            mode="fractal_occlusion",
            level=0.0,
            seed=0,
        )

        self.assertIs(predify_zero, image)
        self.assertIs(fractal_zero, image)

    def test_predify_noise_matches_seeded_normal_sample(self):
        image = torch.zeros((3, 5, 7))
        actual = add_predify_noise(image, sigma=0.75, seed=0)

        generator = torch.Generator(device="cpu").manual_seed(0)
        expected = torch.normal(
            mean=0.0,
            std=0.75,
            size=tuple(image.shape),
            generator=generator,
        )
        torch.testing.assert_close(actual, expected)

    def test_foveated_prime_and_nonprime_follow_schedule(self):
        dataset = self._Dataset()
        schedule = [1, 1, 3, 3, 5, 5, 7, 7]
        prime, prime_meta = make_priming_stimulus(
            dataset,
            schedule[0],
            0,
            priming=True,
            idx_log_list=schedule,
            prime_layout="foveated_full",
            contrast_scale=0.0,
            blur_sigma=10.0,
            downsample_after_blur=True,
            device=torch.device("cpu"),
        )
        nonprime, nonprime_meta = make_priming_stimulus(
            dataset,
            schedule[0],
            0,
            priming=False,
            idx_log_list=schedule,
            prime_layout="foveated_full",
            contrast_scale=0.0,
            blur_sigma=10.0,
            downsample_after_blur=True,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(prime, dataset[1][0])
        torch.testing.assert_close(nonprime, dataset[3][0])
        self.assertEqual(prime_meta["y_priming"], 101)
        self.assertEqual(nonprime_meta["y_priming"], 103)
        self.assertEqual(tuple(prime.shape), (3, 8, 8))

    def test_odd_target_is_identical_across_prime_layouts_and_seeds(self):
        dataset = self._Dataset()
        schedule = [1, 1, 3, 3]
        for mode, level in (("predify_noise", 0.5), ("fractal_occlusion", 0.3)):
            with self.subTest(mode=mode):
                common = dict(
                    dataset=dataset,
                    idx=schedule[1],
                    trial_idx=1,
                    priming=True,
                    idx_log_list=schedule,
                    degradation_mode=mode,
                    degradation_level=level,
                    degradation_seed=1,
                    device=torch.device("cpu"),
                )
                parafovea, _ = make_priming_stimulus(
                    **common, prime_layout="parafoveal_half_field"
                )
                foveated, _ = make_priming_stimulus(
                    **common, prime_layout="foveated_full"
                )
                torch.testing.assert_close(parafovea, foveated, rtol=0, atol=0)

    def test_default_layout_dispatch_preserves_legacy_even_stimulus(self):
        dataset = self._Dataset()
        schedule = [1, 1, 3, 3, 5, 5]
        kwargs = dict(
            dataset=dataset,
            idx=schedule[0],
            trial_idx=0,
            priming=True,
            idx_log_list=schedule,
            device=torch.device("cpu"),
        )
        random.seed(9)
        expected, expected_meta = make_half_field_stimulus_v5(**kwargs)
        random.seed(9)
        actual, actual_meta = make_priming_stimulus(**kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual_meta["priming_on_left"], expected_meta["priming_on_left"])

    def test_fractal_occlusion_contains_no_additive_noise(self):
        image = torch.zeros((3, 32, 32))
        actual = add_fractal_occlusion(image, intensity=0.5, seed=4)
        normalized_black = torch.tensor(
            [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
        ).view(3, 1, 1)

        # Pure black blending uses one shared mask in all channels. Therefore
        # actual[channel] / black[channel] must be identical across channels.
        inferred_masks = actual / normalized_black
        torch.testing.assert_close(inferred_masks[0], inferred_masks[1])
        torch.testing.assert_close(inferred_masks[1], inferred_masks[2])
        self.assertGreaterEqual(inferred_masks.min().item(), 0.0)
        self.assertLessEqual(inferred_masks.max().item(), 1.0)

    def test_prime_blur_is_applied_after_degradation(self):
        left = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8) / 100
        right = torch.flip(left, dims=(2,))
        plain_canvas, _ = make_half_field_canvas(
            left,
            right,
            priming_on_left=True,
            canvas_scale=2.0,
            contrast_scale=1.0,
            degradation_mode="none",
            blur_sigma=0.0,
            downsample_after_blur=False,
        )
        actual, _ = make_half_field_canvas(
            left,
            right,
            priming_on_left=True,
            canvas_scale=2.0,
            contrast_scale=1.0,
            degradation_mode="predify_noise",
            degradation_level=0.25,
            degradation_seed=0,
            blur_sigma=1.0,
            downsample_after_blur=False,
        )
        expected = gaussian_blur(
            apply_input_degradation(
                plain_canvas,
                mode="predify_noise",
                level=0.25,
                seed=0,
            ),
            sigma=1.0,
        )
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
