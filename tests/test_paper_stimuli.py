import unittest

import torch

from predify_experiments.paper_stimuli import (
    SUPER_CATEGORIES,
    build_paper_trial_plans,
    imagenet_supercategory,
    make_paper_canvas,
    materialize_preview_indices,
    select_balanced_paper_targets,
)
from predify_experiments.paper_parafoveal import paper_degradation_seed


class PaperStimulusTests(unittest.TestCase):
    def setUp(self):
        labels = [0, 399, 924, 984, 403]
        self.dataset_targets = [label for label in labels for _ in range(6)]
        self.targets = select_balanced_paper_targets(
            self.dataset_targets,
            targets_per_category=2,
            seed=7,
        )
        self.plans = build_paper_trial_plans(
            self.targets,
            self.dataset_targets,
            seed=7,
        )

    def test_imagenet_mapping_covers_five_paper_categories(self):
        self.assertEqual(imagenet_supercategory(0), "animals")
        self.assertEqual(imagenet_supercategory(399), "clothing")
        self.assertEqual(imagenet_supercategory(924), "food")
        self.assertEqual(imagenet_supercategory(984), "plants")
        self.assertEqual(imagenet_supercategory(403), "vehicles")
        self.assertIsNone(imagenet_supercategory(398))

    def test_balanced_selection_is_interleaved(self):
        self.assertEqual(len(self.targets), 10)
        self.assertEqual(
            [target.supercategory for target in self.targets[:5]],
            list(SUPER_CATEGORIES),
        )
        counts = {
            category: sum(target.supercategory == category for target in self.targets)
            for category in SUPER_CATEGORIES
        }
        self.assertEqual(set(counts.values()), {2})

    def test_one_and_two_back_place_exact_target_in_correct_display(self):
        plan = self.plans[0]
        preview1, preview2, metadata = materialize_preview_indices(
            plan, condition="one_back", prime_identity="same"
        )
        self.assertNotIn(plan.target.dataset_index, preview1)
        self.assertIn(plan.target.dataset_index, preview2)
        self.assertEqual(metadata["prime_display"], 2)

        preview1, preview2, metadata = materialize_preview_indices(
            plan, condition="two_back", prime_identity="same"
        )
        self.assertIn(plan.target.dataset_index, preview1)
        self.assertNotIn(plan.target.dataset_index, preview2)
        self.assertEqual(metadata["prime_display"], 1)

    def test_no_prime_and_different_prime_exclude_target(self):
        plan = self.plans[0]
        for condition, identity in (("no_prime", "same"), ("one_back", "different")):
            preview1, preview2, _ = materialize_preview_indices(
                plan, condition=condition, prime_identity=identity
            )
            self.assertNotIn(plan.target.dataset_index, preview1 + preview2)

    def test_canvas_uses_equal_object_size_without_blur_or_contrast(self):
        left = torch.ones(3, 20, 20)
        right = torch.full((3, 20, 20), 2.0)
        preview = make_paper_canvas(
            ((left, "left"), (right, "right")),
            object_scale=0.2,
            horizontal_offset=0.2,
            background_value=0.5,
        )
        target = make_paper_canvas(
            ((left, "center"),),
            object_scale=0.2,
            horizontal_offset=0.2,
            background_value=0.5,
        )
        self.assertEqual(tuple(preview.shape), (3, 20, 20))
        self.assertEqual(tuple(target.shape), (3, 20, 20))
        # Four-by-four tiles are pasted exactly because the source is constant.
        self.assertEqual(int((preview[0] == 1).sum()), 16)
        self.assertEqual(int((preview[0] == 2).sum()), 16)
        self.assertEqual(int((target[0] == 1).sum()), 16)

    def test_fractal_seed_uses_global_target_position_across_splits(self):
        self.assertEqual(
            paper_degradation_seed(
                7, "fractal_occlusion", target_offset=100, ordinal=3
            ),
            110,
        )
        self.assertEqual(
            paper_degradation_seed(7, "predify_noise", target_offset=100, ordinal=3),
            7,
        )


if __name__ == "__main__":
    unittest.main()
