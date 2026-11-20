from pathlib import Path
import sys
import unittest

from predify_experiments.sequence import run_recurrent_sequence


try:
    import torch

    LEGACY_DIR = Path(__file__).resolve().parents[1] / "Hebbian-CNN"
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    from Hebbian_VGG_Lib import Hebb_VGG_Channel_Boost

    TORCH_IMPORT_ERROR = None
except Exception as error:  # The old Anaconda base can fail while loading native DLLs.
    torch = None
    Hebb_VGG_Channel_Boost = None
    TORCH_IMPORT_ERROR = error


@unittest.skipIf(torch is None, f"PyTorch environment unavailable: {TORCH_IMPORT_ERROR}")
class HebbianModuleTests(unittest.TestCase):
    def test_deferred_sequence_commits_once_and_reset_clears_weight(self):
        layer = Hebb_VGG_Channel_Boost(
            in_channels=4,
            decay=0.0,
            coeff=1.0,
            inh_c=0.0,
            sparse_thres=0.0,
        )

        class LayerModel:
            def reset(self):
                return None

            def __call__(self, value):
                if value is None:
                    value = torch.ones((1, 4, 2, 2))
                return layer(value)

        inputs = torch.ones((1, 4, 2, 2))
        self.assertEqual(torch.count_nonzero(layer.boost_weight).item(), 0)
        run_recurrent_sequence(
            LayerModel(), inputs, 3, hebbian_layers=[layer], commit_update=True
        )
        self.assertGreater(torch.count_nonzero(layer.boost_weight).item(), 0)
        self.assertTrue(layer.update_enabled)

        layer.zero_boost_weight()
        self.assertEqual(torch.count_nonzero(layer.boost_weight).item(), 0)


if __name__ == "__main__":
    unittest.main()
