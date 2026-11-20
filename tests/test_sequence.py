import unittest

from predify_experiments.sequence import reset_hebbian_state, run_recurrent_sequence


class FakeModel:
    def __init__(self):
        self.calls = []
        self.reset_count = 0
        self.input_mem = None

    def reset(self):
        self.reset_count += 1

    def __call__(self, value):
        self.calls.append(value)
        return len(self.calls)


class FakeHebbianLayer:
    def __init__(self, update_enabled=True):
        self.update_enabled = update_enabled
        self.commit_count = 0
        self.reset_count = 0

    def commit_update(self):
        self.commit_count += 1

    def zero_boost_weight(self):
        self.reset_count += 1


class RecurrentSequenceTests(unittest.TestCase):
    def test_input_is_only_supplied_on_first_timestep(self):
        model = FakeModel()
        outputs = run_recurrent_sequence(model, "image", 3)
        self.assertEqual(outputs, [1, 2, 3])
        self.assertEqual(model.calls, ["image", None, None])
        self.assertEqual(model.reset_count, 1)

    def test_hebbian_update_is_committed_once_and_state_is_restored(self):
        model = FakeModel()
        layer = FakeHebbianLayer(update_enabled=True)
        observed_states = []

        run_recurrent_sequence(
            model,
            "image",
            4,
            hebbian_layers=[layer],
            after_step=lambda _step, _output: observed_states.append(layer.update_enabled),
        )

        self.assertEqual(observed_states, [False, False, False, False])
        self.assertEqual(layer.commit_count, 1)
        self.assertTrue(layer.update_enabled)

    def test_disabled_layer_is_not_committed(self):
        layer = FakeHebbianLayer(update_enabled=False)
        run_recurrent_sequence(FakeModel(), "image", 2, hebbian_layers=[layer])
        self.assertEqual(layer.commit_count, 0)
        self.assertFalse(layer.update_enabled)

    def test_commit_can_be_deferred_across_trials(self):
        layer = FakeHebbianLayer(update_enabled=True)
        run_recurrent_sequence(
            FakeModel(), "image", 2, hebbian_layers=[layer], commit_update=False
        )
        self.assertEqual(layer.commit_count, 0)

    def test_state_carry_replaces_input_without_reset(self):
        model = FakeModel()
        model.input_mem = "prime"
        outputs = run_recurrent_sequence(
            model,
            "target",
            2,
            preserve_model_state=True,
        )
        self.assertEqual(outputs, [1, 2])
        self.assertEqual(model.reset_count, 0)
        self.assertEqual(model.input_mem, "target")
        self.assertEqual(model.calls, [None, None])

    def test_reset_clears_all_layers(self):
        layers = [FakeHebbianLayer(), FakeHebbianLayer()]
        reset_hebbian_state(layers)
        self.assertEqual([layer.reset_count for layer in layers], [1, 1])

    def test_invalid_step_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            run_recurrent_sequence(FakeModel(), "image", 0)

    def test_state_carry_requires_input_memory(self):
        model = FakeModel()
        del model.input_mem
        with self.assertRaisesRegex(ValueError, "input_mem"):
            run_recurrent_sequence(model, "target", 1, preserve_model_state=True)


if __name__ == "__main__":
    unittest.main()
