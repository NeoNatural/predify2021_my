import random
import tempfile
from pathlib import Path
import unittest
from predify_experiments.readout_init_20261012 import balanced_pick, make_trials, budget_index, immutable_json


class ReadoutDesignTests(unittest.TestCase):
    def test_split_and_preview_isolation(self):
        labels = [i for i in range(1000) for _ in range(50)]
        rng = random.Random(12)
        a = balanced_pick(labels, set(), 20, rng)
        b = balanced_pick(labels, set(a), 40, rng)
        c = balanced_pick(labels, set(a+b), 20, rng)
        self.assertEqual(len(set(a+b+c)), 400)
        forbidden = set(a+b+c)
        for trial in make_trials(a, labels, forbidden):
            self.assertFalse(set(trial['previews']) & forbidden)
            for name, index in trial['primes'].items():
                if name != 'exact' and index is not None:
                    self.assertNotIn(index, forbidden)
            self.assertEqual(labels[trial['primes']['same_synset']], trial['imagenet_label'])
            self.assertNotEqual(trial['primes']['same_synset'], trial['dataset_index'])

    def test_total_budget_includes_preview_and_initialization(self):
        self.assertEqual(budget_index(range(6,21), 9), 3)
        self.assertEqual(budget_index(range(2,17), 9), 7)
        self.assertEqual(budget_index(range(1,16), 9), 8)
        with self.assertRaises(ValueError):
            budget_index(range(6,21), 4)

    def test_frozen_specification_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            p = Path(directory) / 'frozen.json'
            immutable_json(p, {'split': [1,2]})
            immutable_json(p, {'split': [1,2]})
            with self.assertRaises(ValueError):
                immutable_json(p, {'split': [1,3]})

    def test_frozen_specification_compares_json_tuple_and_key_representations(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            p = Path(directory) / 'frozen.json'
            specification = {'keys': [(1, 2701), (2, 2702)], 'counts': {0: 2}}
            immutable_json(p, specification)
            before = p.read_bytes()
            immutable_json(p, specification)
            self.assertEqual(before, p.read_bytes())
            with self.assertRaises(ValueError):
                immutable_json(p, {'keys': [(1, 9999), (2, 2702)], 'counts': {0: 2}})


if __name__ == '__main__':
    unittest.main()
