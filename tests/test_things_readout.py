import json
from pathlib import Path
import unittest
import numpy as np

from predify_experiments.things_readout_data import SubsetReadout, TARGET_CATEGORIES
from predify_experiments.things_readout_20261012 import OUT


class FakeFiveClassHead:
    classes_ = np.arange(5)
    def decision_function(self, x):
        return np.tile([1, 2, 3, 99, 4], (len(x), 1))


class ThingsReadoutTests(unittest.TestCase):
    def test_fixed_readout_category_projection(self):
        head = SubsetReadout(FakeFiveClassHead())
        scores = head.decision_function(np.zeros((2, 4096)))
        np.testing.assert_array_equal(scores, [[1,2,3,4],[1,2,3,4]])
        np.testing.assert_array_equal(scores.argmax(-1), [3,3])

    @unittest.skipUnless((OUT/'manifest.json').exists(), 'Local THINGS manifest required for integration audit')
    def test_actual_split_and_hash_isolation(self):
        manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
        sets = [{r['dataset_index'] for r in manifest[s]} for s in ('train','dev','test')]
        self.assertEqual([len(s) for s in sets], [160,80,80])
        self.assertEqual(len(set.union(*sets)), 320)
        records = manifest['images']
        self.assertEqual(len({records[str(i)]['sha256'] for i in set.union(*sets)}), 320)
        self.assertFalse(set.union(*sets) & set(manifest['historical_target_exclusions']))
        for split, quota in [('train',40),('dev',20),('test',20)]:
            self.assertEqual([sum(r['category_index']==c for r in manifest[split]) for c in range(4)], [quota]*4)
            for c in range(4):
                self.assertEqual([sum(r['category_index']==c and r['slot']==side for r in manifest[split]) for side in (2,3)], [quota//2]*2)

    @unittest.skipUnless((OUT/'manifest.json').exists(), 'Local THINGS manifest required for integration audit')
    def test_content_controls_and_preview_isolation(self):
        m = json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
        targets = {r['dataset_index'] for s in ('train','dev','test') for r in m[s]}
        images = m['images']
        self.assertFalse(any(r['source']=='imagenet' for r in images.values()))
        for s in ('train','dev','test'):
            for trial in m[s]:
                self.assertEqual(len(trial['previews']),4)
                self.assertFalse(set(trial['previews']) & targets)
                prime=trial['primes']; target=images[str(trial['dataset_index'])]
                self.assertEqual(prime['exact'],trial['dataset_index'])
                self.assertEqual(images[str(prime['same_synset'])]['native_concept'],target['native_concept'])
                self.assertEqual(images[str(prime['same_category'])]['category'],target['category'])
                self.assertNotEqual(images[str(prime['same_category'])]['native_concept'],target['native_concept'])
                self.assertNotEqual(images[str(prime['different_category'])]['category'],target['category'])
                for key,index in prime.items():
                    if key!='exact' and index is not None: self.assertNotIn(index,targets)


if __name__ == '__main__':
    unittest.main()
