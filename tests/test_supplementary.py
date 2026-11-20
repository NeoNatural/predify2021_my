import unittest
from predify_experiments.supplementary import make_manifest, confirmation_trials
from predify_experiments.paper_stimuli import imagenet_supercategory


class SupplementaryDesignTests(unittest.TestCase):
    def test_manifest_pairing_and_confirmation_isolation(self):
        # Full canonical validation-label structure, no images/model required.
        labels = [label for label in range(1000) for _ in range(50)]
        manifest = make_manifest(labels)
        explore = {r['dataset_index'] for r in manifest['exploration']}
        confirm = set(manifest['confirmation_indices'])
        used = set(manifest['excluded_previous_images'])
        self.assertEqual(len(explore), 200)
        self.assertEqual(len(confirm), 200)
        self.assertFalse(explore & confirm)
        self.assertFalse((explore | confirm) & used)
        for trial in manifest['exploration']:
            index = trial['dataset_index']; p = trial['primes']
            self.assertEqual(p['exact'], index)
            self.assertNotEqual(p['same_synset'], index)
            self.assertEqual(labels[p['same_synset']], labels[index])
            self.assertNotEqual(labels[p['same_category']], labels[index])
            self.assertEqual(imagenet_supercategory(labels[p['same_category']]), trial['supercategory'])
            self.assertNotEqual(imagenet_supercategory(labels[p['different_category']]), trial['supercategory'])
            self.assertIsNone(p['neutral'])
            self.assertFalse(set(trial['previews']) & confirm)
            self.assertFalse({x for x in p.values() if x is not None} & confirm)
        confirmation = confirmation_trials(labels, manifest)
        self.assertEqual({r['dataset_index'] for r in confirmation}, confirm)
        for trial in confirmation:
            self.assertFalse(set(trial['previews']) & confirm)
            self.assertEqual(trial['primes']['exact'], trial['dataset_index'])
            self.assertNotEqual(
                imagenet_supercategory(labels[trial['primes']['different_category']]),
                trial['supercategory'],
            )


if __name__ == '__main__':
    unittest.main()
