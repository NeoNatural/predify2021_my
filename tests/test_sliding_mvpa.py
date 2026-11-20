import unittest
import numpy as np
from analysis.sliding_mvpa_20261013 import folds_for

class SlidingFoldsTest(unittest.TestCase):
    def test_grouped_balanced_reproducible(self):
        labels=np.repeat(np.arange(5),60)
        groups=np.concatenate([np.repeat(np.arange(c*10,c*10+10),6) for c in range(5)])
        first=folds_for(labels,groups);second=folds_for(labels,groups)
        coverage=np.zeros(len(labels),int)
        for (train,test),(_,other) in zip(first,second):
            self.assertFalse(set(groups[train])&set(groups[test]))
            np.testing.assert_array_equal(test,other)
            np.testing.assert_array_equal(np.bincount(labels[test]),[12]*5)
            coverage[test]+=1
        np.testing.assert_array_equal(coverage,1)

    def test_rare_class_seven_concepts_and_repeated_noise(self):
        groups=np.concatenate([np.repeat(np.arange(c*20,c*20+(7 if c==3 else 20)),3) for c in range(5)])
        labels=np.concatenate([np.repeat(c,3*(7 if c==3 else 20)) for c in range(5)])
        for train,test in folds_for(labels,groups):
            self.assertEqual(set(labels[test]),set(range(5)))
            self.assertFalse(set(groups[train])&set(groups[test]))

if __name__=='__main__':unittest.main()
