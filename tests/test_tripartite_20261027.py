import json
from pathlib import Path
import unittest
import numpy as np
import torch
from predify_experiments.tripartite_20261027 import OUT,conditions,make_canvas,restore_state,state_snapshot


class TripartiteTests(unittest.TestCase):
    def test_geometry_uses_whole_canvas_and_disjoint_regions(self):
        image=torch.ones(3,224,224)
        result=make_canvas([(image,'left'),(image*2,'center'),(image*3,'right')])
        self.assertEqual(tuple(result.shape),(3,224,224))
        for x,value in [(5,1),(80,2),(155,3)]:
            np.testing.assert_array_equal(result[:,80:144,x:x+64].numpy(),np.full((3,64,64),value))

    def test_frozen_data_are_concept_disjoint_and_confirmation_fresh(self):
        both=[]
        for dataset in ('imagenet','things'):
            m=json.loads((OUT/dataset/'manifest.json').read_text())
            both.append(m['concept_splits'])
            protected={r['dataset_index'] for s in ('train','dev','test') for r in m[s]}
            self.assertEqual(len(protected),176)
            for s,n in [('train',80),('dev',32),('test',64)]:
                self.assertEqual(len(m[s]),n)
                for c in range(4):
                    rows=[r for r in m[s] if r['category_index']==c]
                    self.assertEqual(sum(r['slot']==2 for r in rows),len(rows)//2)
                for r in m[s]:
                    self.assertFalse(set(r['previews'])&protected)
                    self.assertFalse({i for k,i in r['primes'].items() if k!='exact' and i is not None}&protected)
                    if s=='test':self.assertNotIn(r['dataset_index'],m['historical_seen'])
            for a,b in [('train','dev'),('train','test'),('dev','test')]:
                self.assertFalse(set(m['concept_splits'][a])&set(m['concept_splits'][b]))
        self.assertEqual(both[0],both[1])

    def test_all_strength_and_carrier_conditions_are_present(self):
        rows=conditions();self.assertEqual(len(rows),28)
        self.assertEqual(len({r['name'] for r in rows}),28)
        self.assertEqual(sum(r['coeff']==.0125 for r in rows),10)
        self.assertEqual(sum(r['carrier']=='pcoder' for r in rows),5)

    def test_training_sampler_equal_image_and_carrier_weights(self):
        from analysis.tripartite_20261027 import training_rows
        m=json.loads((OUT/'imagenet'/'manifest.json').read_text())
        keys=[(r['dataset_index'],n) for r in m['train'][:4] for n in m['noise_seeds']['train']]
        fake=dict(keys=keys,y=np.repeat(np.arange(4),3),features={c['name']:np.zeros((12,15,2),np.float16) for c in conditions()})
        x,y,w,rows=training_rows(fake,m)
        self.assertEqual(len(x),4*270)
        self.assertEqual(w.sum(),4*180)
        for index in {k[0] for k in keys}:
            for prefix in ('none','pcoder','hebb','joint'):
                self.assertEqual(sum(r['weight'] for r in rows if r['image']==index and (r['condition']==prefix or r['condition'].startswith(prefix+'_'))),45)

    def test_three_memory_pairwise_contrasts_exist_for_every_content_and_strength(self):
        from analysis.tripartite_20261027 import contrast_definitions
        rows=contrast_definitions()
        for strength in ('strong','weak'):
            for identity in ('exact','same_concept','same_category','different_category','neutral'):
                for contrast in ('joint-pcoder','joint-hebb','pcoder-hebb'):
                    self.assertEqual(sum(r['strength']==strength and r['identity']==identity and r['contrast']==contrast for r in rows),1)

    def test_noise_repetition_does_not_increase_cluster_sample_size(self):
        from analysis.tripartite_20261027 import cluster_stats
        groups=np.repeat(np.arange(32),6);labels=np.repeat(np.repeat(np.arange(4),8),6)
        values=np.repeat(np.tile(np.arange(8),4),6).astype(float)
        first=cluster_stats(values,dict(groups=groups,y=labels),draws=200)
        second=cluster_stats(np.repeat(values,3),dict(groups=np.repeat(groups,3),y=np.repeat(labels,3)),draws=200)
        self.assertEqual(first,second);self.assertEqual(first['concepts'],32)


if __name__=='__main__':unittest.main()
