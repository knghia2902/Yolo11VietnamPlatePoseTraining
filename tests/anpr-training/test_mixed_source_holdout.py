"""Grouping failure modes and notebook isolation; synthetic data only."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT/'scripts/test-tools/Yolo11VietnamPlatePoseTraining'
sys.path.insert(0, str(MODULE))
spec = importlib.util.spec_from_file_location('mixed_package', MODULE/'package_mixed_source_holdout.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def row(identifier, source, dhash, **extra):
    return dict(id=identifier, source_groups=[source], group=source,
                component=identifier, sha256=identifier, pixel_sha256=identifier,
                dhash=dhash, reviewed=True, group_reviewed=True, **extra)


class MixedHoldoutTests(unittest.TestCase):
    def test_whole_sources_and_aliases_stay_together(self):
        a=row('a','car',0); b=row('b','green',255)
        c=row('c','bv1',65535); c['source_groups']=['bv1','cam']
        result=tool.plan_source_holdout([a,b,c],['car','green'],['bv1','cam'],0)
        self.assertEqual({k:v[0] for k,v in result.items()}, {'a':'train','b':'train','c':'val'})

    def test_dhash_bridge_cannot_cross_split(self):
        with self.assertRaisesRegex(ValueError,'cuts'):
            tool.plan_source_holdout([row('a','car',0),row('b','bv1',1)],['car'],['bv1'],4)

    def test_shared_reviewed_group_cannot_cross_split(self):
        a=row('a','car',0); b=row('b','bv1',65535); b['group']=a['group']
        with self.assertRaisesRegex(ValueError,'cuts'):
            tool.plan_source_holdout([a,b],['car'],['bv1'],0)

    def test_old_component_cannot_be_cut(self):
        a=row('a','car',0); b=row('b','bv1',65535); b['component']=a['component']
        with self.assertRaisesRegex(ValueError,'cuts'):
            tool.plan_source_holdout([a,b],['car'],['bv1'],0)

    def test_duplicate_alias_connects_sources(self):
        a=row('a','car',0); a['source_groups']=['car','bv1']
        with self.assertRaisesRegex(ValueError,'cuts'):
            tool.plan_source_holdout([a,row('b','bv1',65535)],['car'],['bv1'],0)

    def test_missing_sources_and_overlap_rejected(self):
        rows=[row('a','car',0),row('b','bv1',65535)]
        for train,val in [(['car'],['cam']),(['car'],['car','bv1']),([],['car','bv1'])]:
            with self.subTest(train=train,val=val),self.assertRaises(ValueError):
                tool.plan_source_holdout(rows,train,val,0)

    def test_unreviewed_or_duplicate_rejected(self):
        a=row('a','car',0); b=row('b','bv1',65535); b['reviewed']=False
        with self.assertRaises(ValueError):
            tool.plan_source_holdout([a,b],['car'],['bv1'],0)
        b['reviewed']=True; b['id']='a'
        with self.assertRaises(ValueError):
            tool.plan_source_holdout([a,b],['car'],['bv1'],0)

    def test_renaming_component_cannot_hide_dhash_bridge(self):
        a=row('a','car',0); b=row('b','bv1',0); b['component']='new'
        with self.assertRaisesRegex(ValueError,'cuts'):
            tool.plan_source_holdout([a,b],['car'],['bv1'],0)

    def test_notebook_pin_isolation_and_embedded_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'notebook.ipynb'
            tool.build_mixed_notebook({'dataset_version':tool.VERSION},'a'*64,target,'b'*64)
            nb=json.loads(target.read_text(encoding='utf-8'))
            cells={c['id']:''.join(c['source']) for c in nb['cells']}
            self.assertIn('FULL_TRAIN = True',cells['drive-setup'])
            self.assertIn('nATime-YOLO11-Pose-v2-sourceholdout',cells['drive-setup'])
            self.assertIn('dataset-mixed-v2-training.zip',cells['drive-setup'])
            self.assertIn('a'*64,cells['pose-03'])
            self.assertIn('b'*64,cells['pose-03'])
            self.assertLess(cells['pose-03'].index('package_stream'),cells['pose-03'].index('archive.extractall'))
            self.assertIn("manifest['dataset_version']",cells['pose-03'])
            self.assertLess(cells['pose-07'].index('def file_sha256'),cells['pose-07'].index('contract = dict'))
            self.assertNotIn('checkpoint-support',cells)
            for cell in nb['cells']:
                if cell['cell_type']=='code':
                    self.assertEqual(cell['outputs'],[])
                    source=''.join(cell['source'])
                    if cell['id']!='pose-01':
                        compile(source,cell['id'],'exec')


if __name__=='__main__':
    unittest.main()
