"""Synthetic exclusions, identity/duplicate grouping and complete v3 preflight."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/test-tools/Yolo11VietnamPlatePoseTraining'))
from car_group_review import CarGroupStore
import prepare_pose_dataset as prepare
from package_reviewed_vehicle_v3 import plan_vehicle_v3_groups, create_vehicle_v3_package, V3_VERSION


class ReviewedVehicleV3Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'source';self.source.mkdir()
        self.workspace=self.root/'review';self.workspace.mkdir();(self.workspace/'crops').mkdir()
        records=[]
        for i,source in enumerate(['car']*5+['green','bv1']):
            ident=f'synthetic{i}';split='val' if source=='bv1' else 'train'
            im=Image.frombytes('RGB',(64,48),random.Random(i).randbytes(64*48*3))
            buffer=io.BytesIO();im.save(buffer,format='PNG');raw=buffer.getvalue()
            size,pixel,dhash=prepare.fingerprints(raw)
            points=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]];text=prepare.label_text(points)
            image=f'images/{split}/{ident}.png';label=f'labels/{split}/{ident}.txt'
            for name,content in ((image,raw),(label,text.encode())):
                path=self.source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
            (self.workspace/'crops'/f'{ident}.jpg').write_bytes(raw)
            records.append(dict(id=ident,split=split,image=image,label=label,size=list(size),sha256=prepare.digest(raw),
                pixel_sha256=pixel,dhash=dhash,label_sha256=prepare.digest(text.encode()),points=points,
                source_groups=[source],group=source,component=source,reviewed=True,group_reviewed=True,reviewer='Synthetic corners'))
        self.manifest=dict(schema=1,reviewed=True,corner_order=['TL','TR','BR','BL'],records=records,
            source_sha256='synthetic',dhash_distance=0,original_images=7,removed_duplicates=0,excluded_images=0,actual_val_fraction=1/7)
        prepare.write_json(self.source/'manifest.json',self.manifest)
        (self.source/'plate_pose.yaml').write_text('train: images/train\nval: images/val\nkpt_shape: [4,3]\nflip_idx: []\nnames: {0: license_plate}\n',encoding='utf-8')
        (self.source/'prepare_pose_dataset.py').write_bytes(Path(prepare.__file__).read_bytes())
        proposals=dict(dataset=str(self.source),manifest_sha256=prepare.digest((self.source/'manifest.json').read_bytes()),records=[
            dict(id=r['id'],crop=f"crops/{r['id']}.jpg",plate='30A12345' if i<2 else f'31A1234{i}',score=.9,candidates=[])
            for i,r in enumerate(records[:5])])
        prepare.write_json(self.workspace/'proposals.json',proposals)
        store=CarGroupStore(self.workspace)
        for row in store.snapshot()['records'][:4]:
            store.save(dict(id=row['id'],version=0,plate=row['plate'],manual_group='',reviewer='Synthetic grouping',approved=True))
        self.snapshot=store.snapshot()
    def plan(self,count=1): return plan_vehicle_v3_groups(self.manifest,self.snapshot,count)
    def test_exclusions_are_exact_pending_set_and_count(self):
        plan=self.plan();self.assertEqual(plan['excluded_ids'],['synthetic4']);self.assertNotIn('synthetic4',plan['records'])
        for count in (0,2):
            with self.assertRaisesRegex(ValueError,'Pending count'): self.plan(count)
    def test_plate_group_preserved_and_both_types_in_each_split(self):
        plan=self.plan();self.assertEqual(plan,self.plan())
        self.assertEqual(plan['records']['synthetic0']['split'],plan['records']['synthetic1']['split'])
        for split in ('train','val'):
            for kind in ('car','motorcycle'): self.assertGreater(plan['counts'][split][kind],0)
    def test_manual_and_dhash_transitive_links_cannot_be_cut(self):
        self.snapshot['records'][1]['manual_group']='synthetic vehicle'
        self.snapshot['records'][2]['manual_group']='synthetic vehicle'
        self.manifest['records'][3]['dhash']=self.manifest['records'][2]['dhash']
        with self.assertRaisesRegex(ValueError,'independent car'): self.plan()
    def test_motorcycle_duplicate_conflict_is_rejected(self):
        self.manifest['records'][6]['pixel_sha256']=self.manifest['records'][5]['pixel_sha256']
        with self.assertRaisesRegex(ValueError,'crosses motorcycle'): self.plan()
    def test_invalid_retained_approval_blocks_packaging(self):
        self.snapshot['records'][0]['plate']=''
        with self.assertRaisesRegex(ValueError,'human-reviewed'): self.plan()
    def test_package_and_notebook_real_preflight(self):
        output=self.root/'release'/'dataset-mixed-v3'
        with contextlib.redirect_stdout(io.StringIO()):
            report=create_vehicle_v3_package(self.workspace,output,1)
        self.assertEqual(report['total_images'],6);self.assertEqual(report['additional_excluded_images'],1)
        self.assertEqual(report['total_excluded_images'],1)
        self.assertTrue((output.parent/'review-snapshot.private.json').is_file())
        notebook=json.loads((output.parent/report['notebook']).read_text(encoding='utf-8'))
        cells={c['id']:''.join(c['source']) for c in notebook['cells']}
        self.assertIn('v3-reviewed-groups',cells['drive-setup']);self.assertNotIn('v2-sourceholdout',cells['drive-setup'])
        self.assertIn(report['zip_sha256'],cells['pose-03']);self.assertIn(report['manifest_sha256'],cells['pose-03'])
        self.assertIn('class DriveCheckpointStore',cells['pose-07']);self.assertIn('vehicle-eval-v3',cells)
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
                if not ''.join(cell['source']).lstrip().startswith(('%','!')):
                    ast.parse(''.join(cell['source']))
        # Execute the actual notebook preflight on a synthetic ZIP, with Colab/Drive stubs.
        import types
        from unittest.mock import patch
        archive=Path(str(output)+'-training.zip')
        colab=types.ModuleType('google.colab');colab.files=types.SimpleNamespace(upload=lambda:{str(archive):b''})
        import tempfile as temp_module
        original=temp_module.mkdtemp
        def local_temp(*args,**kwargs): kwargs['dir']=self.root;return original(*args,**kwargs)
        namespace=dict(Path=Path,DRIVE_DATASET=self.root/'drive-cache.zip')
        with patch.dict(sys.modules,{'google.colab':colab}),patch('tempfile.mkdtemp',side_effect=local_temp),contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cells['pose-03'],'v3-preflight','exec'),namespace)
        self.assertEqual(namespace['manifest']['dataset_version'],V3_VERSION)
        self.assertTrue(namespace['DRIVE_DATASET'].is_file())
        # A wrong ZIP is rejected by the actual notebook hash guard.
        namespace['DRIVE_DATASET'].write_bytes(b'wrong archive')
        with patch.dict(sys.modules,{'google.colab':colab}),patch('tempfile.mkdtemp',side_effect=local_temp),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(AssertionError,'Sai ZIP v3'):
                exec(compile(cells['pose-03'],'v3-preflight','exec'),namespace)

if __name__=='__main__': unittest.main()
