"""Synthetic-only review persistence, access and split invariants."""
import base64
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/test-tools/Yolo11VietnamPlatePoseTraining'))
from car_group_review import CarGroupStore, CarGroupServer, atomic_data, normalize_plate, valid_plate, reviewed_split_plan


class CarReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.dataset=self.root/'dataset';self.dataset.mkdir();self.ws=self.root/'review';self.ws.mkdir();(self.ws/'crops').mkdir()
        rows=[]
        for i,source in enumerate(['car','car','car','car','green','bv1']):
            ident=f'image{i}';image=f'{ident}.jpg';(self.dataset/image).write_bytes(b'synthetic');(self.ws/'crops'/image).write_bytes(b'synthetic')
            rows.append(dict(id=ident,image=image,source_groups=[source],group=source,component=source,
                sha256=f'sha{i}',pixel_sha256=f'pixel{i}',dhash=i*1000003))
        self.manifest=dict(dhash_distance=0,records=rows)
        self.write_manifest()
        self.proposals=dict(dataset=str(self.dataset),manifest_sha256=self.digest(),records=[dict(id=r['id'],crop='crops/'+r['image'],plate='30A12345' if i<2 else f'31A1234{i}',score=.9,candidates=[]) for i,r in enumerate(rows[:4])])
        (self.ws/'proposals.json').write_text(json.dumps(self.proposals),encoding='utf-8')
        self.store=CarGroupStore(self.ws)
    def tearDown(self): self.tmp.cleanup()
    def write_manifest(self): (self.dataset/'manifest.json').write_text(json.dumps(self.manifest),encoding='utf-8')
    def digest(self): return hashlib.sha256((self.dataset/'manifest.json').read_bytes()).hexdigest()
    def request(self,i=0,**updates):
        row=self.store.snapshot()['records'][i]
        return dict(id=row['id'],version=row['version'],plate=row['plate'],manual_group=row['manual_group'],reviewer='Synthetic tester',approved=True,**updates)
    def approve_all(self):
        for i in range(4): self.store.save(self.request(i))
    def test_normalization_does_not_guess_ambiguous_characters(self):
        self.assertEqual(normalize_plate('30a-12.345'),'30A12345');self.assertEqual(normalize_plate('OIB8'),'OIB8')
        self.assertTrue(valid_plate('30A12345'));self.assertFalse(valid_plate('3OA12345'))
    def test_save_conflict_history_and_reload(self):
        req=self.request();self.assertEqual(self.store.save(req),{'version':1})
        with self.assertRaises(FileExistsError): self.store.save(req)
        edit=self.request();edit['approved']=False;edit['plate']='30A54321';self.store.save(edit)
        self.assertTrue((self.ws/'group-history/image0-1.json').exists())
        row=CarGroupStore(self.ws).snapshot()['records'][0]
        self.assertEqual(row['plate'],'30A54321');self.assertFalse(row['approved']);self.assertEqual(row['version'],2)
    def test_invalid_approval_and_unknown_fields(self):
        for change in ({'plate':''},{'reviewer':''},{'approved':'yes'},{'unexpected':1}):
            req=self.request();req.update(change)
            with self.assertRaises(ValueError): self.store.save(req)
        req=self.request();req.update(plate='',approved=False);self.store.save(req)
    def test_manifest_binding(self):
        self.manifest['changed']=True;self.write_manifest()
        with self.assertRaises(ValueError): CarGroupStore(self.ws)
    def test_atomic_save_retries_transient_windows_lock(self):
        import os
        replace=os.replace
        with patch('car_group_review.os.replace',side_effect=[PermissionError('temporary lock'),None]) as mocked:
            atomic_data(self.ws/'retry.json',{'synthetic':True})
            self.assertEqual(mocked.call_count,2)
        # Complete the synthetic mocked replacement to inspect UTF-8 JSON.
        replace(self.ws/'retry.json.tmp',self.ws/'retry.json')
        self.assertEqual(json.loads((self.ws/'retry.json').read_text(encoding='utf-8')),{'synthetic':True})
    def test_pending_blocks_split(self):
        with self.assertRaisesRegex(ValueError,'duyệt đủ'): reviewed_split_plan(self.ws)
    def test_split_keeps_plate_group_and_both_vehicle_types(self):
        self.approve_all();plan=reviewed_split_plan(self.ws);self.assertEqual(plan,reviewed_split_plan(self.ws))
        rows={r['id']:r for r in plan['records']}
        self.assertEqual(rows['image0']['split'],rows['image1']['split'])
        self.assertEqual(rows['image0']['component'],rows['image1']['component'])
        for split in ('train','val'):
            for kind in ('car','motorcycle'): self.assertGreater(plan['counts'][split][kind],0)
    def test_manual_groups_and_duplicate_edges_cannot_be_split(self):
        for i in range(4):
            req=self.request(i);req['manual_group']='one-vehicle';self.store.save(req)
        with self.assertRaisesRegex(ValueError,'Không đủ nhóm'): reviewed_split_plan(self.ws)
    def test_near_duplicate_and_exact_duplicate_edges_override_plate_difference(self):
        self.manifest['records'][2]['dhash']=self.manifest['records'][0]['dhash']
        self.manifest['records'][3]['pixel_sha256']=self.manifest['records'][2]['pixel_sha256']
        self.write_manifest();self.proposals['manifest_sha256']=self.digest()
        (self.ws/'proposals.json').write_text(json.dumps(self.proposals),encoding='utf-8')
        self.store=CarGroupStore(self.ws);self.approve_all()
        # Same plate joins 0/1; dHash joins 0/2; identical pixels join 2/3.
        # The complete car component cannot be cut to force a validation ratio.
        with self.assertRaisesRegex(ValueError,'Không đủ nhóm'): reviewed_split_plan(self.ws)
    def test_http_access_origin_paths_and_persistence(self):
        server=CarGroupServer(('127.0.0.1',0),self.ws,{'username':'test','password':'synthetic-only'},['127.0.0.0/8'])
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        auth='Basic '+base64.b64encode(b'test:synthetic-only').decode()
        def call(path,headers=None,body=None):
            conn=HTTPConnection(*server.server_address);conn.request('POST' if body else 'GET',path,body,headers or {})
            res=conn.getresponse();result=(res.status,res.read());conn.close();return result
        try:
            self.assertEqual(call('/api/state')[0],401)
            headers={'Authorization':auth}
            code,data=call('/api/state',headers);self.assertEqual(code,200);self.assertNotIn(str(self.dataset).encode(),data)
            self.assertEqual(call('/../../proposals.json',headers)[0],404)
            self.assertEqual(call('/',headers)[0],200)
            headers.update({'Content-Type':'application/json','X-Review-Request':'1','Origin':'http://attacker.invalid'})
            self.assertEqual(call('/api/review',headers,json.dumps(self.request()))[0],403)
            headers.pop('Origin');self.assertEqual(call('/api/review',headers,json.dumps(self.request()))[0],200)
            self.assertTrue(CarGroupStore(self.ws).snapshot()['records'][0]['approved'])
        finally: server.shutdown();server.server_close();thread.join()

if __name__=='__main__': unittest.main()
