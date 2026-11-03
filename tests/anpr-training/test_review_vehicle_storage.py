import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/test-tools/Yolo11VietnamPlatePoseTraining'))
from curate_acs_training_pool import ExpansionReviewStore, expansion_db, expansion_meta


class VehicleStorageTests(unittest.TestCase):
    def test_classification_preserves_labels_and_blocks_stale_frozen_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/'source.jpg'; image.write_bytes(b'synthetic-image')
            original={'points':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],'plate':'59A12345','corners_reviewed':True,'review_scope':'corners'}
            with expansion_db(root) as db:
                expansion_meta(db,'manifest_sha256','fixture')
                expansion_meta(db,'image_storage_root',str(root/'storage'))
                for n,kind,edit in [('a','acs',original),('b','baseline',{}),('c','acs',{'exclude':True})]:
                    db.execute('INSERT INTO images(id,kind,path,sha,stage,edit) VALUES(?,?,?,?,?,?)',(n*64,kind,str(image if n=='a' else root/(n+'.jpg')),hashlib.sha256(image.read_bytes()).hexdigest(),'ready',json.dumps(edit)))
            store=ExpansionReviewStore(root)
            request=dict(id='a'*64,version=0,action='classify',vehicle_type='car',reviewer='Tester')
            result=store.save(request)
            self.assertEqual(result['vehicle_type'],'car')
            with expansion_db(root) as db:
                row=db.execute('SELECT * FROM images WHERE id=?',('a'*64,)).fetchone()
                after=json.loads(row['edit'])
                for k,v in original.items(): self.assertEqual(after[k],v)
                self.assertIn(str(Path('train')/'images'),row['path']);self.assertIn('o-to',row['path'])
                self.assertEqual(Path(row['path']).read_bytes(),b'synthetic-image');self.assertFalse(image.exists())
                self.assertEqual(db.execute('SELECT count(*) FROM history').fetchone()[0],1)
            with self.assertRaises(RuntimeError): store.save(request)
            for n in ['b','c']:
                with self.assertRaises(ValueError): store.save(dict(request,id=n*64))
            with self.assertRaises(ValueError): store.save(dict(request,version=1,vehicle_type='../invalid'))
            result=store.save(dict(request,version=1,vehicle_type='motorcycle'))
            self.assertEqual(result['vehicle_type'],'motorcycle')
            train=root/'storage'/'du-lieu-yolo-point'/'train'
            self.assertFalse((train/'images'/'o-to'/('a'*64+'.jpg')).exists())
            self.assertTrue((train/'images'/'xe-may'/('a'*64+'.jpg')).is_file())
            self.assertFalse((train/'labels'/'o-to'/('a'*64+'.txt')).exists())
            self.assertTrue((train/'labels'/'xe-may'/('a'*64+'.txt')).is_file())
            page=store.listing('acs','all',999,vehicle='motorcycle',anchor='a'*64)
            self.assertEqual(page['offset'],0)
            self.assertEqual(page['total'],1)
            self.assertEqual(page['records'][0]['id'],'a'*64)

    def test_destination_conflict_keeps_source_and_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'source.jpg';image.write_bytes(b'source')
            target=root/'storage'/'anh-goc'/'ACS'/'o-to'/('a'*64+'.jpg')
            target.parent.mkdir(parents=True);target.write_bytes(b'other')
            with expansion_db(root) as db:
                expansion_meta(db,'manifest_sha256','fixture')
                expansion_meta(db,'image_storage_root',str(root/'storage'))
                db.execute('INSERT INTO images(id,kind,path,sha,stage) VALUES(?,?,?,?,?)',('a'*64,'acs',str(image),hashlib.sha256(b'source').hexdigest(),'ready'))
            store=ExpansionReviewStore(root)
            with self.assertRaises(ValueError):
                store.save(dict(id='a'*64,version=0,action='classify',vehicle_type='car',reviewer='Tester'))
            self.assertEqual(image.read_bytes(),b'source');self.assertEqual(target.read_bytes(),b'other')
            with expansion_db(root) as db:self.assertEqual(db.execute('SELECT version FROM images WHERE id=?',('a'*64,)).fetchone()[0],0)


if __name__=='__main__': unittest.main()
