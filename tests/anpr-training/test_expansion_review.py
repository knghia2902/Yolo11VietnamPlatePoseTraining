import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

TOOLS=Path(__file__).resolve().parents[2]/'scripts/test-tools/Yolo11VietnamPlatePoseTraining'
sys.path.insert(0,str(TOOLS))
import curate_acs_training_pool as expansion


class ExpansionReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'previews').mkdir(); (self.root/'crops').mkdir(); (self.root/'imports').mkdir()
        self.points=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        self.val='a'*64; self.new='b'*64
        with expansion.expansion_db(self.root) as db:
            expansion.expansion_meta(db,'manifest_sha256','fixed-v3-hash')
            for image_id,kind in ((self.val,'validation'),(self.new,'acs')):
                p=dict(points=self.points,near=[])
                db.execute('INSERT INTO images(id,kind,path,stage,proposal) VALUES(?,?,?,?,?)',
                           (image_id,kind,str(self.root/image_id),'ready',json.dumps(p)))
            db.commit()
        self.store=expansion.ExpansionReviewStore(self.root)

    def tearDown(self): self.temp.cleanup()

    def test_approve_validation_moves_labels_supports_undo_and_locks_saved_corners(self):
        original=self.root/'anh-goc'/'ACS'/'xe-may'/(self.new+'.jpg')
        original.parent.mkdir(parents=True);original.write_bytes(b'image fixture')
        with expansion.expansion_db(self.root) as db:
            expansion.expansion_meta(db,'image_storage_root',str(self.root))
            db.execute('UPDATE images SET path=? WHERE id=?',(str(original),self.new))
        request=dict(id=self.new,version=0,action='approve_validation',edit=dict(points=self.points,reviewer='Tester',single_plate_reviewed=True))
        import copy
        row=self.store.save(copy.deepcopy(request))
        dest=self.root/'du-lieu-yolo-point'/'validation'/'images'/'xe-may'/original.name
        label=self.root/'du-lieu-yolo-point'/'validation'/'labels'/'xe-may'/(original.stem+'.txt')
        self.assertEqual(row['kind'],'validation');self.assertFalse(row['train_ready'])
        self.assertTrue(dest.is_file());self.assertTrue(label.is_file());self.assertFalse(original.exists())
        self.assertEqual(self.store.listing('all','validation')['total'],1)
        with self.assertRaises(RuntimeError):self.store.save(copy.deepcopy(request))
        row=self.store.save(dict(id=self.new,version=1,action='undo_corners',edit={}))
        self.assertEqual(row['kind'],'acs');self.assertTrue(original.is_file());self.assertFalse(label.exists())
        self.store.save(dict(copy.deepcopy(request),version=2))
        before=label.read_bytes()
        self.save(self.new,'59A12345',version=3,points=self.points)
        self.assertEqual(label.read_bytes(),before)

    def test_cross_split_exact_copy_is_rejected_before_any_move(self):
        import hashlib
        source=self.root/'anh-goc'/'ACS'/'xe-may'/(self.new+'.jpg')
        source.parent.mkdir(parents=True);source.write_bytes(b'duplicate fixture')
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        for into_validation in (True,False):
            split='train' if into_validation else 'validation'
            other=self.root/'du-lieu-yolo-point'/split/'images'/'xe-may'/(self.val+'.jpg')
            other.parent.mkdir(parents=True,exist_ok=True);other.write_bytes(source.read_bytes())
            with expansion.expansion_db(self.root) as db:
                expansion.expansion_meta(db,'image_storage_root',str(self.root))
                db.execute('UPDATE images SET path=?,sha=? WHERE id=?',(str(source),sha,self.new))
                db.execute('UPDATE images SET kind=?,path=?,sha=? WHERE id=?',('baseline' if into_validation else 'validation',str(other),sha,self.val))
            request=dict(id=self.new,version=0,action='approve_validation' if into_validation else 'approve_corners',edit=dict(points=self.points,reviewer='Tester',single_plate_reviewed=True))
            with self.assertRaisesRegex(ValueError,'tập còn lại'):self.store.save(request)
            self.assertTrue(source.is_file())
            with expansion.expansion_db(self.root) as db:
                self.assertEqual(db.execute('select version from images where id=?',(self.new,)).fetchone()[0],0)

    def test_validation_label_failure_rolls_back_image_and_database(self):
        from unittest.mock import patch
        source=self.root/'anh-goc'/'ACS'/'xe-may'/(self.new+'.jpg')
        source.parent.mkdir(parents=True);source.write_bytes(b'fixture')
        with expansion.expansion_db(self.root) as db:
            expansion.expansion_meta(db,'image_storage_root',str(self.root))
            db.execute('UPDATE images SET path=? WHERE id=?',(str(source),self.new))
        with patch.object(expansion,'sync_labels',side_effect=OSError('simulated label write failure')):
            with self.assertRaises(OSError):
                self.store.save(dict(id=self.new,version=0,action='approve_validation',edit=dict(points=self.points,reviewer='Tester',single_plate_reviewed=True)))
        self.assertTrue(source.is_file())
        with expansion.expansion_db(self.root) as db:
            row=db.execute('select kind,path,version from images where id=?',(self.new,)).fetchone()
            self.assertEqual((row['kind'],row['path'],row['version']),('acs',str(source),0))

    def test_existing_train_can_move_to_validation_and_undo_restores_train_label(self):
        source=self.root/'anh-goc'/'ACS'/'xe-may'/(self.new+'.jpg')
        source.parent.mkdir(parents=True);source.write_bytes(b'fixture')
        with expansion.expansion_db(self.root) as db:
            expansion.expansion_meta(db,'image_storage_root',str(self.root))
            db.execute('UPDATE images SET path=? WHERE id=?',(str(source),self.new))
        self.approve_corners()
        train=self.root/'du-lieu-yolo-point'/'train'
        train_image=train/'images'/'xe-may'/source.name
        train_label=train/'labels'/'xe-may'/(source.stem+'.txt')
        label_bytes=train_label.read_bytes()
        self.store.save(dict(id=self.new,version=1,action='approve_validation',edit=dict(points=self.points,reviewer='Tester',single_plate_reviewed=True)))
        self.assertFalse(train_image.exists());self.assertFalse(train_label.exists())
        self.store.save(dict(id=self.new,version=2,action='undo_corners',edit={}))
        self.assertTrue(train_image.is_file());self.assertEqual(train_label.read_bytes(),label_bytes)

    def test_validation_with_reviewed_corners_is_not_counted_as_train(self):
        (self.root/self.val).write_bytes(b'held out validation')
        with expansion.expansion_db(self.root) as db:
            expansion.expansion_meta(db,'image_storage_root',str(self.root))
            db.execute('UPDATE images SET edit=? WHERE id=?',
                       (json.dumps(dict(corners_reviewed=True)),self.val))
        self.assertEqual(self.store.listing('all','corners_done')['total'],0)
        row=self.store.listing('validation','all')['records'][0]
        self.assertFalse(row['train_ready'])
        self.assertTrue(row['corners_reviewed'])
        page=self.store.listing('all','validation')
        self.assertEqual(page['total'],1)
        self.assertEqual(page['records'][0]['kind'],'validation')

    def test_saving_baseline_identity_preserves_existing_training_label(self):
        image=self.root/'du-lieu-yolo-point'/'train'/'images'/'xe-may'/(self.new+'.jpg')
        label=self.root/'du-lieu-yolo-point'/'train'/'labels'/'xe-may'/(self.new+'.txt')
        image.parent.mkdir(parents=True);label.parent.mkdir(parents=True)
        image.write_bytes(b'frozen baseline image')
        original=b'0 0.5 0.5 0.8 0.8 0.1 0.1 2 0.9 0.1 2 0.9 0.9 2 0.1 0.9 2\n'
        label.write_bytes(original)
        with expansion.expansion_db(self.root) as db:
            expansion.expansion_meta(db,'image_storage_root',str(self.root))
            db.execute("UPDATE images SET kind='baseline',path=? WHERE id=?",(str(image),self.new))
        self.save(self.new,'59A12345',points=self.points)
        self.assertEqual(label.read_bytes(),original)
        self.assertEqual(image.read_bytes(),b'frozen baseline image')

    def test_live_counts_include_existing_baseline_and_hide_removed_sources(self):
        image=self.root/self.new
        image.write_bytes(b'fixture')
        baseline=self.root/('c'*64)
        baseline.write_bytes(b'baseline')
        excluded=self.root/('d'*64)
        excluded.write_bytes(b'excluded')
        with expansion.expansion_db(self.root) as db:
            expansion.expansion_meta(db,'image_storage_root',str(self.root))
            db.execute('UPDATE images SET edit=? WHERE id=?',
                       (json.dumps(dict(corners_reviewed=True,review_scope='corners')),self.new))
            for image_id,kind,path,stage in [('c'*64,'baseline',baseline,'ready'),
                                            ('d'*64,'acs',excluded,'excluded'),
                                            ('e'*64,'acs',self.root/'missing.jpg','ready'),
                                            ('f'*64,'acs',r'\\server\share\image.jpg','excluded')]:
                db.execute('INSERT INTO images(id,kind,path,stage,proposal) VALUES(?,?,?,?,?)',
                           (image_id,kind,str(path),stage,json.dumps(dict(points=self.points))))
        page=self.store.listing('all','corners_done')
        self.assertEqual(page['total'],2)
        self.assertEqual(page['counts'],{'acs':1,'baseline':1})
        self.assertTrue(all(r['train_ready'] for r in page['records']))
        self.assertFalse(next(r for r in page['records'] if r['kind']=='baseline')['corners_reviewed'])
        self.assertEqual(self.store.listing('all','all')['total'],2)
        image.unlink()
        self.assertEqual(self.store.listing('acs','corners_done')['total'],0)
        self.assertEqual(self.store.listing('all','all')['counts'],{'baseline':1})

    def save(self,image_id,plate,version=0,**extra):
        edit=dict(plate=plate,reviewer='reviewer',identity_reviewed=True,single_plate_reviewed=True,**extra)
        return self.store.save(dict(id=image_id,version=version,edit=edit))

    def test_queue_cache_invalidates_after_external_review(self):
        first=self.store.listing('acs','pending')
        self.assertEqual(first['total'],1)
        identity=self.store.identity_index
        def unexpected_scan(db): raise AssertionError('Unchanged queue was rescanned')
        self.store.identity_index=unexpected_scan
        cached=self.store.listing('acs','pending',0,anchor=self.new)
        self.assertEqual(cached['records'][0]['id'],self.new)
        self.store.identity_index=identity
        with expansion.expansion_db(self.root) as db:
            db.execute('UPDATE images SET edit=?,version=version+1 WHERE id=?',(json.dumps({'exclude':True}),self.new))
        self.assertEqual(self.store.listing('acs','pending')['total'],0)

    def test_frozen_vehicle_classification_is_sha_guarded_and_filterable(self):
        with expansion.expansion_db(self.root) as db:
            db.execute('UPDATE images SET sha=? WHERE id=?',('fixed-image-hash',self.val))
        self.store.frozen_vehicle_types={self.val:{'sha256':'fixed-image-hash','vehicle_type':'car'}}
        page=self.store.listing('validation','all',vehicle='car')
        self.assertEqual(page['total'],1)
        self.assertEqual(page['records'][0]['vehicle_type'],'car')
        self.store.frozen_vehicle_types[self.val]['sha256']='wrong-hash'
        self.assertEqual(self.store.listing('validation','all',vehicle='car')['total'],0)

    def test_candidate_waits_for_validation_identity(self):
        row=self.save(self.new,'59A12345',points=self.points,corners_reviewed=True)
        self.assertEqual(row['status'],'waiting_validation')
        self.assertFalse(self.store.export_reviews()['training_ready'])
        self.assertEqual(self.store.listing('acs','pending')['records'],[])
        self.assertEqual(self.store.listing('acs','approved')['records'][0]['status'],'waiting_validation')

    def approve_corners(self, image_id=None, version=0, **fields):
        return self.store.save(dict(id=image_id or self.new,version=version,action='approve_corners',
            edit=dict(points=self.points,reviewer='Corner reviewer',single_plate_reviewed=True,**fields)))

    def test_return_corners_preserves_other_fields_and_checks_revision(self):
        self.approve_corners()
        with expansion.expansion_db(self.root) as db:
            before=json.loads(db.execute('SELECT edit FROM images WHERE id=?',(self.new,)).fetchone()[0])
        result=self.store.save(dict(id=self.new,version=1,action='return_corners',edit={}))
        self.assertFalse(result['corners_reviewed'])
        with expansion.expansion_db(self.root) as db:
            after=json.loads(db.execute('SELECT edit FROM images WHERE id=?',(self.new,)).fetchone()[0])
        self.assertEqual(after,dict(before,corners_reviewed=False))
        with self.assertRaises(RuntimeError):
            self.store.save(dict(id=self.new,version=1,action='return_corners',edit={}))
        with self.assertRaises(ValueError):
            self.store.save(dict(id=self.val,version=0,action='return_corners',edit={}))

    def test_undo_corner_approval_restores_previous_corners(self):
        self.approve_corners()
        result=self.store.save(dict(id=self.new,version=1,action='undo_corners',edit={}))
        self.assertFalse(result['corners_reviewed'])
        self.assertEqual(self.store.listing('acs','corners')['total'],1)
        with self.assertRaises(RuntimeError):
            self.store.save(dict(id=self.new,version=1,action='undo_corners',edit={}))

    def test_corner_approval_moves_file_and_undo_moves_it_back(self):
        original=self.root/'anh-goc'/'ACS'/'xe-may'/(self.new+'.jpg')
        original.parent.mkdir(parents=True);original.write_bytes(b'image fixture')
        with expansion.expansion_db(self.root) as db:
            expansion.expansion_meta(db,'image_storage_root',str(self.root))
            db.execute('UPDATE images SET path=? WHERE id=?',(str(original),self.new));db.commit()
        self.approve_corners()
        approved=self.root/'du-lieu-yolo-point'/'train'/'images'/'xe-may'/original.name
        self.assertTrue(approved.is_file());self.assertFalse(original.exists())
        label=self.root/'du-lieu-yolo-point'/'train'/'labels'/'xe-may'/(original.stem+'.txt')
        self.assertEqual(len(label.read_text().split()),17)
        self.store.save(dict(id=self.new,version=1,action='undo_corners',edit={}))
        self.assertTrue(original.is_file());self.assertFalse(approved.exists())
        self.assertFalse(label.exists())
        with expansion.expansion_db(self.root) as db:
            self.assertEqual(db.execute('SELECT path FROM images WHERE id=?',(self.new,)).fetchone()[0],str(original))

    def test_corner_only_allows_unreadable_identity_but_never_full_approval(self):
        self.save(self.val,'59B12345')
        result=self.approve_corners()
        self.assertEqual(result['status'],'corners_done')
        self.assertTrue(result['corners_reviewed'])
        self.assertFalse(result['identity_reviewed'])
        self.assertEqual(result['plate'],'')
        self.assertEqual(self.store.listing('acs','corners')['total'],0)
        self.assertEqual(self.store.listing('acs','corners_done')['total'],1)
        self.assertEqual(self.store.listing('all','corners_done')['total'],1)
        self.assertEqual(self.store.listing('acs','approved')['total'],0)
        exported=self.store.export_reviews()
        self.assertFalse(exported['training_ready'])
        self.assertEqual(next(r for r in exported['records'] if r['id']==self.new)['status'],'corners_done')
        with expansion.expansion_db(self.root) as db:
            self.assertEqual(db.execute('SELECT value FROM history WHERE id=? AND version=0',(self.new,)).fetchone()[0],'{}')
        with self.assertRaises(RuntimeError): self.approve_corners()

    def test_corner_only_preserves_identity_comparison_and_ocr(self):
        old=dict(plate='59A12345',reviewer='Identity reviewer',identity_reviewed=True,
                 validation_comparison='different',validation_comparison_digest='prior-digest',near_checked=False)
        proposal=dict(points=self.points,near=[self.val],ocr=dict(plate='59A12345',confidence=.4))
        with expansion.expansion_db(self.root) as db:
            db.execute('UPDATE images SET edit=?,proposal=? WHERE id=?',(json.dumps(old),json.dumps(proposal),self.new));db.commit()
        self.approve_corners()
        with expansion.expansion_db(self.root) as db:
            r=db.execute('SELECT * FROM images WHERE id=?',(self.new,)).fetchone();e=json.loads(r['edit'])
            for field in ('plate','identity_reviewed','validation_comparison','validation_comparison_digest','near_checked'):
                self.assertEqual(e[field],old[field])
            self.assertEqual(json.loads(r['proposal']),proposal)

    def test_corner_only_keeps_suspected_overlap_visible(self):
        with expansion.expansion_db(self.root) as db:
            db.execute('UPDATE images SET proposal=? WHERE id=?',
                (json.dumps(dict(points=self.points,ocr=dict(plate='59A12345'))),self.val))
            db.execute('UPDATE images SET proposal=? WHERE id=?',
                (json.dumps(dict(points=self.points,near=[self.val],ocr=dict(plate='59A12345'))),self.new));db.commit()
        result=self.approve_corners()
        self.assertEqual(result['status'],'suspected_overlap')
        self.assertFalse(result['identity_reviewed'])
        self.assertEqual(self.store.listing('acs','corners')['total'],0)
        self.assertEqual(self.store.listing('acs','matched')['total'],1)

    def test_corner_only_rejects_frozen_excluded_and_missing_attestation(self):
        with self.assertRaises(ValueError): self.approve_corners(self.val)
        for field,value in [('single_plate_reviewed',False),('reviewer',''),('points',[])]:
            edit=dict(points=self.points,reviewer='Tester',single_plate_reviewed=True);edit[field]=value
            with self.assertRaises(ValueError): self.store.save(dict(id=self.new,version=0,action='approve_corners',edit=edit))
        for stage in ('duplicate','error','crop_only','pending'):
            with expansion.expansion_db(self.root) as db:
                db.execute('UPDATE images SET stage=? WHERE id=?',(stage,self.new));db.commit()
            with self.assertRaises(ValueError): self.approve_corners()

    def test_only_explicit_full_approval_can_promote_corner_only_review(self):
        self.save(self.val,'59B12345')
        self.approve_corners()
        row=self.save(self.new,'59A12345',version=1,points=self.points,corners_reviewed=True)
        self.assertEqual(row['status'],'corners_done')
        edit=dict(plate='59A12345',reviewer='Tester',identity_reviewed=True,single_plate_reviewed=True,
                  points=self.points,corners_reviewed=True)
        row=self.store.save(dict(id=self.new,version=2,action='approve',edit=edit))
        self.assertEqual(row['status'],'approved')

    def test_later_validation_identity_revokes_candidate_eligibility(self):
        self.save(self.new,'59A12345',points=self.points,corners_reviewed=True)
        self.save(self.val,'59A12345')
        rows=self.store.listing('acs','blocked')['records']
        self.assertEqual(rows[0]['status'],'validation_overlap')

    def test_independent_vehicle_eligible_only_after_complete_index(self):
        self.save(self.val,'59B12345')
        row=self.save(self.new,'59A12345',points=self.points,corners_reviewed=True)
        self.assertEqual(row['status'],'approved')

    def test_ocr_alone_never_approves(self):
        with expansion.expansion_db(self.root) as db:
            db.execute('UPDATE images SET proposal=? WHERE id=?',
                (json.dumps(dict(points=self.points,ocr=dict(plate='59A12345'))),self.new));db.commit()
        self.assertEqual(self.store.listing()['records'][0]['status'],'ready')

    def test_corner_overlay_only_hides_exact_visually_checked_revision(self):
        with expansion.expansion_db(self.root) as db:
            db.execute('UPDATE images SET sha=? WHERE id=?', ('c'*64,self.new));db.commit()
        entry=dict(status='checked_visual',version=0,sha256='c'*64,points=self.points)
        path=self.root/'corner-review-index.json'
        path.write_text(json.dumps(dict(schema=1,records={self.new:entry})),encoding='utf-8')
        self.assertEqual(self.store.listing('acs','corners')['total'],0)
        self.assertEqual(self.store.listing('acs','pending')['total'],1)
        self.assertFalse(self.store.listing('acs','all')['records'][0]['corners_reviewed'])
        self.assertFalse(self.store.export_reviews()['training_ready'])
        for key,value in [('status','needs_human'),('version',1),('sha256','d'*64),('points',[])]:
            with self.subTest(key=key):
                path.write_text(json.dumps(dict(schema=1,records={self.new:dict(entry,**{key:value})})),encoding='utf-8')
                self.assertEqual(self.store.listing('acs','corners')['total'],1)

    def test_corner_overlay_missing_or_corrupt_keeps_unknowns_visible(self):
        path=self.root/'corner-review-index.json'
        for content in [None,'{bad','[]','{"schema":1,"records":[]}', '{"schema":2,"records":{}}']:
            with self.subTest(content=content):
                if content is not None: path.write_text(content,encoding='utf-8')
                self.assertEqual(self.store.listing('acs','corners')['total'],1)
        self.assertEqual(self.store.listing('validation','corners')['total'],0)

    def test_corner_queue_excludes_human_approved_and_excluded(self):
        self.save(self.new,'59A12345',points=self.points,corners_reviewed=True)
        self.assertEqual(self.store.listing('acs','corners')['total'],0)

    def test_secondary_ocr_can_recover_missing_prefix_without_approval(self):
        import types
        from unittest.mock import patch
        class SyntheticOCR:
            def __init__(self,**kwargs): pass
            def __call__(self,path,**kwargs):
                if kwargs['use_det']:
                    return [[[[0,0],[10,0],[10,10],[0,10]],'A12345',.9]],None
                return [['88A12345',.95]],None
        (self.root/'crops'/(self.val+'.jpg')).write_bytes(b'synthetic-not-decoded-by-stub')
        with patch.dict(sys.modules,{'rapidocr_onnxruntime':types.SimpleNamespace(RapidOCR=SyntheticOCR)}):
            self.assertEqual(expansion.expansion_ocr(self.root,limit=1),1)
        item=self.store.listing('validation','pending')['records'][0]
        self.assertEqual(item['plate'],'88A12345')
        self.assertFalse(item['identity_reviewed'])
        self.assertEqual(item['status'],'identity_pending')

    def test_validation_corners_and_membership_are_immutable(self):
        for extra in (dict(points=[[0,0],[1,0],[1,1],[0,1]]),dict(exclude=True),dict(corners_reviewed=True)):
            with self.assertRaises(ValueError): self.save(self.val,'59A12345',**extra)
        self.assertEqual(self.store.listing('validation')['records'][0]['points'],self.points)

    def test_optimistic_version_and_restart_persistence(self):
        self.save(self.val,'59A12345')
        with self.assertRaises(RuntimeError): self.save(self.val,'59A12346')
        second=expansion.ExpansionReviewStore(self.root)
        self.assertEqual(second.listing('validation','all')['records'][0]['plate'],'59A12345')
        with expansion.expansion_db(self.root) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM history').fetchone()[0],1)

    def test_near_match_requires_explicit_comparison(self):
        with expansion.expansion_db(self.root) as db:
            db.execute('UPDATE images SET proposal=? WHERE id=?',
                (json.dumps(dict(points=self.points,near=[self.val])),self.new));db.commit()
        with self.assertRaises(ValueError):
            self.save(self.new,'59A12345',points=self.points,corners_reviewed=True)

    def test_duplicate_crop_multiple_not_approved(self):
        for stage in ('duplicate','crop_only','error','pending'):
            with expansion.expansion_db(self.root) as db:
                db.execute('UPDATE images SET stage=? WHERE id=?',(stage,self.new));db.commit()
            with self.assertRaises(ValueError):
                self.save(self.new,'59A12345',points=self.points,corners_reviewed=True)

    def test_asset_allowlist_and_traversal(self):
        self.assertIsNone(self.store.asset('../curation.sqlite'))
        unknown='c'*64;(self.root/'previews'/(unknown+'.jpg')).write_bytes(b'secret')
        self.assertIsNone(self.store.asset(unknown))

    def test_import_zip_rejects_traversal_and_requires_provenance(self):
        z=self.root/'bad.zip'
        with zipfile.ZipFile(z,'w') as archive: archive.writestr('../secret.jpg',b'bad')
        with self.assertRaises(ValueError):
            expansion.expansion_import(self.root,z,'https://universe.roboflow.com/test/source','CC BY 4.0')
        with self.assertRaises(ValueError):
            expansion.expansion_import(self.root,z,'https://example.org/','')

    def test_import_preserves_attribution_and_boxes_without_approving_corners(self):
        archive=self.root/'synthetic.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('train/images/synthetic.jpg',b'synthetic-fixture-not-run-through-inference')
            z.writestr('train/labels/synthetic.txt','0 0.5 0.5 0.4 0.2')
            z.writestr('README.dataset.txt','Synthetic attribution; CC BY 4.0')
        url='https://universe.roboflow.com/synthetic/fixture/dataset/1'
        self.assertEqual(expansion.expansion_import(self.root,archive,url,'CC BY 4.0'),1)
        self.assertEqual(expansion.expansion_import(self.root,archive,url,'CC BY 4.0'),1)
        with expansion.expansion_db(self.root) as db:
            rows=db.execute("SELECT * FROM images WHERE kind='roboflow'").fetchall()
            self.assertEqual(len(rows),1)
            proposal=json.loads(rows[0]['proposal'])
            self.assertEqual(proposal['source_detection_boxes'],[[.5,.5,.4,.2]])
            self.assertNotIn('points',proposal)
            self.assertEqual(rows[0]['edit'],'{}')
        self.assertEqual(len(list((self.root/'imports').glob('*/source-readmes.json'))),1)

    def test_plate_normalization_never_guesses_ambiguous_characters(self):
        self.assertEqual(expansion.expansion_plate('59-O1 12.345'),'59O112345')
        self.assertNotEqual(expansion.expansion_plate('59-O1 12.345'),'590112345')

    def test_near_hash_chunks_cover_four_differences(self):
        with expansion.expansion_db(self.root) as db:
            db.execute('UPDATE images SET dhash=? WHERE id=?',('0000000000000000',self.val))
            expansion.expansion_index_hash(db,self.val,'0000000000000000')
            four=sum(1<<(13*i) for i in range(4))
            self.assertEqual(expansion.expansion_near_hashes(db,self.new,format(four,'016x')),[self.val])
            self.assertEqual(expansion.expansion_near_hashes(db,self.new,format(four|1<<52,'016x')),[])

    def test_model_multiple_proposals_can_be_corrected_but_source_multi_cannot(self):
        with expansion.expansion_db(self.root) as db:
            db.execute("UPDATE images SET stage='multiple' WHERE id=?",(self.new,));db.commit()
        row=self.save(self.new,'59A12345',points=self.points,corners_reviewed=True)
        self.assertEqual(row['status'],'waiting_validation')
        with expansion.expansion_db(self.root) as db:
            db.execute('UPDATE images SET proposal=? WHERE id=?',
                       (json.dumps(dict(points=self.points,source_detection_boxes=[[.5,.5,.2,.2]]*2)),self.new));db.commit()
        with self.assertRaises(ValueError):
            self.save(self.new,'59A12345',version=1,points=self.points,corners_reviewed=True)

    def test_non_vietnam_or_no_full_plate_attestation_cannot_approve(self):
        edit=dict(plate='59A12345',reviewer='Tester',identity_reviewed=True,
                  corners_reviewed=True,points=self.points,single_plate_reviewed=False)
        with self.assertRaises(ValueError):
            self.store.save(dict(id=self.new,version=0,edit=edit))

    def test_http_auth_host_and_cross_origin_save(self):
        import base64,ipaddress,threading,urllib.request,urllib.error
        from http.server import ThreadingHTTPServer
        server=ThreadingHTTPServer(('127.0.0.1',0),expansion.ExpansionReviewHandler)
        server.store=self.store;server.template='<html>Synthetic test</html>'
        server.networks=[ipaddress.ip_network('127.0.0.0/8')]
        server.expected_auth='Basic '+base64.b64encode(b'test:synthetic-password').decode()
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url='http://127.0.0.1:'+str(server.server_port)
        try:
            with self.assertRaises(urllib.error.HTTPError) as unauth:
                urllib.request.urlopen(url+'/api/list')
            self.assertEqual(unauth.exception.code,401)
            request=urllib.request.Request(url+'/',headers={'Authorization':server.expected_auth,'Host':'example.org'})
            with self.assertRaises(urllib.error.HTTPError) as wronghost: urllib.request.urlopen(request)
            self.assertEqual(wronghost.exception.code,403)
            request=urllib.request.Request(url+'/api/save',data=b'{}',headers={
                'Authorization':server.expected_auth,'Origin':'http://example.org',
                'X-Review-Request':'1','Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as cross: urllib.request.urlopen(request)
            self.assertEqual(cross.exception.code,403)
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__': unittest.main()
