import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS=Path(__file__).resolve().parents[2]/'scripts/test-tools/Yolo11VietnamPlatePoseTraining'
sys.path.insert(0,str(TOOLS))
import expansion_triage as triage
import expansion_ocr_all as ocr_all
import curate_acs_training_pool as expansion

POINTS=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]


def row(image_id='b'*64,kind='acs',plate='29B98765',confidence=.98,edit=None,stage='ready',**ocr):
    return dict(id=image_id,kind=kind,stage=stage,version=0,
                proposal=json.dumps(dict(points=POINTS,ocr=dict(schema=2,layout_aware=True,plate=plate,
                    confidence=confidence,candidates=[dict(plate=plate,confidence=confidence,method='det-lines',region=0),
                        dict(plate=plate,confidence=confidence,method='whole-crop',region=0)],**ocr))),edit=json.dumps(edit or {}))


class ExpansionTriageTests(unittest.TestCase):
    def index(self,plate='88A12345',human=False):
        e=dict(identity_reviewed=True,plate=plate) if human else {}
        return triage.triage_validation_index([row('a'*64,'validation',plate=plate,edit=e)])

    def test_exact_ocr_match_is_suspect_never_automatic_approval(self):
        result=triage.triage_record(row(plate='88A12345'),self.index())
        self.assertEqual(result['category'],'suspected_overlap')
        self.assertTrue(result['has_exact_match']);self.assertFalse(result['training_ready'])

    def test_human_confirmed_both_identities_block(self):
        r=row(plate='88A12345',edit=dict(identity_reviewed=True,plate='88A12345'))
        self.assertEqual(triage.triage_record(r,self.index(human=True))['category'],'confirmed_overlap')

    def test_one_character_and_missing_character_are_quarantined(self):
        for plate in ('88A12346','88A1234'):
            r=triage.triage_record(row(plate=plate),self.index())
            self.assertEqual(r['category'],'suspected_overlap')
            self.assertEqual(r['matches'][0]['method'],'one-character')

    def test_normalization_does_not_guess_letters_as_numbers(self):
        self.assertEqual(triage.triage_plate('88-A 12O.45'),'88A12O45')
        self.assertFalse(triage.triage_valid('88A12O45'))

    def test_low_confidence_and_invalid_read_go_to_uncertain(self):
        for r in (row(confidence=.89),row(plate='BADREAD',confidence=.99),row(confidence=float('nan'))):
            self.assertEqual(triage.triage_record(r,self.index())['category'],'ocr_uncertain')

    def test_disagreeing_ocr_methods_on_same_region_are_uncertain(self):
        r=row();p=json.loads(r['proposal'])
        p['ocr']['candidates']=[dict(plate='29B98765',confidence=.98,region=0),
                              dict(plate='29B98766',confidence=.97,region=0)]
        r['proposal']=json.dumps(p)
        self.assertEqual(triage.triage_record(r,self.index())['category'],'ocr_uncertain')

    def test_full_frame_and_multiple_regions_never_clear_automatically(self):
        for r in (row(full_frame=True),row(stage='multiple'),row(regions_truncated=True)):
            self.assertEqual(triage.triage_record(r,self.index())['category'],'ocr_uncertain')

    def test_clear_ocr_is_only_provisional_and_requires_corners(self):
        result=triage.triage_record(row(),self.index())
        self.assertEqual(result['category'],'ocr_distinct');self.assertFalse(result['training_ready'])
        self.assertEqual(triage.triage_record(row(stage='no_detection'),self.index())['category'],'corners_pending')

    def test_old_ocr_awaits_complete_pass(self):
        r=row();p=json.loads(r['proposal']);p['ocr'].pop('schema');r['proposal']=json.dumps(p)
        self.assertEqual(triage.triage_record(r,self.index())['category'],'ocr_pending')

    def test_one_high_confidence_method_is_not_enough_to_clear(self):
        r=row();p=json.loads(r['proposal']);p['ocr']['candidates']=p['ocr']['candidates'][:1]
        r['proposal']=json.dumps(p)
        self.assertEqual(triage.triage_record(r,self.index())['category'],'ocr_uncertain')

    def test_near_validation_image_is_suspect_even_if_ocr_differs(self):
        reference=row('a'*64,'validation',plate='88A12345');reference['dhash']='0000000000000000'
        index=triage.triage_validation_index([reference])
        candidate=row();candidate['dhash']='000000000000000f'
        result=triage.triage_record(candidate,index)
        self.assertEqual(result['category'],'suspected_overlap')
        self.assertEqual(result['matches'][0]['method'],'near-image')

    def test_near_id_still_flags_reference_with_unreadable_plate(self):
        reference=row('a'*64,'validation',plate='')
        candidate=row();p=json.loads(candidate['proposal']);p['near']=['a'*64];candidate['proposal']=json.dumps(p)
        result=triage.triage_record(candidate,triage.triage_validation_index([reference]))
        self.assertEqual(result['category'],'suspected_overlap')

    def test_human_difference_resolution_is_bound_to_reference_set(self):
        index=self.index(plate='29B98766',human=True)
        r=row(edit=dict(plate='29B98765',identity_reviewed=True))
        first=triage.triage_record(r,index)
        e=json.loads(r['edit']);e.update(validation_comparison='different',validation_comparison_digest=first['match_digest'])
        r['edit']=json.dumps(e)
        self.assertEqual(triage.triage_record(r,index)['category'],'ocr_distinct')
        extra=row('c'*64,'validation',plate='29B98766',edit=dict(identity_reviewed=True,plate='29B98766'))
        changed=triage.triage_validation_index([row('a'*64,'validation',plate='29B98766',edit=dict(identity_reviewed=True,plate='29B98766')),extra])
        self.assertEqual(triage.triage_record(r,changed)['category'],'suspected_overlap')

    def test_different_override_cannot_clear_exact_plate(self):
        index=self.index();r=row(plate='88A12345',edit=dict(plate='88A12345',identity_reviewed=True))
        result=triage.triage_record(r,index)
        e=json.loads(r['edit']);e.update(validation_comparison='different',validation_comparison_digest=result['match_digest'])
        r['edit']=json.dumps(e)
        self.assertEqual(triage.triage_record(r,index)['category'],'suspected_overlap')


class ExpansionCompleteOCRTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'previews').mkdir();(self.root/'crops').mkdir()
        with expansion.expansion_db(self.root) as db:
            expansion.expansion_meta(db,'manifest_sha256','synthetic')
            expansion.expansion_meta(db,'processing_complete',True)

    def tearDown(self):self.temp.cleanup()

    def add(self,r):
        with expansion.expansion_db(self.root) as db:
            p=json.loads(r['proposal']);p.pop('ocr',None)
            db.execute('INSERT INTO images(id,kind,path,stage,proposal,edit) VALUES(?,?,?,?,?,?)',
                       (r['id'],r['kind'],str(self.root/r['id']),r['stage'],json.dumps(p),r['edit']));db.commit()

    class Engine:
        def __init__(self):self.calls=0
        def __call__(self,image,**options):
            self.calls+=1
            if options['use_det']:
                return [[[[0,0],[10,0],[10,10],[0,10]],'29B98765',.98]],None
            return [['29B98765',.98]],None

    def test_two_methods_and_resume_do_not_approve_or_reprocess(self):
        self.add(row());engine=self.Engine()
        with patch.object(ocr_all,'expansion_ocr_regions',return_value=([('synthetic',False)],False,False)):
            self.assertEqual(ocr_all.expansion_ocr_all(self.root,engine=engine),1)
            self.assertEqual(ocr_all.expansion_ocr_all(self.root,engine=engine),0)
        self.assertEqual(engine.calls,2)
        with expansion.expansion_db(self.root) as db:
            saved=db.execute('SELECT proposal,edit FROM images').fetchone()
            self.assertEqual(json.loads(saved['edit']),{})
            self.assertEqual(json.loads(saved['proposal'])['ocr']['schema'],2)
            self.assertTrue(expansion.expansion_meta(db,'ocr_all_complete'))

    def test_motorcycle_second_pass_reads_two_lines_instead_of_whole_square(self):
        import numpy as np
        class TwoLineEngine:
            def __call__(self,image,**options):
                if options['use_det']:
                    return [[[[0,10],[400,10],[400,110],[0,110]],'88A1',.99],
                            [[[0,260],[400,260],[400,360],[0,360]],'12345',.99]],None
                return ([['88A1',.99]] if image.mean()<128 else [['12345',.99]]),None
        image=np.zeros((400,480,3),dtype='uint8');image[200:]=255
        result=ocr_all.expansion_recognize_all(TwoLineEngine(),[(image,False)])
        self.assertEqual(result['plate'],'88A112345')
        self.assertTrue(result['layout_aware'])
        self.assertEqual({c['method'] for c in result['candidates']},{'det-lines','split-lines'})
        r=row(plate='88A112345');p=json.loads(r['proposal']);p['ocr']=result;r['proposal']=json.dumps(p)
        self.assertEqual(triage.triage_record(r,triage.triage_validation_index([]))['category'],'ocr_distinct')

    def test_no_detection_gets_full_frame_attempt_and_remains_uncertain(self):
        self.add(row(stage='no_detection'));engine=self.Engine()
        with patch.object(ocr_all,'expansion_ocr_regions',return_value=([('synthetic',True)],True,False)):
            self.assertEqual(ocr_all.expansion_ocr_all(self.root,engine=engine),1)
        with expansion.expansion_db(self.root) as db:
            saved=db.execute('SELECT * FROM images').fetchone()
            index=triage.triage_validation_index([])
            self.assertEqual(triage.triage_record(saved,index)['category'],'ocr_uncertain')
        self.assertEqual(engine.calls,1)

    def test_duplicate_never_runs_redundant_ocr(self):
        self.add(row(stage='duplicate'));engine=self.Engine()
        self.assertEqual(ocr_all.expansion_ocr_all(self.root,engine=engine),0)
        self.assertEqual(engine.calls,0)

    def test_filters_and_exact_match_override_guard(self):
        self.add(row('a'*64,'validation',plate='88A12345'))
        self.add(row(plate='88A12345'))
        with expansion.expansion_db(self.root) as db:
            for image_id,plate in [('a'*64,'88A12345'),('b'*64,'88A12345')]:
                p=dict(points=POINTS,ocr=dict(schema=2,plate=plate,confidence=.99))
                db.execute('UPDATE images SET proposal=? WHERE id=?',(json.dumps(p),image_id))
            db.commit()
        store=expansion.ExpansionReviewStore(self.root)
        self.assertEqual(store.listing('acs','matched')['total'],1)
        edit=dict(plate='88A12345',reviewer='Synthetic',identity_reviewed=True,
                  near_checked=True,validation_comparison='different')
        with self.assertRaises(ValueError):store.save(dict(id='b'*64,version=0,edit=edit))

    def test_manual_same_vehicle_is_persisted_as_blocked(self):
        self.add(row('a'*64,'validation',plate='29B98766'))
        self.add(row())
        with expansion.expansion_db(self.root) as db:
            for image_id,plate in [('a'*64,'29B98766'),('b'*64,'29B98765')]:
                db.execute('UPDATE images SET proposal=? WHERE id=?',
                    (json.dumps(dict(points=POINTS,ocr=dict(schema=2,plate=plate,confidence=.99))),image_id))
            db.commit()
        store=expansion.ExpansionReviewStore(self.root)
        edit=dict(plate='29B98765',reviewer='Synthetic',identity_reviewed=True,
                  near_checked=True,validation_comparison='same')
        saved=store.save(dict(id='b'*64,version=0,edit=edit))
        self.assertEqual(saved['status'],'validation_overlap')
        self.assertEqual(saved['triage']['category'],'confirmed_overlap')


if __name__=='__main__':unittest.main()
