"""Synthetic auto-confirmation policies, audit and leakage guards."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/test-tools/Yolo11VietnamPlatePoseTraining'))
import expansion_auto_identity as auto
from expansion_triage import triage_validation_index, triage_record
from curate_acs_training_pool import expansion_db, expansion_meta

POINTS=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]


def row(i=1,plate='99A12345',edit=None,score=.98,**kwargs):
    o=dict(schema=2,layout_aware=True,crop_pipeline=1,crop_check=dict(status='pass'),
           plate=plate,confidence=score,region=0,region_count=1,candidates=[
               dict(plate=plate,confidence=score,method='det-lines',region=0),
               dict(plate=plate,confidence=score,method='split-lines',region=0)])
    o.update(kwargs)
    return dict(id=format(i,'064x'),kind='validation',stage='ready',path=str(i)+'.jpg',
                sha='a'*64,version=0,proposal=json.dumps(dict(points=POINTS,ocr=o)),edit=json.dumps(edit or {}))


class AutoIdentityTests(unittest.TestCase):
    def evaluate(self,r,others=()):return auto.auto_identity_evaluate(r,triage_validation_index([r,*others]))

    def test_two_distinct_clear_methods_and_crop_pass_are_required(self):
        self.assertTrue(self.evaluate(row())['eligible'])
        for r in (row(score=.94),row(crop_pipeline=0),row(crop_check=dict(status='rejected')),
                  row(error='ReadError'),row(full_frame=True),row(region_count=2),row(schema=1),
                  row(candidates=[dict(plate='99A12345',confidence=.99,method='det-lines',region=0)])):
            self.assertFalse(self.evaluate(r)['eligible'])

    def test_current_or_clear_historical_conflict_requires_review(self):
        for historical in (False,True):
            r=row();p=json.loads(r['proposal'])
            key='comparison_candidates' if historical else 'candidates'
            p['ocr'].setdefault(key,[]).append(dict(plate='88B76543',confidence=.99,method='whole-crop'))
            r['proposal']=json.dumps(p)
            self.assertEqual(self.evaluate(r)['reason'],'conflicting-plate-evidence')

    def test_fuzzy_validation_plate_is_manual_but_same_plate_repeat_is_allowed(self):
        self.assertEqual(self.evaluate(row(),[row(2,'99A12346')])['reason'],'ambiguous-validation-reference')
        self.assertTrue(self.evaluate(row(),[row(2)])['eligible'])

    def test_saved_human_identity_is_never_replaced(self):
        r=row(edit=dict(identity_reviewed=True,plate='88B76543',reviewer='Synthetic reviewer'))
        self.assertEqual(self.evaluate(r)['reason'],'human-confirmed-preserved')
        self.assertEqual(self.evaluate(row(edit=dict(plate='88B76543')))['reason'],'saved-manual-plate-differs')

    def database(self,folder,rows):
        with expansion_db(folder) as db:
            expansion_meta(db,'manifest_sha256','synthetic')
            for r in rows:
                db.execute('INSERT INTO images(id,kind,path,sha,stage,proposal,edit) VALUES(?,?,?,?,?,?,?)',
                    (r['id'],r['kind'],r['path'],r['sha'],r['stage'],r['proposal'],r['edit']))
            db.commit()

    def test_confirmation_is_audited_idempotent_and_preserves_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            r=row(edit=dict(points=POINTS))
            self.database(folder,[r])
            first=auto.auto_identity_pass(folder);second=auto.auto_identity_pass(folder)
            self.assertEqual(first['newly_confirmed'],1)
            self.assertEqual(second.get('newly_confirmed',0),0)
            with expansion_db(folder) as db:
                saved=db.execute('SELECT * FROM images').fetchone();e=json.loads(saved['edit'])
                self.assertTrue(e['identity_reviewed']);self.assertEqual(e['identity_method'],auto.METHOD)
                self.assertEqual(e['points'],POINTS);self.assertEqual(saved['path'],r['path'])
                self.assertEqual(saved['sha'],r['sha']);self.assertEqual(saved['version'],1)
                self.assertEqual(json.loads(saved['proposal'])['points'],POINTS)
                self.assertEqual(db.execute('SELECT count(*) FROM history').fetchone()[0],1)
                self.assertEqual(db.execute('SELECT count(*) FROM auto_identity_audit').fetchone()[0],1)
                coverage=triage_validation_index([saved])['coverage']
                self.assertEqual(coverage['auto_confirmed'],1);self.assertEqual(coverage.get('human_confirmed',0),0)

    def test_changed_ocr_revokes_auto_confirmation_and_creates_history(self):
        with tempfile.TemporaryDirectory() as folder:
            r=row();self.database(folder,[r]);auto.auto_identity_pass(folder)
            with expansion_db(folder) as db:
                p=json.loads(db.execute('SELECT proposal FROM images').fetchone()[0]);p['ocr']['confidence']=.5
                db.execute('UPDATE images SET proposal=?',(json.dumps(p),));db.commit()
            result=auto.auto_identity_pass(folder)
            self.assertEqual(result['revoked'],1)
            with expansion_db(folder) as db:
                saved=db.execute('SELECT * FROM images').fetchone()
                self.assertFalse(json.loads(saved['edit'])['identity_reviewed']);self.assertEqual(saved['version'],2)
                self.assertEqual(db.execute('SELECT count(*) FROM history').fetchone()[0],2)

    def test_ambiguous_validation_stays_in_manual_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            self.database(folder,[row(),row(2,'99A12346')]);result=auto.auto_identity_pass(folder)
            self.assertEqual(result['needs_review'],2)
            with expansion_db(folder) as db:
                rows=db.execute('SELECT * FROM images').fetchall();index=triage_validation_index(rows)
                for saved in rows:
                    self.assertEqual(saved['version'],0)
                    self.assertEqual(triage_record(saved,index)['category'],'ocr_uncertain')

    def test_auto_confirmed_validation_blocks_new_same_vehicle(self):
        with tempfile.TemporaryDirectory() as folder:
            self.database(folder,[row()]);auto.auto_identity_pass(folder)
            with expansion_db(folder) as db: reference=dict(db.execute('SELECT * FROM images').fetchone())
            candidate=row(2,edit=dict(plate='99A12345',identity_reviewed=True));candidate['kind']='acs'
            self.assertEqual(triage_record(candidate,triage_validation_index([reference]))['category'],'confirmed_overlap')

    def test_old_clear_reads_are_prioritized_but_not_confirmed_without_crop_check(self):
        with tempfile.TemporaryDirectory() as folder:
            self.database(folder,[row(crop_pipeline=0),row(2,'88B76543',score=.8,crop_pipeline=0)])
            result=auto.auto_identity_pass(folder)
            self.assertEqual(result.get('newly_confirmed',0),0)
            self.assertEqual(result['waiting_crop'],2)
            with expansion_db(folder) as db:
                priorities=dict(db.execute('SELECT id,priority FROM crop_ocr_priority'))
                self.assertEqual(priorities[format(1,'064x')],-1)
                self.assertEqual(priorities[format(2,'064x')],0)
                self.assertTrue(all(r[0]==0 for r in db.execute('SELECT version FROM images')))


if __name__=='__main__':unittest.main()
