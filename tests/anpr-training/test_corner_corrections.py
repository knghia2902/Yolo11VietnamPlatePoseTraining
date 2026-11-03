"""Synthetic transactional corner correction and review protection checks."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/test-tools/Yolo11VietnamPlatePoseTraining'))
from apply_corner_corrections import apply_corner_corrections
from curate_acs_training_pool import expansion_db


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root/'workspace'; self.workspace.mkdir()
        self.source = self.root/'source'; self.source.mkdir()
        image = self.source/'synthetic.bin'; image.write_bytes(b'synthetic image fingerprint')
        sha = hashlib.sha256(image.read_bytes()).hexdigest()
        self.old = [[.1,.1],[.8,.1],[.8,.8],[.1,.8]]
        self.new = [[.2,.2],[.8,.2],[.8,.8],[.2,.8]]
        self.edit = dict(plate='88A12345', reviewer='Synthetic reviewer', identity_reviewed=True,
            corners_reviewed=False, near_checked=True, single_plate_reviewed=True)
        self.proposal = dict(points=self.old, ocr=dict(schema=2, crop_pipeline=1,
            plate='88A12345', confidence=.95, candidates=[dict(plate='88A12345', confidence=.95)]))
        with expansion_db(self.workspace) as db:
            db.execute('CREATE TABLE crop_ocr_priority(id TEXT PRIMARY KEY,priority INTEGER NOT NULL)')
            db.execute('INSERT INTO images(id,kind,path,stage,sha,proposal,edit) VALUES(?,?,?,?,?,?,?)',
                ('a'*64,'acs',str(image),'ready',sha,json.dumps(self.proposal),json.dumps(self.edit)))
        self.plan = dict(method='visual-source-corner-correction', reviewer='Synthetic visual checker',
            corrections=[dict(id='a'*64, version=0, sha256=sha, original_points=self.old,
                points=self.new, single_plate_source_checked=True)])
        self.planfile = self.root/'plan.json'
        self.audit = self.root/'audit.json'

    def apply(self):
        self.planfile.write_text(json.dumps(self.plan), encoding='utf-8')
        return apply_corner_corrections(self.workspace,self.planfile,self.audit,self.source)

    def row(self):
        with expansion_db(self.workspace) as db:
            return dict(db.execute('SELECT * FROM images').fetchone())

    def test_preserves_identity_and_approval_and_requeues_only_corrected_image(self):
        result = self.apply()
        self.assertEqual(result['corrected'],1)
        row = self.row(); edit=json.loads(row['edit']); proposal=json.loads(row['proposal'])
        self.assertEqual(edit,dict(self.edit,points=self.new))
        self.assertEqual(proposal['points'],self.old)
        self.assertEqual(proposal['ocr']['crop_pipeline'],0)
        self.assertEqual(proposal['ocr']['comparison_candidates'],self.proposal['ocr']['candidates'])
        self.assertEqual(proposal['ocr']['previous_ocr_summary'],self.proposal['ocr'])
        self.assertEqual(row['version'],1); self.assertEqual(row['ocr_done'],0)
        with expansion_db(self.workspace) as db:
            self.assertEqual(json.loads(db.execute('SELECT value FROM history').fetchone()[0]),self.edit)
            self.assertEqual(db.execute('SELECT priority FROM crop_ocr_priority').fetchone()[0],-3)
        self.assertEqual(json.loads(self.audit.read_text())['before'][0]['proposal'],json.dumps(self.proposal))
        self.assertEqual(self.apply()['already_applied'],1)
        self.assertEqual(self.row()['version'],1)

    def test_frozen_reviewed_excluded_and_revision_conflicts_are_rejected(self):
        for kind,edit,version in [('validation',self.edit,0),('baseline',self.edit,0),
                ('acs',dict(self.edit,corners_reviewed=True),0),('acs',dict(self.edit,exclude=True),0),
                ('acs',self.edit,1)]:
            with self.subTest(kind=kind,edit=edit,version=version):
                with expansion_db(self.workspace) as db:
                    db.execute('UPDATE images SET kind=?,edit=?,version=?',(kind,json.dumps(edit),version))
                before=self.row()
                with self.assertRaises((ValueError,RuntimeError)): self.apply()
                self.assertEqual(self.row(),before)
                self.assertFalse(self.audit.exists())

    def test_source_changes_and_invalid_or_unchecked_corners_are_rejected(self):
        for change in ('source','geometry','not-checked'):
            with self.subTest(change=change):
                if change=='source': self.plan['corrections'][0]['sha256']='0'*64
                if change=='geometry':
                    self.plan['corrections'][0]['sha256']=hashlib.sha256(b'synthetic image fingerprint').hexdigest()
                    self.plan['corrections'][0]['points']=[[0,0]]*4
                if change=='not-checked':
                    self.plan['corrections'][0]['points']=self.new
                    self.plan['corrections'][0]['single_plate_source_checked']=False
                before=self.row()
                with self.assertRaises(ValueError): self.apply()
                self.assertEqual(self.row(),before)

    def test_existing_audit_prevents_unlogged_write(self):
        self.audit.write_text('existing audit')
        before=self.row()
        with self.assertRaises(FileExistsError): self.apply()
        self.assertEqual(self.row(),before)


if __name__=='__main__': unittest.main()
