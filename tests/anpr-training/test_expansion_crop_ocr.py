"""Synthetic crop safety cases. No customer images or plate identities."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/test-tools/Yolo11VietnamPlatePoseTraining'))
import expansion_crop_ocr as crop
import expansion_triage as triage
import curate_acs_training_pool as expansion

QUAD = [[.1, .1], [.9, .1], [.9, .8], [.1, .8]]
OTHER = [[.2, .2], [.8, .2], [.8, .7], [.2, .7]]


def record(kind='acs', edit=None, **proposal):
    return dict(id='a'*64, kind=kind, stage='ready', path='synthetic.jpg', sha=None,
                edit=json.dumps(edit or {}), proposal=json.dumps(dict(points=OTHER, **proposal)))


class CropQualityTests(unittest.TestCase):
    def test_pose_precedes_contour_and_stays_inside_source_box(self):
        row = record(corner_method='contour-in-source-bbox', source_detection_boxes=[[.5, .5, .9, .9]],
                     detections=[dict(points=QUAD, confidence=.95)])
        proposals = crop.crop_proposals(row)
        self.assertEqual(proposals[0]['source'], 'yolo-four-points')
        self.assertEqual(proposals[0]['points'], QUAD)
        outside = record(source_detection_boxes=[[.05, .05, .05, .05]],
                         detections=[dict(points=QUAD, confidence=.99)])
        self.assertFalse(any(p['source'] == 'yolo-four-points' for p in crop.crop_proposals(outside)))

    def test_reviewed_corners_and_frozen_labels_override_model(self):
        for kind, edit in [('validation', {}), ('baseline', {}),
                           ('acs', dict(corners_reviewed=True, points=QUAD))]:
            proposals = crop.crop_proposals(record(kind, edit, detections=[dict(points=QUAD, confidence=.99)]))
            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0]['source'], 'reviewed-corners')
            self.assertEqual(proposals[0]['points'], edit.get('points', OTHER))

    def test_saved_manual_correction_is_used_without_creating_approval(self):
        r = record(edit=dict(points=QUAD), detections=[dict(points=OTHER, confidence=.99)])
        proposals = crop.crop_proposals(r)
        self.assertEqual(proposals[0]['source'], 'saved-corners')
        self.assertEqual(proposals[0]['points'], QUAD)
        self.assertNotIn('corners_reviewed', json.loads(r['edit']))

    def test_tiny_and_crossed_corners_rejected_before_upscale(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        for points in ([[.1,.1],[.2,.1],[.2,.2],[.1,.2]],
                       [[.1,.1],[.9,.8],[.9,.1],[.1,.8]]):
            with self.assertRaises(ValueError): crop.crop_warp(image, points)

    def test_no_text_does_not_call_recognizer_or_delete_image(self):
        calls = []
        image = np.random.default_rng(1).integers(0, 255, (200, 480, 3), dtype=np.uint8)
        def engine(image, **options):
            calls.append(options)
            self.assertFalse(options['use_rec'])
            return None, None
        row = record(detections=[dict(points=QUAD, confidence=.99)])
        with patch.object(crop, 'crop_load', return_value=image), patch.object(crop, 'crop_warp', return_value=image):
            result = crop.crop_recognize('unused', row, engine)
        self.assertTrue(calls)
        self.assertEqual(result['crop_check']['status'], 'rejected')
        self.assertEqual(result['candidates'], [])
        self.assertEqual(json.loads(row['edit']), {})

    def test_elongated_text_lines_pass_and_blank_crop_rejected(self):
        image = np.random.default_rng(2).integers(0, 255, (200, 480, 3), dtype=np.uint8)
        def detector(image, **options):
            self.assertFalse(options['use_rec'])
            return [[[30,30],[400,30],[400,85],[30,85]]], None
        self.assertEqual(crop.crop_check(detector, image)['status'], 'pass')
        self.assertEqual(crop.crop_check(detector, np.full_like(image, 250))['status'], 'rejected')

    def test_quality_failure_blocks_old_high_confidence_ocr_but_keeps_match(self):
        read = dict(schema=2, layout_aware=True, plate='88A12345', confidence=.99,
                    candidates=[dict(plate='88A12345', confidence=.99, method='det-lines'),
                                dict(plate='88A12345', confidence=.99, method='whole-crop')],
                    crop_check=dict(status='rejected', reason='No plausible text'))
        r = record(ocr=read)
        self.assertEqual(triage.triage_record(r, triage.triage_validation_index([]))['category'], 'ocr_uncertain')
        reference = record('validation', ocr=read)
        r = record(ocr=dict(schema=2, layout_aware=True, plate='', confidence=0,
                           comparison_candidates=read['candidates']))
        self.assertEqual(triage.triage_record(r, triage.triage_validation_index([reference]))['category'], 'suspected_overlap')

    def test_history_is_leakage_evidence_not_a_vote_on_a_new_crop(self):
        read = dict(schema=2, layout_aware=True, plate='29B98765', confidence=.99,
                    candidates=[dict(plate='29B98765', confidence=.99, method='det-lines'),
                                dict(plate='29B98765', confidence=.99, method='whole-crop')],
                    comparison_candidates=[dict(plate='88A12345', confidence=.99)],
                    crop_check=dict(status='pass'))
        r = record(ocr=read)
        self.assertEqual(triage.triage_record(r, triage.triage_validation_index([]))['category'], 'ocr_distinct')
        ref = record('validation', dict(plate='88A12345', identity_reviewed=True))
        self.assertEqual(triage.triage_record(r, triage.triage_validation_index([ref]))['category'], 'suspected_overlap')

    def test_worker_resumes_preserves_human_review_and_prioritizes_uncertain_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            with expansion.expansion_db(folder) as db:
                expansion.expansion_meta(db, 'processing_complete', True)
                for i, kind in enumerate(('roboflow', 'validation', 'acs')):
                    r = record(kind, dict(reviewer='Synthetic reviewer'),
                        ocr=dict(schema=2, layout_aware=True, plate='', confidence=0))
                    db.execute('INSERT INTO images(id,kind,path,stage,proposal,edit) VALUES(?,?,?,?,?,?)',
                        (str(i)*64, kind, str(i)+r['path'], r['stage'], r['proposal'], r['edit']))
                db.commit()
            order = []
            def recognize(folder, row, engine):
                order.append(row['kind'])
                return dict(schema=2, layout_aware=True, crop_pipeline=1, plate='', confidence=0,
                            crop_check=dict(status='rejected'))
            with patch.object(crop, 'crop_recognize', side_effect=recognize):
                self.assertEqual(crop.crop_worker(folder, engine=object(), ocr_threads=4, opencv_threads=1), 3)
                self.assertEqual(crop.crop_worker(folder, engine=object()), 0)
            self.assertEqual(order[0], 'validation')
            with expansion.expansion_db(folder) as db:
                for r in db.execute('SELECT edit,version FROM images'):
                    self.assertEqual(json.loads(r['edit']), dict(reviewer='Synthetic reviewer'))
                    self.assertEqual(r['version'], 0)

    def test_thread_configuration_only_changes_runtime_resources(self):
        import types
        engine_factory = Mock(return_value=object())
        with tempfile.TemporaryDirectory() as folder:
            with expansion.expansion_db(folder) as db:
                expansion.expansion_meta(db, 'processing_complete', True)
            with patch.dict(sys.modules, {'rapidocr_onnxruntime':types.SimpleNamespace(RapidOCR=engine_factory)}), \
                    patch('cv2.setNumThreads') as set_threads, patch('cv2.getNumThreads', return_value=1):
                self.assertEqual(crop.crop_worker(folder, ocr_threads=4, opencv_threads=1), 0)
            set_threads.assert_called_once_with(1)
            engine_factory.assert_called_once_with(intra_op_num_threads=4, inter_op_num_threads=1, text_score=0.0)
        for kwargs in ({'ocr_threads':0}, {'opencv_threads':0}, {'ocr_threads':True}):
            with self.assertRaises(ValueError):
                crop.crop_worker('unused', engine=object(), **kwargs)

    def test_excluded_and_physically_deleted_sources_are_not_reopened(self):
        with tempfile.TemporaryDirectory() as folder:
            with expansion.expansion_db(folder) as db:
                expansion.expansion_meta(db, 'processing_complete', True)
                for i, p, e in [('1', {}, {'exclude':True}), ('2', {'source_deleted':{'reason':'front'}}, {})]:
                    db.execute('INSERT INTO images(id,kind,path,stage,proposal,edit) VALUES(?,?,?,?,?,?)',
                        (i*64, 'acs', i+'-deleted.jpg', 'ready', json.dumps(p), json.dumps(e)))
                db.commit()
            with patch.object(crop, 'crop_recognize') as recognize:
                self.assertEqual(crop.crop_worker(folder, engine=object()), 0)
                recognize.assert_not_called()

    def test_listing_checks_image_files_for_returned_page_not_entire_pool(self):
        with tempfile.TemporaryDirectory() as folder:
            with expansion.expansion_db(folder) as db:
                expansion.expansion_meta(db, 'manifest_sha256', 'synthetic')
                for i in range(90):
                    r=record()
                    db.execute('INSERT INTO images(id,kind,path,stage,proposal,edit) VALUES(?,?,?,?,?,?)',
                        (format(i,'064x'), 'acs', str(i)+'.jpg', 'ready', r['proposal'], r['edit']))
                db.commit()
            store=expansion.ExpansionReviewStore(folder)
            with patch.object(Path, 'exists', return_value=False) as exists:
                result=store.listing('acs', 'all')
            self.assertEqual(result['total'], 90)
            self.assertEqual(len(result['records']), 40)
            self.assertLessEqual(exists.call_count, 2*len(result['records']))


if __name__ == '__main__': unittest.main()
