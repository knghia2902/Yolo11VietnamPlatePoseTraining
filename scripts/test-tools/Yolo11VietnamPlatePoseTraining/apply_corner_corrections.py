"""Apply visually checked draft corners with revision checks and private audit.

This is not an automatic label approver. Frozen labels and reviewed corners
are excluded. Only explicit per-image correction plans may be applied.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from prepare_pose_dataset import valid_points


def apply_corner_corrections(workspace, plan_path, audit_path, data_root):
    workspace, plan_path, audit_path, data_root = map(lambda p: Path(p).resolve(),
        (workspace, plan_path, audit_path, data_root))
    repo = Path(__file__).resolve().parents[3]
    if any(p == repo or repo in p.parents for p in (workspace, plan_path, audit_path, data_root)):
        raise ValueError('Customer plans, audit and data must remain outside Git')
    payload = plan_path.read_bytes()
    plan = json.loads(payload)
    plan_hash = hashlib.sha256(payload).hexdigest()
    if plan.get('method') != 'visual-source-corner-correction' or not plan.get('reviewer'):
        raise ValueError('Explicit visual source review is required')
    corrections = plan.get('corrections', [])
    if not corrections or len({c['id'] for c in corrections}) != len(corrections):
        raise ValueError('Empty or duplicate correction targets')
    db = sqlite3.connect(workspace/'curation.sqlite', timeout=60)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA busy_timeout=60000')
    before, updates, already = [], [], []
    try:
        db.execute('BEGIN IMMEDIATE')
        for correction in corrections:
            image_id = correction['id']
            if not re.fullmatch('[a-f0-9]{64}', image_id):
                raise ValueError('Invalid image ID')
            points = correction['points']
            valid_points(points)
            if correction.get('single_plate_source_checked') is not True:
                raise ValueError('Inspect the original image before applying corners')
            row = db.execute('SELECT * FROM images WHERE id=?', (image_id,)).fetchone()
            if row is None:
                raise ValueError('Unknown image')
            proposal, edit = json.loads(row['proposal']), json.loads(row['edit'])
            if row['kind'] not in ('acs', 'roboflow') or row['stage'] != 'ready':
                raise ValueError('Frozen, multi-object or unavailable images cannot be changed')
            if edit.get('corners_reviewed') or edit.get('exclude') or proposal.get('source_deleted'):
                raise ValueError('Reviewed, excluded or deleted image is protected')
            if len(proposal.get('source_detection_boxes', [])) > 1:
                raise ValueError('Source has multiple labeled objects')
            if edit.get('points') == points and proposal.get('corner_correction', {}).get('plan_sha256') == plan_hash:
                already.append(image_id)
                continue
            if type(correction.get('version')) is not int or row['version'] != correction['version']:
                raise RuntimeError('Revision changed; inspect the latest edit instead of overwriting it')
            if edit.get('points', proposal.get('points')) != correction['original_points']:
                raise RuntimeError('Draft geometry changed')
            source = Path(row['path']).resolve()
            if data_root not in source.parents or not source.is_file():
                raise ValueError('Source must be an existing image inside the authorized data root')
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != row['sha'] or actual != correction['sha256']:
                raise ValueError('Original source fingerprint changed')
            new_edit = dict(edit, points=points)
            old_ocr = proposal.get('ocr', {})
            evidence = old_ocr.get('candidates', []) + old_ocr.get('comparison_candidates', [])
            proposal['ocr'] = dict(schema=2, layout_aware=True, crop_pipeline=0,
                plate='', confidence=0, valid_format=False, candidates=[],
                comparison_candidates=evidence, previous_ocr_summary=old_ocr,
                crop_check=dict(status='review', reason='Bốn góc đã sửa; chờ OCR đọc lại'))
            proposal['corner_correction'] = dict(method=plan['method'], reviewer=plan['reviewer'],
                plan_sha256=plan_hash, previous_points=correction['original_points'],
                at_ns=time.time_ns(), corners_approved=False)
            before.append(dict(row))
            updates.append((image_id, row['version'], row['edit'], json.dumps(new_edit), json.dumps(proposal)))
        if updates:
            # Durable write-ahead snapshot contains both edit AND OCR/proposal.
            # If the transaction fails, this file is retained for investigation.
            with audit_path.open('x', encoding='utf-8') as audit:
                json.dump(dict(plan_sha256=plan_hash, before=before,
                    note='Write-ahead snapshot; consult the adjacent receipt for commit outcome'), audit)
                audit.flush()
                os.fsync(audit.fileno())
            for image_id, version, old_edit, new_edit, proposal in updates:
                db.execute('INSERT INTO history VALUES(?,?,?,?)', (image_id, version, old_edit, time.time_ns()))
                changed = db.execute('UPDATE images SET edit=?,proposal=?,version=version+1,ocr_done=0 '
                    'WHERE id=? AND version=?', (new_edit, proposal, image_id, version))
                if changed.rowcount != 1:
                    raise RuntimeError('Concurrent revision conflict')
                db.execute('INSERT INTO crop_ocr_priority(id,priority) VALUES(?,-3) '
                    'ON CONFLICT(id) DO UPDATE SET priority=-3', (image_id,))
        db.commit()
        receipt = dict(committed=True, corrected=len(updates), already_applied=len(already),
            plan_sha256=plan_hash, frozen_labels_changed=0, training_approvals_added=0)
        if updates:
            audit_path.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        return receipt
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--data-root', required=True)
    args = parser.parse_args()
    print(json.dumps(apply_corner_corrections(args.workspace, args.plan, args.audit, args.data_root)))
