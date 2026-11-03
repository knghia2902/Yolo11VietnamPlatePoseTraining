"""Quality-gated OCR: reviewed corners, trained pose proposals, then fallbacks.

Rejection applies to a proposed crop, never to the original image or its label.
No network services, automatic identity approvals or source deletion are used.
"""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import time

from expansion_ocr_all import expansion_recognize_all
from expansion_triage import OCR_SCHEMA, triage_plate, triage_valid, triage_score
from prepare_pose_dataset import valid_points

CROP_PIPELINE = 1


def crop_load(workspace, row):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    source = Path(row['path'])
    # A deleted source must not silently acquire a new identity from an old
    # preview. A preview is allowed only for OCR full-scene fallback elsewhere.
    data = source.read_bytes()
    if row['sha'] and hashlib.sha256(data).hexdigest() != row['sha']:
        raise ValueError('Source image differs from its recorded fingerprint')
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
    return np.array(image)[:, :, ::-1].copy()


def crop_proposals(row):
    p, e = json.loads(row['proposal']), json.loads(row['edit'])
    points = e.get('points', p.get('points', []))
    frozen = row['kind'] in ('validation', 'baseline')
    if frozen or e.get('corners_reviewed'):
        return [dict(points=points, source='reviewed-corners', region=0)]
    if e.get('points') and e['points'] != p.get('points'):
        return [dict(points=e['points'], source='saved-corners', region=0)]
    boxes = p.get('source_detection_boxes', [])
    proposals = []
    detections = sorted(p.get('detections', []), key=lambda d: d.get('confidence', 0), reverse=True)
    for detection in detections[:8]:
        quad = detection.get('points', [])
        try:
            valid_points(quad)
        except (ValueError, TypeError):
            continue
        cx, cy = [sum(q[axis] for q in quad)/4 for axis in (0, 1)]
        matching = [i for i, (x, y, w, h) in enumerate(boxes)
                    if x-w/2 <= cx <= x+w/2 and y-h/2 <= cy <= y+h/2]
        if boxes and not matching:
            continue
        region = matching[0] if matching else len(proposals)
        proposals.append(dict(points=quad, source='yolo-four-points', region=region,
                              confidence=detection.get('confidence', 0)))
    # Contours/old ACS crops are fallbacks only. Never let an available contour
    # take precedence over an acceptable trained pose proposal.
    if p.get('points') and not any(item['points'] == p['points'] for item in proposals):
        proposals.append(dict(points=p['points'], source='legacy-crop' if row['stage'] == 'crop_only'
                              else p.get('corner_method', 'proposal-fallback'), region=0))
    for region, (x, y, w, h) in enumerate(boxes[:8]):
        proposals.append(dict(points=[[max(0, x-w*.52), max(0, y-h*.52)],
            [min(1, x+w*.52), max(0, y-h*.52)], [min(1, x+w*.52), min(1, y+h*.52)],
            [max(0, x-w*.52), min(1, y+h*.52)]], source='source-box-fallback', region=region))
    return proposals[:24]


def crop_warp(array, points):
    import cv2
    import numpy as np
    valid_points(points)
    height, width = array.shape[:2]
    pts = np.asarray(points, dtype=np.float32)*[width, height]
    w = max(np.linalg.norm(pts[1]-pts[0]), np.linalg.norm(pts[2]-pts[3]))
    h = max(np.linalg.norm(pts[3]-pts[0]), np.linalg.norm(pts[2]-pts[1]))
    if w < 40 or h < 12 or not .15 <= h/max(w, 1) <= 1.5:
        raise ValueError('crop-too-small-or-invalid-aspect')
    if not cv2.isContourConvex(pts.astype('float32')) or abs(cv2.contourArea(pts.astype('float32'))) < 300:
        raise ValueError('crop-invalid-quadrilateral')
    cw, ch = 480, round(480*h/w)
    transform = cv2.getPerspectiveTransform(pts.astype('float32'),
        np.asarray([[0, 0], [cw-1, 0], [cw-1, ch-1], [0, ch-1]], dtype=np.float32))
    return cv2.warpPerspective(array, transform, (cw, ch))


def crop_check(engine, image):
    """Detector-only preflight: don't run a recognizer on headlights/plastic.

    These are conservative evidence checks, not a calibrated plate classifier.
    Missing text is a crop rejection to review/retry, never proof of no plate.
    """
    import cv2
    import numpy as np
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    contrast = float(np.percentile(gray, 95)-np.percentile(gray, 5))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if contrast < 18:
        return dict(status='rejected', reason='crop-low-contrast', contrast=contrast, blur=blur)
    boxes, _ = engine(image, use_det=True, use_cls=False, use_rec=False)
    usable = []
    for box in boxes or []:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        bw, bh = max(xs)-min(xs), max(ys)-min(ys)
        if bw >= w*.12 and bh >= h*.035 and bw/max(bh, 1) >= 1.35:
            usable.append((min(xs), min(ys), max(xs), max(ys)))
    if not usable or len(usable) > 6:
        return dict(status='rejected', reason='crop-no-plausible-text-lines',
                    text_lines=len(usable), contrast=contrast, blur=blur)
    if blur < 5:
        return dict(status='review', reason='crop-blurred', text_lines=len(usable), contrast=contrast, blur=blur)
    return dict(status='pass', text_lines=len(usable), contrast=contrast, blur=blur)


def crop_variants(image):
    import cv2
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    yield cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    yield cv2.addWeighted(image, 1.5, cv2.GaussianBlur(image, (0, 0), 1), -.5, 0)


def crop_read_consistent(result):
    best = result.get('plate', '')
    if not triage_valid(best) or triage_score(result.get('confidence')) < .9:
        return False
    methods = {c.get('method') for c in result.get('candidates', [])
               if c.get('plate') == best and triage_score(c.get('confidence')) >= .9}
    alternatives = {c.get('plate') for c in result.get('candidates', [])
                    if triage_valid(c.get('plate', '')) and triage_score(c.get('confidence')) >= .7}
    return len(alternatives) == 1 and 'det-lines' in methods and bool(methods & {'split-lines', 'whole-crop'})


def crop_recognize(workspace, row, engine):
    previous = json.loads(row['proposal']).get('ocr', {})
    result = dict(schema=OCR_SCHEMA, layout_aware=True, crop_pipeline=CROP_PIPELINE,
                  plate='', confidence=0, valid_format=False, candidates=[], region=0,
                  engine='RapidOCR local; crop quality preflight', region_count=0)
    array = crop_load(workspace, row)
    proposals = crop_proposals(row)
    checks, accepted, regions, selected = [], [], set(), {}
    for proposal in proposals:
        region = proposal['region']
        if region in regions:
            continue
        try:
            image = crop_warp(array, proposal['points'])
            check = crop_check(engine, image)
            # A failed detector gets a contrast retry before the crop is held.
            if check['status'] != 'pass':
                enhanced = next(crop_variants(image))
                retry = crop_check(engine, enhanced)
                if retry['status'] == 'pass': image, check = enhanced, retry
            checks.append(dict(**check, source=proposal['source'], region=region))
            if check['status'] != 'pass': continue
            read = expansion_recognize_all(engine, [(image, False)])
            candidates = read['candidates']
            if not crop_read_consistent(read):
                for variant, enhanced in enumerate(crop_variants(image)):
                    more = expansion_recognize_all(engine, [(enhanced, False)])
                    for candidate in more['candidates']:
                        candidate['variant'] = variant
                    candidates.extend(more['candidates'])
            for candidate in candidates:
                candidate['region'] = region
                candidate['crop_source'] = proposal['source']
            accepted.extend(candidates);regions.add(region)
            selected[region] = (image, proposal['source'])
        except ValueError as exc:
            checks.append(dict(status='rejected', reason=str(exc), source=proposal['source'], region=region))
        # For a single plate, don't OCR weaker fallbacks after a plausible pose
        # crop. Multiple source boxes/pose detections retain independent regions.
        if regions and row['stage'] != 'multiple': break
    accepted.sort(key=lambda c:(triage_valid(c['plate']), c['confidence']), reverse=True)
    if accepted:
        best = accepted[0]
        result.update(plate=best['plate'], confidence=best['confidence'], region=best['region'],
                      valid_format=triage_valid(best['plate']), candidates=accepted, region_count=len(regions))
        from PIL import Image
        from curate_acs_training_pool import expansion_jpeg
        image, source = selected[best['region']]
        target = Path(workspace)/'ocr-crops'/(row['id']+'.jpg')
        target.parent.mkdir(exist_ok=True)
        expansion_jpeg(Image.fromarray(image[:, :, ::-1]), target, 95)
        result['crop_source'] = source
    p = json.loads(row['proposal'])
    result['regions_truncated'] = len(p.get('detections', [])) > 8 or len(p.get('source_detection_boxes', [])) > 8
    result['crop_check'] = dict(status='pass' if accepted else 'rejected', checks=checks,
        reason=None if accepted else 'Crop không có đủ dấu hiệu hàng chữ biển số hoặc hình học không đạt; cần kiểm tra ảnh gốc')
    # Retain historical evidence for conservative leakage checks, but not as
    # support for accepting a new crop. Old hallucinations cannot approve it.
    result['comparison_candidates'] = previous.get('candidates', []) + previous.get('comparison_candidates', [])
    # Bound repeated runs and keep only unique evidence values.
    result['comparison_candidates'] = list({(c.get('plate'), c.get('region', 0)): c
                                           for c in result['comparison_candidates']}.values())
    result['previous_ocr_summary'] = {key: previous.get(key) for key in ('plate', 'confidence', 'crop_check')}
    return result


def crop_worker(workspace, follow=False, limit=0, engine=None, ocr_threads=2, opencv_threads=None):
    from curate_acs_training_pool import expansion_db, expansion_meta
    from expansion_triage import triage_record, triage_validation_index
    if type(ocr_threads) is not int or ocr_threads < 1:
        raise ValueError('ocr_threads must be a positive integer')
    if opencv_threads is not None and (type(opencv_threads) is not int or opencv_threads < 1):
        raise ValueError('opencv_threads must be a positive integer')
    if engine is None:
        import cv2
        from rapidocr_onnxruntime import RapidOCR
        if opencv_threads is not None:
            cv2.setNumThreads(opencv_threads)
        engine = RapidOCR(intra_op_num_threads=ocr_threads, inter_op_num_threads=1, text_score=0.0)
        print(json.dumps(dict(ocr_runtime=dict(intra_op_num_threads=ocr_threads,
            inter_op_num_threads=1, opencv_threads=cv2.getNumThreads(), text_score=0.0))), flush=True)
    count = 0
    with expansion_db(workspace) as db:
        rows = db.execute('SELECT * FROM images').fetchall()
        index = triage_validation_index(rows)
        # Store priority outside proposal, so an image worker cannot overwrite
        # it with an older proposal while inference is still running.
        db.execute('CREATE TABLE IF NOT EXISTS crop_ocr_priority(id TEXT PRIMARY KEY,priority INTEGER NOT NULL)')
        for row in rows:
            if triage_record(row, index)['category'] == 'ocr_uncertain':
                db.execute('INSERT OR IGNORE INTO crop_ocr_priority VALUES(?,?)',
                           (row['id'], 0 if row['kind'] == 'validation' else 1))
        db.commit();expansion_meta(db, 'ocr_all_complete', False)
        expansion_meta(db, 'crop_pipeline', CROP_PIPELINE)
    while not limit or count < limit:
        with expansion_db(workspace) as db:
            rows = db.execute("SELECT i.* FROM images i LEFT JOIN crop_ocr_priority q ON q.id=i.id "
                "WHERE i.stage NOT IN ('pending','duplicate','error') "
                "AND coalesce(json_extract(i.edit,'$.exclude'),0)=0 "
                "AND json_extract(i.proposal,'$.source_deleted') IS NULL "
                "AND coalesce(json_extract(i.proposal,'$.ocr.crop_pipeline'),0)<>? "
                "ORDER BY coalesce(q.priority,2),CASE i.kind WHEN 'validation' THEN 0 WHEN 'acs' THEN 1 WHEN 'roboflow' THEN 2 ELSE 3 END,i.rowid LIMIT 25",
                (CROP_PIPELINE,)).fetchall()
            if not rows:
                pending = db.execute("SELECT 1 FROM images WHERE stage='pending' LIMIT 1").fetchone()
                if expansion_meta(db, 'processing_complete') and not pending:
                    expansion_meta(db, 'ocr_all_complete', True);break
                if not follow: break
                time.sleep(3);continue
            for row in rows:
                if limit and count >= limit: break
                try:
                    result = crop_recognize(workspace, row, engine)
                except Exception as exc:
                    previous = json.loads(row['proposal']).get('ocr', {})
                    result = dict(schema=OCR_SCHEMA, layout_aware=True, crop_pipeline=CROP_PIPELINE,
                        plate='', confidence=0, candidates=[], error=type(exc).__name__,
                        comparison_candidates=previous.get('candidates', []) + previous.get('comparison_candidates', []),
                        crop_check=dict(status='rejected', reason='Không đọc được ảnh gốc; cần kiểm tra đường dẫn hoặc file đã bị xóa'))
                db.execute('BEGIN IMMEDIATE')
                latest = db.execute('SELECT proposal,edit FROM images WHERE id=?', (row['id'],)).fetchone()
                proposal = json.loads(latest['proposal'])
                # A reviewer may have changed points while this OCR was running.
                # Never publish geometry based on an older edit; retry next pass.
                if latest['edit'] != row['edit']:
                    db.rollback();continue
                proposal['ocr'] = result
                db.execute('UPDATE images SET proposal=?,ocr_done=1 WHERE id=?', (json.dumps(proposal), row['id']))
                db.commit();count += 1
                if count % 10 == 0: print(json.dumps(dict(crop_ocr_this_run=count)), flush=True)
            expansion_meta(db, 'crop_ocr_progress', dict(processed_this_run=count, at=time.time()))
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--ocr-threads', type=int, default=2)
    parser.add_argument('--opencv-threads', type=int)
    args = parser.parse_args()
    crop_worker(args.workspace, args.follow, args.limit, ocr_threads=args.ocr_threads,
                opencv_threads=args.opencv_threads)
