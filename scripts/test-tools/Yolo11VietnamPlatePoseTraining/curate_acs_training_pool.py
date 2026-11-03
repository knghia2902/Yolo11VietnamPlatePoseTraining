"""Resumable private Vietnam-only expansion. Proposals never become ground truth.

Images, OCR, SQLite, credentials and model weights must stay outside Git.
Validation identities may be reviewed; its membership, images and corners stay frozen.
"""
import argparse
import base64
from collections import Counter
import hashlib
from http.server import ThreadingHTTPServer
import io
import ipaddress
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import parse_qs, urlsplit
import zipfile

from PIL import Image, ImageOps
from prepare_pose_dataset import fingerprints, valid_points, contour_proposal
from serve_pose_review import ReviewHandler
from expansion_triage import triage_validation_index, triage_record
from review_train_storage import train_images, sync_labels, check_split_overlap


class ExpansionConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


def expansion_db(workspace):
    db = sqlite3.connect(Path(workspace) / 'curation.sqlite', timeout=60, factory=ExpansionConnection)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA busy_timeout=60000')
    db.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
      CREATE TABLE IF NOT EXISTS images(id TEXT PRIMARY KEY,kind TEXT,path TEXT UNIQUE,
      stage TEXT DEFAULT 'pending',sha TEXT,pixel TEXT,dhash TEXT,proposal TEXT DEFAULT '{}',
      edit TEXT DEFAULT '{}',version INTEGER DEFAULT 0,ocr_done INTEGER DEFAULT 0,error TEXT);
      CREATE INDEX IF NOT EXISTS stage_idx ON images(stage);
      CREATE INDEX IF NOT EXISTS sha_idx ON images(sha);
      CREATE INDEX IF NOT EXISTS pixel_idx ON images(pixel);
      CREATE TABLE IF NOT EXISTS hash_parts(part INTEGER,value INTEGER,id TEXT,
      PRIMARY KEY(part,value,id));
      CREATE TABLE IF NOT EXISTS history(id TEXT,version INTEGER,value TEXT,at INTEGER,
      PRIMARY KEY(id,version));''')
    return db


def expansion_meta(db, key, value=None):
    if value is not None:
        db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, json.dumps(value)))
        db.commit()
        return value
    row = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else None


def expansion_plate(value):
    return re.sub('[^A-Z0-9]', '', str(value).upper())


def expansion_index_hash(db, image_id, dhash):
    number=int(dhash,16)
    db.executemany('INSERT OR IGNORE INTO hash_parts VALUES(?,?,?)',
        [(part,(number>>(13*part))&8191,image_id) for part in range(5)])


def expansion_near_hashes(db, image_id, dhash):
    number=int(dhash,16); candidates=set(); matches=[]
    # Five disjoint chunks guarantee a shared chunk for distance <= four.
    for part in range(5):
        candidates.update(r[0] for r in db.execute('SELECT id FROM hash_parts WHERE part=? AND value=?',
                                                (part,(number>>(13*part))&8191)))
    for candidate in candidates:
        if candidate==image_id: continue
        row=db.execute('SELECT dhash FROM images WHERE id=?',(candidate,)).fetchone()
        if row and row[0] and (int(row[0],16)^number).bit_count()<=4: matches.append(candidate)
    return sorted(matches)


def expansion_valid_plate(value):
    return bool(re.fullmatch(r'[0-9]{2}[A-Z]{1,2}[0-9]{4,6}', value))


def expansion_init(workspace, dataset, snapshot=None):
    workspace, dataset = Path(workspace).resolve(), Path(dataset).resolve()
    repo = Path(__file__).resolve().parents[3]
    if workspace == repo or repo in workspace.parents:
        raise ValueError('Private workspace must be outside the source checkout')
    workspace.mkdir(parents=True, exist_ok=True)
    for folder in ('previews', 'crops', 'imports', 'yolo-config'):
        (workspace / folder).mkdir(exist_ok=True)
    source = (dataset / 'manifest.json').read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    manifest = json.loads(source)
    reviewed = {}
    if snapshot:
        state = json.loads(Path(snapshot).read_text(encoding='utf-8'))['snapshot']
        reviewed = {r['id']: r for r in state['records']}
    with expansion_db(workspace) as db:
        old = expansion_meta(db, 'manifest_sha256')
        if old and old != digest:
            raise ValueError('Workspace is pinned to a different v3 manifest')
        expansion_meta(db, 'manifest_sha256', digest)
        expansion_meta(db, 'dataset', str(dataset))
        expansion_meta(db, 'target_useful_train_images', 50000)
        expansion_meta(db, 'foreign_plates_allowed', False)
        for r in manifest['records']:
            kind = 'validation' if r['split'] == 'val' else 'baseline'
            proposal = dict(points=r['points'], frozen_label=r['label'],
                            frozen_label_sha256=r['label_sha256'], plate='', near=[])
            old_review = reviewed.get(r['id'], {})
            # Only previously explicit human approvals are trusted, never OCR alone.
            edit = {}
            if old_review.get('approved') and expansion_valid_plate(old_review.get('plate', '')):
                expected=r.get('car_identity_sha256')
                actual=hashlib.sha256(old_review['plate'].encode()).hexdigest()
                if expected and expected!=actual:
                    raise ValueError('Approved plate snapshot differs from frozen v3 identity')
                edit = dict(plate=old_review['plate'], identity_reviewed=True,
                            reviewer=old_review.get('reviewer', 'previous-review'))
            db.execute('INSERT OR IGNORE INTO images(id,kind,path,stage,sha,pixel,dhash,proposal,edit) '
                       'VALUES(?,?,?,?,?,?,?,?,?)', (r['id'], kind, str(dataset/r['image']),
                       'ready', r['sha256'], r['pixel_sha256'], format(r['dhash'],'016x') if isinstance(r['dhash'],int) else r['dhash'],
                       json.dumps(proposal), json.dumps(edit)))
        db.commit()
    return dict(manifest_sha256=digest, counts=dict(Counter(r['split'] for r in manifest['records'])))


def expansion_inventory(workspace, root):
    root = Path(root).resolve()
    with expansion_db(workspace) as db:
        previous = expansion_meta(db, 'acs_root')
        if previous and previous != str(root):
            raise ValueError('ACS inventory root differs from the recorded source')
        expansion_meta(db, 'acs_root', str(root))
        count = 0
        for day in sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 999):
            if not day.is_dir() or not day.name.isdigit() or not 1 <= int(day.name) <= 31:
                continue
            for file in sorted(day.iterdir()):
                if file.suffix.lower() not in ('.jpg', '.jpeg', '.png') or not file.is_file():
                    continue
                image_id = hashlib.sha256(('ACS-JAN2026\0'+str(file)).encode()).hexdigest()
                db.execute('INSERT OR IGNORE INTO images(id,kind,path) VALUES(?,?,?)',
                           (image_id, 'acs', str(file)))
                count += 1
            db.commit()
            expansion_meta(db, 'inventory_progress', dict(day=day.name, files=count))
        expansion_meta(db, 'inventory_complete', True)
        return count


def expansion_import(workspace, archive, source_url, license_name):
    """Import image members only; never execute/extract arbitrary ZIP paths or annotations."""
    if not source_url.startswith('https://universe.roboflow.com/') or not license_name.strip():
        raise ValueError('Roboflow provenance and explicit license are required')
    archive = Path(archive).resolve()
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    target = Path(workspace)/'imports'/digest
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as z, expansion_db(workspace) as db:
        members=set(z.namelist())
        if len(members)!=len(z.infolist()): raise ValueError('Duplicate archive member names')
        readmes={}
        for name in ('README.dataset.txt','README.roboflow.txt'):
            if name in members:
                if z.getinfo(name).file_size>100_000: raise ValueError('Oversized source README')
                readmes[name]=z.read(name).decode('utf-8-sig')
        (target/'source-readmes.json').write_text(json.dumps(readmes,ensure_ascii=False,indent=2),encoding='utf-8')
        for member in z.infolist():
            path = Path(member.filename.replace('\\', '/'))
            if path.is_absolute() or '..' in path.parts or ':' in member.filename:
                raise ValueError('Unsafe archive member')
            if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            if member.file_size > 30_000_000 or member.file_size/max(1, member.compress_size) > 1000:
                raise ValueError('Oversized archive image')
            image_id = hashlib.sha256((digest+'\0'+member.filename).encode()).hexdigest()
            output = target/(image_id+path.suffix.lower())
            if not output.exists():
                output.write_bytes(z.read(member))
            proposal = dict(source_url=source_url, license=license_name, archive_sha256=digest,
                            archive_member=member.filename)
            label_member=str(path.parent.parent/'labels'/(path.stem+'.txt')).replace('\\','/')
            boxes=[]
            if label_member in members:
                if z.getinfo(label_member).file_size>100_000: raise ValueError('Oversized label')
                for line in z.read(label_member).decode('utf-8').splitlines():
                    values=list(map(float,line.split()))
                    if not values: continue
                    if len(values)!=5 or any(not 0<=v<=1 for v in values[1:]) or min(values[3:])<=0:
                        raise ValueError('Expected YOLO detection boxes; four corners are still unlabelled')
                    boxes.append(values[1:])
            proposal['source_detection_boxes']=boxes
            db.execute('INSERT OR IGNORE INTO images(id,kind,path,proposal) VALUES(?,?,?,?)',
                       (image_id, 'roboflow', str(output), json.dumps(proposal)))
            old=db.execute('SELECT proposal FROM images WHERE id=?',(image_id,)).fetchone()
            merged=json.loads(old[0]); merged.update(proposal)
            db.execute('UPDATE images SET proposal=? WHERE id=?',(json.dumps(merged),image_id))
            count += 1
        db.commit()
        expansion_meta(db, 'roboflow_import', dict(count=count, source_url=source_url,
                                                  license=license_name, sha256=digest))
    return count


def expansion_assets(workspace, image_id, data, points):
    import cv2
    import numpy as np
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
    preview = image.copy()
    preview.thumbnail((1280, 1280))
    expansion_jpeg(preview,Path(workspace)/'previews'/(image_id+'.jpg'),90)
    if not points:
        return
    valid_points(points)
    array = np.array(image)
    h, w = array.shape[:2]
    pts = np.array(points, dtype=np.float32)*[w, h]
    width = max(np.linalg.norm(pts[1]-pts[0]), np.linalg.norm(pts[2]-pts[3]))
    height = max(np.linalg.norm(pts[3]-pts[0]), np.linalg.norm(pts[2]-pts[1]))
    cw, ch = 480, int(np.clip(480*height/max(1,width), 64, 480))
    transform = cv2.getPerspectiveTransform(pts.astype('float32'),
        np.array([[0,0],[cw-1,0],[cw-1,ch-1],[0,ch-1]], dtype='float32'))
    crop = cv2.warpPerspective(array, transform, (cw, ch))
    expansion_jpeg(Image.fromarray(crop),Path(workspace)/'crops'/(image_id+'.jpg'),95)


def expansion_jpeg(image, path, quality):
    temporary=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    image.save(temporary,format='JPEG',quality=quality)
    try:
        for attempt in range(8):
            try:
                os.replace(temporary,path)
                break
            except PermissionError:
                if attempt==7: raise
                time.sleep(.05*(attempt+1))
    finally:
        temporary.unlink(missing_ok=True)


def expansion_process(workspace, model_path, limit=0):
    import torch
    torch.set_num_threads(4)
    os.environ['YOLO_CONFIG_DIR'] = str(Path(workspace)/'yolo-config')
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    if model.model.yaml.get('kpt_shape') != [4, 3]:
        raise ValueError('Expected the locally trained four-corner plate model')
    digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    with expansion_db(workspace) as db:
        previous = expansion_meta(db, 'model_sha256')
        if previous and previous != digest:
            raise ValueError('Cannot mix proposal models in one workspace')
        expansion_meta(db, 'model_sha256', digest)
        expansion_meta(db, 'processing_complete', False)
        # Exact/pixel matches block immediately. Near matches are review candidates only.
        for r in db.execute('SELECT id,dhash FROM images WHERE dhash IS NOT NULL').fetchall():
            expansion_index_hash(db,r['id'],r['dhash'])
        db.commit()
        done = 0
        while not limit or done < limit:
            row = db.execute("SELECT * FROM images WHERE stage='pending' ORDER BY "
                "CASE kind WHEN 'roboflow' THEN 0 ELSE 1 END,rowid LIMIT 1").fetchone()
            if not row:
                break
            proposal = json.loads(row['proposal'])
            try:
                data = Path(row['path']).read_bytes()
                sha = hashlib.sha256(data).hexdigest()
                size, pixel, dhash = fingerprints(data)
                if isinstance(dhash,int): dhash=format(dhash,'016x')
                match = db.execute('SELECT id,kind FROM images WHERE id<>? AND (sha=? OR pixel=?) '
                                   'AND sha IS NOT NULL LIMIT 1', (row['id'],sha,pixel)).fetchone()
                if match:
                    stage = 'duplicate'
                    proposal['duplicate_of'] = dict(match)
                else:
                    points = []
                    detections = []
                    if row['kind']=='acs' and min(size) < 350 and max(size) < 800:
                        stage = 'crop_only'
                        points = [[.01,.01],[.99,.01],[.99,.99],[.01,.99]]
                    else:
                        result = model.predict(Image.open(io.BytesIO(data)), imgsz=960,
                                               conf=.12, device='cpu', verbose=False)[0]
                        if result.keypoints is not None:
                            for box, pts in zip(result.boxes, result.keypoints.xyn):
                                p = pts.cpu().tolist()
                                try:
                                    valid_points(p)
                                except ValueError:
                                    continue
                                detections.append(dict(points=p, confidence=float(box.conf[0])))
                        if detections:
                            detections.sort(key=lambda d:d['confidence'], reverse=True)
                            points = detections[0]['points']
                        stage = 'ready' if len(detections)==1 else ('multiple' if detections else 'no_detection')
                        boxes=proposal.get('source_detection_boxes',[])
                        if row['kind']=='roboflow' and len(boxes)==1:
                            cx,cy,bw,bh=boxes[0]
                            contour=contour_proposal(data,[cx,cy,bw,bh])
                            if contour:
                                # A contour is only a proposal, never a verified corner label.
                                points=contour;proposal['corner_method']='contour-in-source-bbox'
                                stage='ready'
                            elif points:
                                px=sum(p[0] for p in points)/4;py=sum(p[1] for p in points)/4
                                if not (cx-bw/2<=px<=cx+bw/2 and cy-bh/2<=py<=cy+bh/2):
                                    points=[];stage='no_detection'
                        if row['kind']=='roboflow' and len(boxes)>1: stage='multiple'
                    near=expansion_near_hashes(db,row['id'],dhash)
                    proposal.update(points=points, detections=detections, size=size,
                                    near=near[:30], near_total=len(near))
                    expansion_assets(workspace, row['id'], data, points)
                db.execute('UPDATE images SET stage=?,sha=?,pixel=?,dhash=?,proposal=?,error=NULL '
                           'WHERE id=?', (stage,sha,pixel,dhash,json.dumps(proposal),row['id']))
                expansion_index_hash(db,row['id'],dhash)
            except Exception as exc:
                db.execute("UPDATE images SET stage='error',error=? WHERE id=?",
                           (type(exc).__name__, row['id']))
            db.commit()
            done += 1
            if done % 100 == 0:
                print(json.dumps(dict(processed_this_run=done, stages=dict(db.execute(
                    'SELECT stage,count(*) FROM images GROUP BY stage').fetchall()))), flush=True)
        if not db.execute("SELECT 1 FROM images WHERE stage='pending' LIMIT 1").fetchone():
            expansion_meta(db, 'processing_complete', True)
    return done


def expansion_ocr(workspace, follow=False, limit=0):
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1, text_score=0.0)
    count = 0
    while not limit or count < limit:
        with expansion_db(workspace) as db:
            rows = db.execute("SELECT * FROM images WHERE ocr_done=0 AND stage NOT IN "
                "('pending','duplicate','error','no_detection') ORDER BY "
                "CASE kind WHEN 'validation' THEN 0 WHEN 'roboflow' THEN 1 WHEN 'acs' THEN 2 ELSE 3 END,rowid LIMIT 100").fetchall()
            if not rows:
                if follow and not expansion_meta(db, 'processing_complete'):
                    time.sleep(3)
                    continue
                break
            for row in rows:
                if limit and count >= limit:
                    break
                proposal = json.loads(row['proposal'])
                path = Path(workspace)/'crops'/(row['id']+'.jpg')
                try:
                    if not path.exists():
                        expansion_assets(workspace,row['id'],Path(row['path']).read_bytes(),proposal['points'])
                    result,_ = engine(str(path),use_det=True,use_cls=False,use_rec=True)
                    lines = sorted(result or [],key=lambda x:(sum(p[1] for p in x[0])/4,min(p[0] for p in x[0])))
                    text = expansion_plate(''.join(str(x[1]) for x in lines))
                    candidates=[dict(plate=text,confidence=min((float(x[2]) for x in lines),default=0),method='det-lines')]
                    if not expansion_valid_plate(text):
                        whole,_=engine(str(path),use_det=False,use_cls=False,use_rec=True)
                        if whole:
                            candidates.append(dict(plate=expansion_plate(whole[0][0]),confidence=float(whole[0][1]),method='whole-crop'))
                    candidates.sort(key=lambda c:(expansion_valid_plate(c['plate']),c['confidence']),reverse=True)
                    ocr = dict(**candidates[0],candidates=candidates,
                               valid_format=expansion_valid_plate(candidates[0]['plate']), engine='RapidOCR 1.4.4')
                except Exception as exc:
                    ocr = dict(plate='',error=type(exc).__name__)
                # Merge fresh JSON: another worker may have advanced the row meanwhile.
                latest = db.execute('SELECT proposal FROM images WHERE id=?',(row['id'],)).fetchone()
                proposal = json.loads(latest[0]); proposal['ocr'] = ocr
                db.execute('UPDATE images SET proposal=?,ocr_done=1 WHERE id=?',
                           (json.dumps(proposal),row['id']))
                db.commit()
                count += 1
                if count % 100 == 0:
                    print(json.dumps(dict(ocr_this_run=count)),flush=True)
    return count


class ExpansionReviewStore:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        classification=self.workspace/'frozen-vehicle-types.json'
        self.frozen_vehicle_types=json.loads(classification.read_text(encoding='utf-8')).get('records',{}) if classification.exists() else {}
        with expansion_db(self.workspace) as db:
            if not expansion_meta(db,'manifest_sha256'):
                raise ValueError('Initialize the pinned validation workspace first')

    def vehicle_type_for_row(self, row):
        if row['kind'] in ('baseline','validation'):
            entry=self.frozen_vehicle_types.get(row['id'],{})
            category=entry.get('vehicle_type','unknown') if entry.get('sha256')==row['sha'] else 'unknown'
            if row['kind']=='validation' and json.loads(row['proposal']).get('validation_approval'):
                category=json.loads(row['edit']).get('vehicle_type','unknown')
        else:
            category=json.loads(row['edit']).get('vehicle_type','unknown')
        return category if category in ('car','motorcycle') else 'unknown'

    def identity_index(self, db):
        plates, missing = set(), 0
        for row in db.execute("SELECT edit FROM images WHERE kind='validation'"):
            edit = json.loads(row[0])
            if edit.get('identity_reviewed') and expansion_valid_plate(edit.get('plate','')):
                plates.add(edit['plate'])
            else:
                missing += 1
        return plates, missing

    def suggestion_index(self, db):
        suggestions={}
        for row in db.execute("SELECT id,edit,proposal FROM images WHERE kind='validation'"):
            edit,proposal=json.loads(row['edit']),json.loads(row['proposal'])
            plate=edit.get('plate','') if edit.get('identity_reviewed') else proposal.get('ocr',{}).get('plate','')
            if expansion_valid_plate(plate): suggestions.setdefault(plate,[]).append(row['id'])
        return suggestions

    def public_row(self, row, plates, missing, suggestions=None, triage=None, assets=True):
        p,e = json.loads(row['proposal']),json.loads(row['edit'])
        plate = e.get('plate',p.get('ocr',{}).get('plate',''))
        status = row['stage']
        if row['kind']=='validation':
            status = 'identity_done' if e.get('identity_reviewed') else 'identity_pending'
        elif e.get('exclude'):
            status = 'excluded'
        elif e.get('validation_comparison')=='same':
            status = 'validation_overlap'
        elif e.get('identity_reviewed') and plate in plates:
            status = 'validation_overlap'
        elif plate and (plate in plates or plate in (suggestions or {})):
            status = 'suspected_overlap'
        elif e.get('corners_reviewed'):
            status = 'corners_done' if e.get('review_scope') == 'corners' else ('waiting_validation' if missing else 'approved')
        item=dict(id=row['id'],kind=row['kind'],stage=row['stage'],status=status,
            points=e.get('points',p.get('points',[])),plate=plate,version=row['version'],
            reviewer=e.get('reviewer',''),identity_reviewed=e.get('identity_reviewed',False),
            identity_method=e.get('identity_method','human' if e.get('identity_reviewed') else ''),
            single_plate_reviewed=e.get('single_plate_reviewed',False),
            corners_reviewed=e.get('corners_reviewed',False),
            train_ready=row['kind']!='validation' and bool(e.get('corners_reviewed') or row['kind']=='baseline') and not e.get('exclude') and row['stage'] not in ('excluded','duplicate'),
            review_scope=e.get('review_scope',''),vehicle_type=self.vehicle_type_for_row(row),
            validation_comparison=e.get('validation_comparison',''),
            near=p.get('near',[]),duplicate_of=p.get('duplicate_of'),
            suspected_validation_ids=(suggestions or {}).get(plate,[])[:12] if row['kind']!='validation' else [],
            source_url=p.get('source_url'),ocr=p.get('ocr'),
            preview=assets and (row['kind'] in ('validation','baseline') or (self.workspace/'previews'/(row['id']+'.jpg')).exists()),
            crop=assets and (row['kind'] in ('validation','baseline') or (self.workspace/'crops'/(row['id']+'.jpg')).exists()))
        checked=p.get('ocr',{}).get('crop_check')
        if checked:
            item['crop']=assets and checked.get('status')=='pass' and (self.workspace/'ocr-crops'/(row['id']+'.jpg')).is_file()
        if triage is not None:
            item['triage']=triage
            item['suspected_validation_ids']=list(dict.fromkeys(item['suspected_validation_ids']+
                                                    [m['id'] for m in triage['matches']]))[:20]
        return item

    def listing(self, kind='acs', status='pending', offset=0, vehicle='all', anchor=''):
        if vehicle not in ('all','unknown','car','motorcycle'): raise ValueError('Invalid vehicle filter')
        if kind not in ('acs','roboflow','validation','baseline','all') or status not in ('pending','all','approved','blocked','matched','uncertain','distinct','ocr_wait','corners','corners_done','validation'):
            raise ValueError('Invalid filter')
        from review_queue_cache import queue_signature, cached_page
        cache_key=queue_signature(self.workspace,kind,status,vehicle)
        with expansion_db(self.workspace) as db:
            storage_root=expansion_meta(db,'image_storage_root')
        # External moves/deletions do not change SQLite; live disk counts must rescan.
        if storage_root: cache_key=None
        cached=cached_page(self,cache_key,offset,anchor)
        if cached is not None: return cached
        from corner_review_index import load_corner_review_index, needs_corner_review
        corner_index=load_corner_review_index(self.workspace) if status=='corners' else {}
        with expansion_db(self.workspace) as db:
            plates,missing=self.identity_index(db)
            suggestions=self.suggestion_index(db)
            reference_index=triage_validation_index(db.execute("SELECT * FROM images WHERE kind='validation'").fetchall())
            rows=db.execute('SELECT * FROM images ORDER BY rowid').fetchall()
            if storage_root:
                root=Path(storage_root).absolute()
                canonical_root=root.resolve()
                live=[]
                for row in rows:
                    edit=json.loads(row['edit']);proposal=json.loads(row['proposal'])
                    if row['stage'] in ('excluded','duplicate') or edit.get('exclude') or proposal.get('source_deleted'): continue
                    path=Path(row['path'])
                    if not path.is_absolute() or not (path.is_relative_to(root) or path.is_relative_to(canonical_root)): continue
                    if path.is_file(): live.append(row)
                rows=live
            stages=dict(Counter(row['stage'] for row in rows))
            counts=dict(Counter(row['kind'] for row in rows))
            selected=[]; matches=0; statuses=Counter(); triage_counts=Counter(); candidates=[]
            for row in rows:
                if kind!='all' and row['kind']!=kind: continue
                if row['stage']=='pending':
                    statuses['pending']+=1
                    triage_counts['pending']+=1
                    continue
                if vehicle!='all' and self.vehicle_type_for_row(row)!=vehicle: continue
                if status=='corners' and not needs_corner_review(row,corner_index): continue
                triage=triage_record(row,reference_index) if status in ('matched','uncertain','distinct','ocr_wait') else None
                if triage: triage_counts[triage['category']]+=1
                item=self.public_row(row,plates,missing,suggestions,triage,assets=False); statuses[item['status']]+=1
                if status=='corners' and not needs_corner_review(row,corner_index): continue
                if status=='corners_done' and not item['train_ready']: continue
                if status=='validation' and row['kind']!='validation': continue
                if status=='matched' and triage['category'] not in ('suspected_overlap','confirmed_overlap'): continue
                if status=='uncertain' and triage['category'] not in ('ocr_uncertain','file_error'): continue
                if status=='distinct' and triage['category']!='ocr_distinct': continue
                if status=='ocr_wait' and triage['category']!='ocr_pending': continue
                if status=='pending' and (item['status'] in ('approved','identity_done','excluded','duplicate','validation_overlap')
                                         or item['train_ready'] or row['stage']=='crop_only'): continue
                if status=='approved' and item['status'] not in ('approved','identity_done','waiting_validation'): continue
                if status=='blocked' and item['status'] not in ('duplicate','validation_overlap','suspected_overlap','error'): continue
                candidates.append((row,triage))
                matches+=1
            if anchor:
                offset=next((n for n,(row,_) in enumerate(candidates) if row['id']==anchor),offset)
            offset=min(offset,max(0,matches-1))
            selected=[self.public_row(row,plates,missing,suggestions,triage or triage_record(row,reference_index)) for row,triage in candidates[offset:offset+40]]
            if status not in ('matched','uncertain','distinct','ocr_wait'):
                triage_counts=Counter(item['triage']['category'] for item in selected)
            result=dict(records=selected,total=matches,offset=offset,counts=counts,stages=stages,
                statuses=dict(statuses),validation_missing_identities=missing,
                triage_counts=dict(triage_counts),triage_counts_scope=('filter' if status in ('matched','uncertain','distinct','ocr_wait') else 'page'),validation_ocr_coverage=reference_index['coverage'],
                processing_complete=bool(expansion_meta(db,'processing_complete')),
                ocr_complete=bool(expansion_meta(db,'ocr_all_complete')),
                ocr_pending_processed=db.execute("SELECT count(*) FROM images WHERE stage NOT IN ('pending','duplicate','error') AND coalesce(json_extract(edit,'$.exclude'),0)=0 AND json_extract(proposal,'$.source_deleted') IS NULL AND (coalesce(json_extract(proposal,'$.ocr.schema'),0)<>2 OR coalesce(json_extract(proposal,'$.ocr.layout_aware'),0)<>1 OR (?<>0 AND coalesce(json_extract(proposal,'$.ocr.crop_pipeline'),0)<>?))",(expansion_meta(db,'crop_pipeline') or 0,expansion_meta(db,'crop_pipeline') or 0)).fetchone()[0],
                roboflow=expansion_meta(db,'roboflow_import'))
            if cache_key is not None:
                self._review_queue_snapshot=(cache_key,candidates,plates,missing,suggestions,reference_index,result)
            return result

    def save(self, request):
        if request.get('action')=='classify':
            from review_vehicle_storage import classify_vehicle
            return classify_vehicle(self,request)
        image_id=request.get('id'); version=request.get('version'); edit=request.get('edit')
        action=request.get('action','save')
        if action not in ('save','approve','approve_corners','approve_validation','return_corners','undo_corners'): raise ValueError('Invalid action')
        corners_only=action in ('approve_corners','approve_validation')
        if corners_only and (not isinstance(edit,dict) or set(edit)-{'points','reviewer','single_plate_reviewed'}):
            raise ValueError('Duyệt góc chỉ nhận bốn điểm, người duyệt và xác nhận một biển')
        if not isinstance(edit,dict) or set(edit)-{'points','plate','reviewer','identity_reviewed','corners_reviewed','exclude','near_checked','single_plate_reviewed','validation_comparison'}:
            raise ValueError('Invalid fields')
        if type(version) is not int or version<0: raise ValueError('Invalid revision')
        for key in ('identity_reviewed','corners_reviewed','exclude','near_checked','single_plate_reviewed'):
            if type(edit.get(key,False)) is not bool: raise ValueError('Invalid boolean')
        for key in ('plate','reviewer'):
            if not isinstance(edit.get(key,''),str) or len(edit.get(key,''))>100: raise ValueError('Invalid text')
        edit['plate']=expansion_plate(edit.get('plate',''))
        comparison=edit.get('validation_comparison','')
        if comparison not in ('','different','same'): raise ValueError('Invalid validation comparison')
        if edit.get('identity_reviewed') and (not expansion_valid_plate(edit['plate']) or not edit.get('reviewer','').strip()):
            raise ValueError('Cần biển số đúng định dạng và người duyệt')
        with expansion_db(self.workspace) as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM images WHERE id=?',(image_id,)).fetchone()
            if row is None: raise ValueError('Unknown image')
            if row['version']!=version: raise RuntimeError('Ảnh đã được sửa trên tab khác. Tải lại trước khi lưu.')
            proposal=json.loads(row['proposal'])
            previous=json.loads(row['edit'])
            validation_origin=proposal.get('validation_approval')
            undo_validation=(action=='undo_corners' and row['kind']=='validation' and
                             isinstance(validation_origin,dict) and validation_origin.get('version')==version)
            effective_kind=validation_origin['kind'] if undo_validation else row['kind']
            target_kind=effective_kind
            if action=='undo_corners':
                if effective_kind not in ('acs','roboflow') or previous.get('exclude') or proposal.get('source_deleted'):
                    raise ValueError('Ảnh bị khóa hoặc đã loại không thể hoàn tác')
                prior=db.execute('SELECT value FROM history WHERE id=? AND version=? ORDER BY rowid DESC LIMIT 1',(image_id,version-1)).fetchone()
                if prior is None: raise ValueError('Không tìm thấy lịch sử để hoàn tác')
                old=json.loads(prior['value']);edit=dict(previous)
                for key in ('points','corners_reviewed','single_plate_reviewed','review_scope','reviewer'):
                    if key in old: edit[key]=old[key]
                    else: edit.pop(key,None)
            if action=='return_corners':
                if row['kind'] not in ('acs','roboflow') or row['stage'] in ('duplicate','excluded') or previous.get('exclude') or proposal.get('source_deleted'):
                    raise ValueError('Ảnh bị khóa hoặc đã loại không thể chuyển về kiểm tra góc')
                edit=dict(previous,corners_reviewed=False)
            if 'vehicle_type' in previous:
                for key in ('vehicle_type','vehicle_reviewer','vehicle_reviewed_at'):
                    if key in previous: edit[key]=previous[key]
            if corners_only:
                if row['kind'] not in ('acs','roboflow') or previous.get('exclude') or proposal.get('source_deleted'):
                    raise ValueError('Ảnh bị khóa hoặc đã loại không thể duyệt góc')
                if not edit.get('reviewer','').strip(): raise ValueError('Cần nhập người duyệt góc')
                edit=dict(previous,points=edit.get('points',[]),reviewer=edit['reviewer'].strip(),
                          single_plate_reviewed=edit.get('single_plate_reviewed',False),
                          corners_reviewed=True,review_scope='corners')
            elif previous.get('review_scope')=='corners' and action!='approve':
                edit['review_scope']='corners'
            reference_index=triage_validation_index(db.execute("SELECT * FROM images WHERE kind='validation'").fetchall())
            if comparison and not edit.get('exclude') and not corners_only:
                if row['kind']=='validation': raise ValueError('Validation chỉ duyệt định danh')
                if not edit.get('identity_reviewed'): raise ValueError('Xác nhận biển số trước khi kết luận trùng xe')
                proposed=dict(row);proposed['edit']=json.dumps(edit)
                result=triage_record(proposed,reference_index)
                if not result['match_count']: raise ValueError('Không có ảnh validation tham chiếu để kết luận')
                if comparison=='different' and result['has_exact_match']:
                    raise ValueError('Biển số khớp chính xác: kiểm tra và sửa định danh validation nếu OCR sai')
                previous=json.loads(row['edit'])
                if (previous.get('validation_comparison')!=comparison or previous.get('validation_comparison_digest')!=result['match_digest']) and not edit.get('near_checked'):
                    raise ValueError('Phải xem ảnh validation tham chiếu và đánh dấu đã so ảnh')
                edit['validation_comparison_digest']=result['match_digest']
            if effective_kind in ('validation','baseline'):
                if edit.get('points',proposal['points'])!=proposal['points'] or edit.get('exclude') or edit.get('corners_reviewed'):
                    raise ValueError('Nhãn và thành viên v3 được khóa; chỉ duyệt định danh biển số')
            elif edit.get('corners_reviewed'):
                if row['stage'] in ('duplicate','error','pending','crop_only'):
                    raise ValueError('Ảnh trùng/lỗi/crop/nhiều biển chưa thể xuất train')
                if not edit.get('single_plate_reviewed'):
                    raise ValueError('Phải xác nhận ảnh có đúng một biển Việt Nam và đủ bốn góc')
                if len(proposal.get('source_detection_boxes',[]))>1:
                    raise ValueError('Nguồn gán nhiều biển: cần công cụ nhãn nhiều đối tượng')
                valid_points(edit.get('points',[]))
                if edit.get('review_scope')!='corners':
                    if not edit.get('identity_reviewed'): raise ValueError('Phải xác nhận biển số')
                    if proposal.get('near') and not edit.get('near_checked'): raise ValueError('Phải kiểm tra ảnh gần trùng v3')
            if 'points' in edit and edit['points']: valid_points(edit['points'])
            source=Path(row['path']); target=source; moved=False
            root_value=expansion_meta(db,'image_storage_root')
            if action=='approve_validation' and not root_value: raise ValueError('Cần cấu hình thư mục dữ liệu trước khi duyệt validation')
            if root_value and effective_kind in ('acs','roboflow') and edit.get('corners_reviewed') and not edit.get('exclude') and not undo_validation:
                check_split_overlap(db,root_value,row,edit,validation=action=='approve_validation')
            if root_value and effective_kind in ('acs','roboflow'):
                root=Path(root_value).resolve();approved=train_images(root).resolve()
                vehicle=edit.get('vehicle_type') or previous.get('vehicle_type') or 'motorcycle'
                folder={'car':'o-to','motorcycle':'xe-may','unknown':'xe-may'}.get(vehicle,'xe-may')
                if undo_validation:
                    target=Path(validation_origin['path'])
                    if not target.resolve().is_relative_to(root): raise ValueError('Đường dẫn khôi phục không hợp lệ')
                    proposal=validation_origin['proposal']
                elif action=='approve_validation':
                    target=root/'du-lieu-yolo-point'/'validation'/'images'/folder/source.name
                    proposal=dict(proposal,points=edit['points'],validation_approval=dict(kind=row['kind'],path=str(source),proposal=proposal,version=version+1))
                    target_kind='validation'
                elif edit.get('corners_reviewed') and not edit.get('exclude'):
                    target=approved/folder/source.name
                    if not source.resolve().is_relative_to(approved): edit['corner_source_path']=str(source)
                elif source.resolve().is_relative_to(approved):
                    original=previous.get('corner_source_path')
                    target=Path(original) if original else root/'anh-goc'/('ACS' if row['kind']=='acs' else 'Roboflow')/folder/source.name
                    if not target.resolve().is_relative_to(root/'anh-goc'): raise ValueError('Đường dẫn ảnh gốc không hợp lệ')
                    edit.pop('corner_source_path',None)
                if source.resolve()!=target.resolve():
                    if not source.is_file(): raise ValueError('Không tìm thấy tệp ảnh gốc để chuyển')
                    if target.exists(): raise ValueError('Ảnh đã tồn tại ở thư mục đích; kiểm tra trùng trước khi chuyển')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    os.replace(source,target);moved=True
            rollback_labels=lambda: None
            try:
                if root_value and effective_kind in ('acs','roboflow'):
                    rollback_labels=sync_labels(Path(root_value),source,target,edit,proposal)
                db.execute('INSERT INTO history VALUES(?,?,?,?)',(image_id,version,row['edit'],time.time_ns()))
                db.execute('UPDATE images SET path=?,kind=?,proposal=?,edit=?,version=version+1 WHERE id=?',(str(target),target_kind,json.dumps(proposal),json.dumps(edit),image_id))
                db.commit()
            except Exception:
                rollback_labels()
                if moved: os.replace(target,source)
                raise
            plates,missing=self.identity_index(db)
            updated=db.execute('SELECT * FROM images WHERE id=?',(image_id,)).fetchone()
            return self.public_row(updated,plates,missing,self.suggestion_index(db),triage_record(updated,reference_index))

    def asset(self, image_id, crop=False):
        if not re.fullmatch('[a-f0-9]{64}', image_id): return None
        with expansion_db(self.workspace) as db:
            row=db.execute('SELECT * FROM images WHERE id=?',(image_id,)).fetchone()
            if row is None: return None
        checked=json.loads(row['proposal']).get('ocr',{}).get('crop_check')
        if crop and checked and checked.get('status')!='pass': return None
        path=self.workspace/('ocr-crops' if crop and checked else ('crops' if crop else 'previews'))/(image_id+'.jpg')
        if not path.exists() and row['kind'] in ('validation','baseline'):
            data=Path(row['path']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=row['sha']: raise ValueError('Frozen image changed')
            expansion_assets(self.workspace,image_id,data,json.loads(row['proposal'])['points'])
        return path.read_bytes() if path.is_file() else None

    def export_reviews(self):
        with expansion_db(self.workspace) as db:
            plates,missing=self.identity_index(db)
            suggestions=self.suggestion_index(db)
            records=[self.public_row(r,plates,missing,suggestions) for r in db.execute('SELECT * FROM images') if json.loads(r['edit'])]
            return dict(schema=1,manifest_sha256=expansion_meta(db,'manifest_sha256'),
                training_ready=False,validation_missing_identities=missing,records=records,
                note='Review export only. Recheck frozen files and all identities before packaging.')


class ExpansionReviewHandler(ReviewHandler):
    def do_GET(self):
        if not self.authenticated(): return
        parts=urlsplit(self.path); query=parse_qs(parts.query)
        try:
            if parts.path=='/': return self.respond(200,self.server.template.encode('utf-8'),'text/html; charset=utf-8')
            if parts.path=='/api/list':
                return self.respond(200,self.server.store.listing(query.get('kind',['acs'])[0],
                    query.get('status',['pending'])[0],max(0,int(query.get('offset',['0'])[0])),query.get('vehicle',['all'])[0],query.get('anchor',[''])[0]))
            if parts.path=='/api/export': return self.respond(200,self.server.store.export_reviews())
            match=re.fullmatch(r'/(preview|crop)/([a-f0-9]{64})\.jpg',parts.path)
            if match:
                data=self.server.store.asset(match[2],match[1]=='crop')
                if data: return self.respond(200,data,'image/jpeg')
            self.respond(404,{'error':'Not found'})
        except (ValueError,KeyError): self.respond(400,{'error':'Invalid request'})

    def do_POST(self):
        if not self.authenticated(): return
        if self.path!='/api/save': return self.respond(404,{'error':'Not found'})
        if self.headers.get('Origin')!='http://'+self.headers.get('Host','') or self.headers.get('X-Review-Request')!='1':
            return self.respond(403,{'error':'Invalid origin'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=8192 or self.headers.get('Content-Type')!='application/json': raise ValueError('Invalid body')
            self.respond(200,self.server.store.save(json.loads(self.rfile.read(length))))
        except RuntimeError as exc: self.respond(409,{'error':str(exc)})
        except (ValueError,KeyError,TypeError) as exc: self.respond(400,{'error':str(exc)})


def expansion_serve(workspace, credentials, bind, port, networks):
    server=ThreadingHTTPServer((bind,port),ExpansionReviewHandler)
    server.daemon_threads=True
    server.store=ExpansionReviewStore(workspace)
    server.networks=[ipaddress.ip_network(n) for n in networks]
    login=json.loads(Path(credentials).read_text(encoding='utf-8'))
    server.expected_auth='Basic '+base64.b64encode((login['username']+':'+login['password']).encode()).decode()
    server.template=Path(__file__).with_name('review_expansion.html').read_text(encoding='utf-8')
    print('Private expansion review server ready',flush=True)
    try: server.serve_forever()
    finally: server.server_close()


def curate_acs_training_pool():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('init'); p.add_argument('--dataset',required=True); p.add_argument('--snapshot')
    p=sub.add_parser('inventory'); p.add_argument('--root',required=True)
    p=sub.add_parser('import'); p.add_argument('--archive',required=True); p.add_argument('--source-url',required=True); p.add_argument('--license',required=True)
    p=sub.add_parser('process'); p.add_argument('--model',required=True); p.add_argument('--limit',type=int,default=0)
    p=sub.add_parser('ocr'); p.add_argument('--follow',action='store_true'); p.add_argument('--limit',type=int,default=0)
    p=sub.add_parser('ocr-all'); p.add_argument('--follow',action='store_true'); p.add_argument('--limit',type=int,default=0)
    p=sub.add_parser('status')
    p=sub.add_parser('serve'); p.add_argument('--credentials',required=True); p.add_argument('--bind',default='127.0.0.1'); p.add_argument('--port',type=int,default=8767); p.add_argument('--allow-network',action='append',default=['127.0.0.0/8'])
    args=parser.parse_args(); w=args.workspace
    if args.command=='init': result=expansion_init(w,args.dataset,args.snapshot)
    elif args.command=='inventory': result=expansion_inventory(w,args.root)
    elif args.command=='import': result=expansion_import(w,args.archive,args.source_url,args.license)
    elif args.command=='process': result=expansion_process(w,args.model,args.limit)
    elif args.command=='ocr': result=expansion_ocr(w,args.follow,args.limit)
    elif args.command=='ocr-all':
        from expansion_ocr_all import expansion_ocr_all
        result=expansion_ocr_all(w,args.follow,args.limit)
    elif args.command=='status': result=ExpansionReviewStore(w).listing()
    else: return expansion_serve(w,args.credentials,args.bind,args.port,args.allow_network)
    if args.command=='status': result={k:v for k,v in result.items() if k!='records'}
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    curate_acs_training_pool()
