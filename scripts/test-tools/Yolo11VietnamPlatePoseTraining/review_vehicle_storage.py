"""Explicit vehicle classification with one canonical image file."""
import hashlib
import json
import os
from pathlib import Path
import time
from review_train_storage import train_images, sync_labels


def classify_vehicle(store, request):
    from curate_acs_training_pool import expansion_db, expansion_meta
    category = request.get('vehicle_type')
    if category not in ('unknown', 'car', 'motorcycle'):
        raise ValueError('Invalid vehicle category')
    if (type(request.get('version')) is not int or request['version']<0 or
        not isinstance(request.get('reviewer'),str) or not request['reviewer'].strip() or len(request['reviewer'])>100):
        raise ValueError('Revision and reviewer required')
    with expansion_db(store.workspace) as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM images WHERE id=?', (request.get('id'),)).fetchone()
        if row is None: raise ValueError('Unknown image')
        if row['version'] != request['version']: raise RuntimeError('Ảnh đã thay đổi. Tải lại trước khi chuyển nhóm.')
        edit, proposal = json.loads(row['edit']), json.loads(row['proposal'])
        if row['kind'] not in ('acs', 'roboflow') or edit.get('exclude') or proposal.get('source_deleted') or row['stage'] in ('excluded', 'duplicate'):
            raise ValueError('Ảnh bị khóa hoặc đã loại không thể chuyển nhóm')
        root_value = expansion_meta(db, 'image_storage_root')
        if not root_value: raise ValueError('Chưa hoàn tất cấu hình thư mục ảnh')
        root = Path(root_value).resolve()
        source = Path(row['path'])
        folder = {'unknown': 'chua-phan-loai', 'car': 'o-to', 'motorcycle': 'xe-may'}[category]
        base = train_images(root) if edit.get('corners_reviewed') else root/'anh-goc'/('ACS' if row['kind']=='acs' else 'Roboflow')
        target = base/folder/(row['id']+source.suffix.lower())
        target = target.resolve()
        if not target.is_relative_to(root): raise ValueError('Invalid destination')
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha']: raise ValueError('Source hash mismatch')
        target.parent.mkdir(parents=True, exist_ok=True)
        moved = source.resolve()!=target
        if moved and target.exists(): raise ValueError('Ảnh đã tồn tại ở thư mục đích; kiểm tra trùng trước khi chuyển')
        edit.update(vehicle_type=category, vehicle_reviewer=request['reviewer'].strip(), vehicle_reviewed_at=time.time_ns())
        if edit.get('corners_reviewed'):
            edit['corner_source_path']=str(root/'anh-goc'/('ACS' if row['kind']=='acs' else 'Roboflow')/folder/source.name)
        if moved: os.replace(source,target)
        rollback_labels=lambda: None
        try:
            rollback_labels=sync_labels(root,source,target,edit,proposal)
            db.execute('INSERT INTO history VALUES(?,?,?,?)', (row['id'],row['version'],row['edit'],time.time_ns()))
            db.execute('UPDATE images SET path=?,edit=?,version=version+1 WHERE id=?', (str(target),json.dumps(edit),row['id']))
            db.commit()
        except Exception:
            rollback_labels()
            if moved: os.replace(target,source)
            raise
        updated=db.execute('SELECT * FROM images WHERE id=?',(row['id'],)).fetchone()
        plates,missing=store.identity_index(db)
        return store.public_row(updated,plates,missing,store.suggestion_index(db))
