"""Transactional label updates for the live pose training folder."""
from pathlib import Path
from prepare_pose_dataset import label_text
import json


def train_images(root):
    return Path(root) / 'du-lieu-yolo-point' / 'train' / 'images'


def check_split_overlap(db, root, row, edit, validation=False):
    """Reject exact copies or reliable matching plate identities across splits."""
    from curate_acs_training_pool import expansion_valid_plate
    other=Path(root)/'du-lieu-yolo-point'/('train' if validation else 'validation')/'images'
    def plate(record, value):
        if value.get('identity_reviewed') and expansion_valid_plate(value.get('plate','')):
            return value['plate']
        ocr=json.loads(record['proposal']).get('ocr',{})
        candidate=ocr.get('plate','')
        return candidate if ocr.get('confidence',0)>=.97 and expansion_valid_plate(candidate) else None
    candidate_plate=plate(row,edit)
    for candidate in db.execute("SELECT * FROM images WHERE id<>? AND stage NOT IN ('excluded','duplicate')",(row['id'],)):
        path=Path(candidate['path'])
        if not path.is_relative_to(other): continue
        value=json.loads(candidate['edit'])
        if value.get('exclude'): continue
        same=(bool(row['sha']) and candidate['sha']==row['sha']) or (bool(row['pixel']) and candidate['pixel']==row['pixel'])
        if not same and candidate_plate: same=plate(candidate,value)==candidate_plate
        if same and path.is_file():
            raise ValueError('Ảnh trùng hoặc cùng biển số với tập còn lại; cần kiểm tra nhóm trước khi duyệt')


def sync_labels(root, source, target, edit, proposal):
    """Return a rollback callback; callers commit SQLite only after this succeeds."""
    image_roots=[train_images(root).resolve(),(Path(root)/'du-lieu-yolo-point'/'validation'/'images').resolve()]
    def label_for(image):
        resolved=Path(image).resolve()
        for images in image_roots:
            if resolved.is_relative_to(images):
                return (images.parent/'labels'/resolved.relative_to(images)).with_suffix('.txt')
        return None
    paths = {}
    for image in (Path(source), Path(target)):
        label=label_for(image)
        if label is not None:
            paths[label] = label.read_bytes() if label.exists() else None

    def rollback():
        for path, data in paths.items():
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)

    try:
        active = None
        if edit.get('corners_reviewed') and not edit.get('exclude') and label_for(target) is not None:
            active = label_for(target)
            content = label_text(edit.get('points') or proposal.get('points', []))
            active.parent.mkdir(parents=True, exist_ok=True)
            active.write_text(content, encoding='utf-8')
        for path in paths:
            if path != active:
                path.unlink(missing_ok=True)
    except Exception:
        rollback()
        raise
    return rollback
