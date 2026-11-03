"""Register trusted, locally trained weights as an immutable test model version."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from datetime import date


def register(weights, root, version, name, best_epoch):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', version):
        raise ValueError('Invalid model version')
    source = Path(weights).resolve()
    if not source.is_file() or source.suffix != '.pt':
        raise ValueError('Expected trusted .pt weights from a completed training run')
    root = Path(root).resolve()
    target = root / version
    root.mkdir(parents=True, exist_ok=True)
    target.mkdir()  # An existing model version must never be overwritten.
    try:
        shutil.copyfile(source, target/'best.pt')
        checksum = hashlib.sha256((target/'best.pt').read_bytes()).hexdigest()
        (target/'model.json').write_text(json.dumps(dict(name=name, sha256=checksum,
            trained_at=str(date.today()), best_epoch=best_epoch), ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        shutil.rmtree(target)
        raise
    return target


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True)
    p.add_argument('--root', default='D:/ANPR-Training-Data/model-yolo-point')
    p.add_argument('--version', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--best-epoch', type=int, required=True)
    a=p.parse_args()
    print(register(a.weights,a.root,a.version,a.name,a.best_epoch))
