"""Repackage reviewed data by whole source/duplicate components, never invent vehicle IDs.

Customer images/manifests and release outputs must remain outside the checkout.
The original package and its reviewed labels are read-only inputs.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import prepare_pose_dataset as prepare

VERSION = "mixed-v2-source-holdout"


def plan_source_holdout(records, train_sources, val_sources, distance):
    train_sources, val_sources = set(train_sources), set(val_sources)
    if not train_sources or not val_sources or train_sources & val_sources:
        raise ValueError("Train and validation sources must be nonempty and disjoint")
    if not 0 <= distance <= 16:
        raise ValueError("Invalid dHash threshold")
    if len({r['id'] for r in records}) != len(records):
        raise ValueError("Duplicate image IDs")
    known = {s for r in records for s in r['source_groups']}
    if known != train_sources | val_sources:
        raise ValueError("Source selection must cover all and only known sources")
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[root(j)] = root(i)

    owners = {}
    for i, row in enumerate(records):
        if not row.get('reviewed') or not row.get('group_reviewed') or not row.get('group'):
            raise ValueError("Unreviewed or missing leakage group")
        # Retain source aliases, manually reviewed groups and old components.
        keys = [('source', s) for s in row['source_groups']]
        keys += [('group', row['group']), ('component', row['component'])]
        keys += [('sha256', row['sha256']), ('pixel', row['pixel_sha256'])]
        for key in keys:
            if key in owners:
                union(i, owners[key])
            owners[key] = i
        for j in range(i):
            if (row['dhash'] ^ records[j]['dhash']).bit_count() <= distance:
                union(i, j)
    components = defaultdict(list)
    for i, row in enumerate(records):
        components[root(i)].append(row)
    assignment = {}
    for rows in components.values():
        sources = {s for r in rows for s in r['source_groups']}
        if sources & train_sources and sources & val_sources:
            raise ValueError("Requested split cuts a source/group/duplicate component")
        split = 'train' if sources <= train_sources else 'val'
        component = min(r['id'] for r in rows)
        for row in rows:
            assignment[row['id']] = (split, component)
    if {s for s, _ in assignment.values()} != {'train', 'val'}:
        raise ValueError("Empty split")
    return assignment


def build_mixed_notebook(package, zip_sha256, target, manifest_sha256=None):
    source = Path(__file__).with_name('Train_YOLO11n_Plate_Pose.ipynb')
    nb = json.loads(source.read_text(encoding='utf-8'))
    cells = {c['id']: c for c in nb['cells']}
    def put(key, text):
        cells[key]['source'] = text.splitlines(True)
    def get(key):
        return ''.join(cells[key]['source'])

    put('pose-00', '# YOLO11 Pose v2: train có ô tô và xe máy\n'
        'Bộ mới giữ nguyên nhóm nguồn: car + green trong train; bv1/cam trong validation. '
        'Validation chỉ đánh giá xe máy, chưa đo khả năng tổng quát ô tô độc lập. '
        'Chạy từ trên xuống, bật GPU. Không resume checkpoint bộ v1.\n')
    setup = get('drive-setup').replace('FULL_TRAIN = False', 'FULL_TRAIN = True')
    setup = setup.replace("nATime-YOLO11-Pose'", "nATime-YOLO11-Pose-v2-sourceholdout'")
    setup = setup.replace('datasets/dataset-training.zip', 'datasets/dataset-mixed-v2-training.zip')
    put('drive-setup', setup)
    put('drive-help', get('drive-help').replace('Chạy smoke trước với `FULL_TRAIN=False`.',
        'Mặc định train đầy đủ 100 epoch; có thể đặt `FULL_TRAIN=False` để thử 2 epoch.') +
        '\nBộ v2 dùng thư mục Drive/cache riêng. Giữ RUN_ID mới; không dùng RUN_ID/checkpoint v1.\n')
    data = get('pose-03')
    data = data.replace("'dataset-training.zip'", "'dataset-mixed-v2-training.zip'")
    data = data.replace('ROOT = Path(tempfile.mkdtemp',
        f"with open(archives[0], 'rb') as package_stream:\n"
        f"    assert hashlib.file_digest(package_stream, 'sha256').hexdigest() == '{zip_sha256}', 'Sai ZIP v2; chọn dataset-mixed-v2-training.zip'\n"
        'ROOT = Path(tempfile.mkdtemp', 1)
    # Pin bytes from the packaged file, including the writer's newline convention.
    manifest_hash = manifest_sha256 or prepare.digest(json.dumps(package, ensure_ascii=False, indent=2).encode())
    data = data.replace('manifest = prepare.verify(ROOT)',
        "manifest = prepare.verify(ROOT)\n"
        f"assert hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest() == '{manifest_hash}', 'Sai manifest v2'\n"
        f"assert manifest['dataset_version'] == '{VERSION}'\n"
        "from collections import Counter\n"
        "print('Số ảnh theo split/nguồn:', Counter((r['split'], r['group']) for r in manifest['records']))\n"
        "print('Lưu ý: validation chỉ có xe máy; chưa có test ô tô độc lập.')")
    put('pose-03', data)
    # Embed the helper at the point of use: skipping a separate support cell
    # must not produce the missing-file_sha256 error observed in the v1 run.
    helper = Path(__file__).with_name('colab_checkpoint_support.py').read_text(encoding='utf-8')
    put('pose-07', helper + '\n\n' + get('pose-07'))
    put('support-help', '## Checkpoint và bảng hoạt động\nMã hỗ trợ đã nhúng trực tiếp vào ô train; không cần chạy ô helper riêng. Checkpoint vẫn lưu Drive mỗi epoch.\n')
    put('pose-06', '## Train đầy đủ\nMặc định 100 epoch, ảnh 960, batch 4, early stopping 20 epoch. '
        'Bắt đầu từ pretrained YOLO11n Pose; bộ dữ liệu đã đổi nên không resume v1. '
        'TensorBoard và checkpoint Drive vẫn hoạt động như trước.\n')
    put('pose-08', get('pose-08') + '\nValidation v2 chỉ có xe máy. Chỉ số này không đại diện chất lượng ô tô; cần test ô tô mới độc lập.\n')
    nb['cells'] = [c for c in nb['cells'] if c['id'] != 'checkpoint-support']
    for cell in nb['cells']:
        cell['metadata'] = {}
        if cell['cell_type'] == 'code':
            cell.update(execution_count=None, outputs=[])
    Path(target).write_text(json.dumps(nb, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def package_source_holdout(source, out, train_sources=('car', 'green'), val_sources=('bv1', 'cam')):
    source, out = Path(source).resolve(), Path(out).resolve()
    archive_path = Path(str(out)+'-training.zip')
    if archive_path.exists():
        raise ValueError("Training archive already exists")
    original = prepare.verify(source)
    rows = original['records']
    assignment = plan_source_holdout(rows, train_sources, val_sources, original['dhash_distance'])
    out = prepare.fresh_output(out)
    packaged = []
    for row in rows:
        split, component = assignment[row['id']]
        image = f"images/{split}/{Path(row['image']).name}"
        label = f"labels/{split}/{row['id']}.txt"
        for old, new in ((row['image'], image), (row['label'], label)):
            (out/new).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/old, out/new)
        packaged.append({**row, 'previous_split': row['split'], 'image': image,
                         'label': label, 'split': split, 'component': component})
    counts = Counter(r['split'] for r in packaged)
    package = {**original, 'records': packaged, 'dataset_version': VERSION,
               'parent_manifest_sha256': prepare.digest((source/'manifest.json').read_bytes()),
               'split_policy': {'mode': 'whole-source-held-out',
                                'train_sources': sorted(train_sources), 'val_sources': sorted(val_sources),
                                'validation_scope': 'motorcycle-only; no independent car accuracy claim',
                                'source_group_refinement': False},
               'requested_val_fraction': counts['val']/len(rows),
               'actual_val_fraction': counts['val']/len(rows)}
    prepare.write_json(out/'manifest.json', package)
    for name in ('plate_pose.yaml', 'prepare_pose_dataset.py'):
        shutil.copyfile(source/name, out/name)
    prepare.verify(out)
    # Re-evaluate grouping from the output rather than trusting stored component IDs.
    checked = plan_source_holdout(packaged, train_sources, val_sources, package['dhash_distance'])
    if any(checked[r['id']][0] != r['split'] for r in packaged):
        raise ValueError("Split audit failed")
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(out).as_posix())
    sha = prepare.digest(archive_path.read_bytes())
    notebook = out.parent/'Train_YOLO11n_Plate_Pose_Mixed_v2.ipynb'
    build_mixed_notebook(package, sha, notebook, prepare.digest((out/'manifest.json').read_bytes()))
    report = {'dataset_version': VERSION, 'counts': dict(counts),
              'sources': {s: dict(Counter(r['group'] for r in packaged if r['split']==s)) for s in counts},
              'components': len({r['component'] for r in packaged}),
              'zip': archive_path.name, 'zip_sha256': sha,
              'notebook': notebook.name, 'notebook_sha256': prepare.digest(notebook.read_bytes()),
              'original_labels_and_images_unchanged': True,
              'verification': 'exact/pixel/group/component/dHash and whole source checks passed',
              'validation_limit': 'Motorcycles only; car generalization requires new independent test data'}
    prepare.write_json(out.parent/'package-report.json', report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    package_source_holdout(args.source, args.out)
