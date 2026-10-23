"""Package explicitly reviewed vehicle identities; exclude only owner-approved pending rows."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import zipfile

from car_group_review import CarGroupStore, normalize_plate, valid_plate
import prepare_pose_dataset as prepare

V3_VERSION = 'mixed-v3-reviewed-vehicle-groups'


def plan_vehicle_v3_groups(source, snapshot, exclude_pending_count, car_val_fraction=.2, seed=42):
    rows=source['records']
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate source IDs')
    edits={r['id']:r for r in snapshot['records']}
    cars={r['id'] for r in rows if 'car' in r['source_groups']}
    if len(edits)!=len(snapshot['records']) or set(edits)!=cars:
        raise ValueError('Review snapshot does not cover exactly the car source')
    excluded={key for key,value in edits.items() if value['approved'] is not True}
    if type(exclude_pending_count) is not int or len(excluded)!=exclude_pending_count:
        raise ValueError('Pending count changed; require explicit matching exclusion count')
    edits={key:value for key,value in edits.items() if key not in excluded}
    if not edits or any(not valid_plate(r['plate']) or not r['reviewer'].strip() for r in edits.values()):
        raise ValueError('Every retained car requires a valid human-reviewed plate and reviewer')
    if not 0<car_val_fraction<1: raise ValueError('Invalid car holdout fraction')
    rows=[r for r in rows if r['id'] not in excluded]
    parent=list(range(len(rows)));owners={}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j): parent[root(j)]=root(i)
    for i,row in enumerate(rows):
        if not row.get('reviewed') or not row.get('group_reviewed') or not row.get('reviewer'):
            raise ValueError('Retained source corners/groups must already be reviewed')
        if not set(row['source_groups'])<={'car','green','bv1','cam'}:
            raise ValueError('Unknown source: classify before splitting')
        if row['id'] in edits:
            edit=edits[row['id']];keys=[('plate',normalize_plate(edit['plate']))]
            if edit['manual_group'].strip(): keys.append(('manual',edit['manual_group'].strip()))
        else:
            keys=[('group',row['group']),('old-component',row['component'])]
        keys += [('source',s) for s in row['source_groups'] if s!='car']
        keys += [('bytes',row['sha256']),('pixel',row['pixel_sha256'])]
        for key in keys:
            if key in owners: union(i,owners[key])
            owners[key]=i
        for j in range(i):
            if (row['dhash']^rows[j]['dhash']).bit_count()<=source['dhash_distance']: union(i,j)
    components=defaultdict(list)
    for i,row in enumerate(rows): components[root(i)].append(row)
    chosen={};free=[];fixed_val=0
    for key,items in sorted(components.items(),key=lambda item:min(r['id'] for r in item[1])):
        motor={s for r in items for s in r['source_groups'] if s!='car'}
        forced=set()
        if 'green' in motor: forced.add('train')
        if motor&{'bv1','cam'}: forced.add('val')
        if len(forced)>1: raise ValueError('Duplicate/group component crosses motorcycle holdout sources')
        count=sum(r['id'] in edits for r in items)
        if forced:
            chosen[key]=forced.pop()
            if chosen[key]=='val': fixed_val+=count
        else: free.append((key,count))
    random.Random(seed).shuffle(free)
    possible={0:[]}
    for key,count in free:
        for total,keys in list(possible.items()): possible.setdefault(total+count,keys+[key])
    eligible=[n for n in possible if 0<n+fixed_val<len(edits)]
    if not eligible: raise ValueError('Not enough independent car groups for both splits')
    target=min(eligible,key=lambda n:(abs(n+fixed_val-car_val_fraction*len(edits)),n))
    val=set(possible[target])
    for key,_ in free: chosen[key]='val' if key in val else 'train'
    assignments={}
    for key,items in components.items():
        component=min(r['id'] for r in items)
        for row in items:
            car=row['id'] in edits
            extra={}
            if car:
                edit=edits[row['id']]
                identity=prepare.digest(normalize_plate(edit['plate']).encode())
                extra=dict(car_identity_sha256=identity,car_group_approved=True,car_group_review_version=edit['version'])
                if edit['manual_group'].strip(): extra['manual_group_sha256']=prepare.digest(edit['manual_group'].strip().encode())
            assignments[row['id']]=dict(split=chosen[key],component=component,
                group='car-vehicle-'+extra['car_identity_sha256'] if car else row['group'],
                vehicle_type='car' if car else 'motorcycle',**extra)
    counts={split:dict(Counter(r['vehicle_type'] for r in assignments.values() if r['split']==split)) for split in ('train','val')}
    if any(not counts[s].get(k) for s in counts for k in ('car','motorcycle')):
        raise ValueError('Both vehicle types required in both splits')
    return dict(records=assignments,excluded_ids=sorted(excluded),counts=counts,
        seed=seed,car_val_fraction=car_val_fraction,car_groups=len({r['plate'] for r in edits.values()}),
        components=len(components))


@contextmanager
def lock_vehicle_v3_reviews(workspace):
    with (Path(workspace)/'group-server.lock').open('a+b') as stream:
        stream.seek(0,2)
        if not stream.tell(): stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            stream.seek(0)
            if os.name=='nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def make_vehicle_v3_notebook(package, zip_hash, manifest_hash, target):
    nb=json.loads(Path(__file__).with_name('Train_YOLO11n_Plate_Pose.ipynb').read_text(encoding='utf-8'))
    cells={c['id']:c for c in nb['cells']}
    def get(key): return ''.join(cells[key]['source'])
    def put(key,text): cells[key]['source']=text.splitlines(True)
    put('pose-00','# YOLO11 Pose v3 — train và validation có ô tô, xe máy\n'
        'Chia theo biển số/nhóm ô tô đã duyệt và thành phần ảnh gần trùng; giữ nguyên nhóm nguồn xe máy. '
        'Đã loại các ảnh chưa duyệt theo yêu cầu owner. Train mới từ pretrained; không resume v1/v2.\n')
    setup=get('drive-setup').replace('FULL_TRAIN = False','FULL_TRAIN = True')
    setup=setup.replace('nATime-YOLO11-Pose\'','nATime-YOLO11-Pose-v3-reviewed-groups\'')
    setup=setup.replace('dataset-training.zip','dataset-mixed-v3-training.zip').replace('/content/natime-pose-runs','/content/natime-pose-v3-runs')
    put('drive-setup',setup)
    put('drive-help',get('drive-help').replace('Chạy smoke trước với `FULL_TRAIN=False`.',
        'Mặc định train đầy đủ 100 epoch; đặt `FULL_TRAIN=False` để smoke 2 epoch.')+
        '\nV3 dùng Drive/cache riêng. Chỉ resume checkpoint cùng v3, RUN_ID và manifest. Không dùng best.pt v2 cho lượt đánh giá này.\n')
    data=get('pose-03').replace('dataset-training.zip','dataset-mixed-v3-training.zip')
    needle='ROOT = Path(tempfile.mkdtemp'
    assert data.count(needle)==1
    data=data.replace(needle,
        "with open(archives[0], 'rb') as package_stream:\n"
        f"    assert hashlib.file_digest(package_stream, 'sha256').hexdigest() == '{zip_hash}', 'Sai ZIP v3'\n"+needle,1)
    audit=("\nassert hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest() == '"+manifest_hash+"', 'Sai manifest v3'\n"
        f"assert manifest['dataset_version'] == '{V3_VERSION}'\n"
        "from collections import Counter\n"
        "vehicle_counts = Counter((r['split'], r['vehicle_type']) for r in manifest['records'])\n"
        "assert all(vehicle_counts[(s,k)] > 0 for s in ('train','val') for k in ('car','motorcycle'))\n"
        "identity_splits = {}\n"
        "for row in manifest['records']:\n"
        "    if row['vehicle_type'] == 'car':\n"
        "        assert row['car_group_approved'] is True and row['car_identity_sha256']\n"
        "    for key in ('car_identity_sha256','manual_group_sha256'):\n"
        "        if row.get(key):\n"
        "            token = (key, row[key])\n"
        "            assert token not in identity_splits or identity_splits[token] == row['split'], 'Vehicle group leakage'\n"
        "            identity_splits[token] = row['split']\n"
        "print('Số ảnh theo split/loại xe:', dict(vehicle_counts))\n"
        "print('Validation mới có cả ô tô và xe máy. Không so trực tiếp mAP với v2.')\n")
    data=data.replace('manifest = prepare.verify(ROOT)','manifest = prepare.verify(ROOT)'+audit,1)
    put('pose-03',data)
    helper=Path(__file__).with_name('colab_checkpoint_support.py').read_text(encoding='utf-8')
    put('pose-07',helper+'\n\n'+get('pose-07'))
    put('support-help','## Checkpoint và theo dõi\nHelper đã nhúng ở ô train. Có checkpoint Drive mỗi epoch, dashboard và TensorBoard.\n')
    put('pose-06','## Train mới bộ v3\nMặc định 100 epoch, imgsz 960, batch 4, patience 20. '
        'Bắt đầu từ pretrained yolo11n-pose.pt. Chỉ bật RESUME khi tiếp tục đúng lượt v3.\n')
    put('pose-08',get('pose-08')+'\nĐánh giá cả ô tô và xe máy trên validation v3. Bộ v3 không phải test độc lập của model đã train v2.\n')
    nb['cells']=[c for c in nb['cells'] if c['id']!='checkpoint-support']
    evaluation="""# Đánh giá riêng từng loại xe và xem tối đa 6 ảnh dự đoán mỗi loại.
from PIL import Image as PILImage
vehicle_metrics = {}
for kind in ('car', 'motorcycle'):
    samples = [ROOT/r['image'] for r in manifest['records'] if r['split']=='val' and r['vehicle_type']==kind]
    listing = ROOT/f'val-{kind}.txt'
    listing.write_text(''.join(str(p)+'\\n' for p in samples), encoding='utf-8')
    subset_yaml = ROOT/f'pose-{kind}.yaml'
    subset_yaml.write_text(f'path: {ROOT.as_posix()}\\ntrain: images/train\\nval: {listing.as_posix()}\\nkpt_shape: [4, 3]\\nflip_idx: []\\nnames:\\n  0: license_plate\\n', encoding='utf-8')
    result = trained.val(data=str(subset_yaml), imgsz=IMGSZ, device=0, plots=True)
    vehicle_metrics[kind] = dict(images=len(samples), metrics=result.results_dict)
    print(kind, vehicle_metrics[kind])
    predictions = trained.predict(source=[str(p) for p in samples[:6]], imgsz=IMGSZ, device=0,
                                  save=True, project=str(RUN/'predictions'), name=kind, exist_ok=True)
    for prediction in predictions:
        display(PILImage.fromarray(prediction.plot()[..., ::-1]))
report = RUN/'vehicle-validation-metrics.json'
report.write_text(json.dumps(vehicle_metrics, indent=2), encoding='utf-8')
atomic_copy(report, DRIVE_RUN/'exports'/report.name)
"""
    nb['cells'].append(dict(id='vehicle-eval-v3',cell_type='code',metadata={},source=evaluation.splitlines(True),execution_count=None,outputs=[]))
    for cell in nb['cells']:
        cell['metadata']={}
        if cell['cell_type']=='code': cell.update(execution_count=None,outputs=[])
    Path(target).write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def create_vehicle_v3_package(workspace, out, exclude_pending_count):
    out=Path(out).resolve();archive=Path(str(out)+'-training.zip')
    if out.exists() or archive.exists(): raise ValueError('Output already exists')
    with lock_vehicle_v3_reviews(workspace):
        store=CarGroupStore(workspace);snapshot=store.snapshot()
        print('Verifying source images and labels...',flush=True)
        source=prepare.verify(store.dataset)
        if prepare.digest((store.dataset/'manifest.json').read_bytes())!=snapshot['manifest_sha256']:
            raise ValueError('Review source changed')
        plan=plan_vehicle_v3_groups(source,snapshot,exclude_pending_count)
        out=prepare.fresh_output(out)
        receipt=dict(schema=1,owner_excluded_pending_count=exclude_pending_count,
            reason='Owner requested exclusion of the remaining unreviewed images',
            source_manifest_sha256=snapshot['manifest_sha256'],excluded_ids=plan['excluded_ids'],snapshot=snapshot)
        private=out.parent/'review-snapshot.private.json'
        if private.exists(): raise ValueError('Review receipt already exists')
        prepare.write_json(private,receipt)
        packaged=[]
        for row in source['records']:
            if row['id'] in plan['excluded_ids']: continue
            assignment=plan['records'][row['id']];split=assignment['split']
            image=f"images/{split}/{Path(row['image']).name}";label=f"labels/{split}/{row['id']}.txt"
            for old,new in ((row['image'],image),(row['label'],label)):
                (out/new).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(store.dataset/old,out/new)
            packaged.append({**row,**assignment,'image':image,'label':label,'previous_split':row['split']})
        val_count=sum(r['split']=='val' for r in packaged)
        package={**source,'records':packaged,'dataset_version':V3_VERSION,
            'parent_manifest_sha256':snapshot['manifest_sha256'],
            'review_snapshot_sha256':prepare.digest(private.read_bytes()),
            'excluded_images':source['excluded_images']+len(plan['excluded_ids']),
            'additional_excluded_images':len(plan['excluded_ids']),
            'seed':plan['seed'],'requested_val_fraction':val_count/len(packaged),'actual_val_fraction':val_count/len(packaged),
            'split_policy':dict(mode='reviewed-vehicle-groups',car_val_fraction=plan['car_val_fraction'],
                train_motorcycle_sources=['green'],val_motorcycle_sources=['bv1','cam'],
                validation_scope='car and motorcycle',counts=plan['counts'],source_group_refinement=True)}
        prepare.write_json(out/'manifest.json',package)
        for name in ('plate_pose.yaml','prepare_pose_dataset.py'): shutil.copyfile(store.dataset/name,out/name)
        print('Verifying v3 output...',flush=True);prepare.verify(out)
        # Independently re-evaluate the original review snapshot and compare every assignment.
        checked=plan_vehicle_v3_groups(source,snapshot,exclude_pending_count)
        if any(any(row[key]!=checked['records'][row['id']][key] for key in ('split','group','component')) for row in packaged):
            raise ValueError('Post-copy group audit failed')
        if store.snapshot()!=snapshot: raise ValueError('Review state changed during packaging')
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as stream:
            for path in sorted(out.rglob('*')):
                if path.is_file(): stream.write(path,path.relative_to(out).as_posix())
        zip_hash=prepare.digest(archive.read_bytes());manifest_hash=prepare.digest((out/'manifest.json').read_bytes())
        notebook=out.parent/'Train_YOLO11n_Plate_Pose_Mixed_v3.ipynb'
        make_vehicle_v3_notebook(package,zip_hash,manifest_hash,notebook)
        report=dict(dataset_version=V3_VERSION,total_images=len(packaged),counts=plan['counts'],
            additional_excluded_images=len(plan['excluded_ids']),total_excluded_images=package['excluded_images'],
            reviewed_car_plate_groups=plan['car_groups'],components=plan['components'],seed=plan['seed'],
            zip=archive.name,zip_sha256=zip_hash,manifest_sha256=manifest_hash,
            notebook=notebook.name,notebook_sha256=prepare.digest(notebook.read_bytes()),
            original_images_and_labels_unchanged=True,
            verification='Source/output hashes, four corners, exact/pixel/group/component/dHash and reviewed identity groups passed',
            training_requirement='Fresh pretrained initialization; no resume from v1/v2',
            limitation='Known reviewed groups and dHash; not proof against every unknown cross-source vehicle identity')
        prepare.write_json(out.parent/'package-report-v3.json',report)
        print(json.dumps(report,indent=2),flush=True)
        return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--exclude-pending-count',type=int,required=True)
    args=parser.parse_args()
    create_vehicle_v3_package(args.workspace,args.out,args.exclude_pending_count)
