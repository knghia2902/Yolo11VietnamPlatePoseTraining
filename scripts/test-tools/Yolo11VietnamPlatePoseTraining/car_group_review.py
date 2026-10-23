"""Local OCR suggestions, explicit vehicle-group review and leakage-safe split plans.

OCR never approves groups. Customer images, OCR strings and reviews stay outside Git.
"""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
from http.server import ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import random
import re
import threading
import time
from urllib.parse import urlsplit

from prepare_pose_dataset import fresh_output, verify
from serve_pose_review import ReviewHandler


def normalize_plate(value):
    # Do not silently replace O/0, I/1 or B/8: the reviewer must resolve them.
    return re.sub('[^A-Z0-9]', '', str(value).upper())


def valid_plate(value):
    return bool(re.fullmatch(r'[0-9]{2}[A-Z]{1,2}[0-9]{4,6}', value))


def atomic_data(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    # Windows readers/antivirus can briefly hold a destination without delete sharing.
    for attempt in range(8):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt==7: raise
            time.sleep(.05*(attempt+1))


def propose_groups(dataset, workspace):
    import cv2
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR
    dataset, workspace = Path(dataset).resolve(), Path(workspace).resolve()
    manifest = verify(dataset)
    manifest_hash = hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()
    cars = [r for r in manifest['records'] if 'car' in r['source_groups']]
    if not cars:
        raise ValueError('No car source')
    if workspace.exists():
        metadata = json.loads((workspace/'proposals.json').read_text(encoding='utf-8'))
        if metadata['manifest_sha256'] != manifest_hash:
            raise ValueError('Workspace belongs to another dataset')
    else:
        fresh_output(workspace)
        (workspace/'crops').mkdir()
        metadata = dict(schema=1, dataset=str(dataset), manifest_sha256=manifest_hash,
                        engine='rapidocr_onnxruntime==1.4.4', automatic_approval=False, records=[])
        atomic_data(workspace/'proposals.json', metadata)
    done = {r['id'] for r in metadata['records']}
    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1, text_score=0.0)
    for row in cars:
        if row['id'] in done:
            continue
        image = cv2.imdecode(np.fromfile(dataset/row['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
        h,w = image.shape[:2]
        points = np.array(row['points'], dtype=np.float32)*[w,h]
        width = max(np.linalg.norm(points[1]-points[0]), np.linalg.norm(points[2]-points[3]))
        height = max(np.linalg.norm(points[3]-points[0]), np.linalg.norm(points[2]-points[1]))
        cw = 480
        ch = int(np.clip(cw*height/max(1,width), 64, 480))
        transform = cv2.getPerspectiveTransform(points.astype(np.float32), np.array([[0,0],[cw-1,0],[cw-1,ch-1],[0,ch-1]],dtype=np.float32))
        crop = cv2.warpPerspective(image, transform, (cw,ch))
        crop = cv2.copyMakeBorder(crop,12,12,12,12,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.imencode('.jpg',crop)[1].tofile(workspace/'crops'/f"{row['id']}.jpg")
        candidates=[]
        try:
            result,_ = engine(crop,use_det=True,use_cls=False,use_rec=True)
            if result:
                ordered=sorted(result,key=lambda x:(sum(p[1] for p in x[0])/4, min(p[0] for p in x[0])))
                text=''.join(str(line[1]) for line in ordered)
                candidates.append(dict(text=normalize_plate(text), score=float(min(line[2] for line in ordered)), method='det-lines'))
            result,_ = engine(crop,use_det=False,use_cls=False,use_rec=True)
            if result:
                candidates.append(dict(text=normalize_plate(result[0][0]),score=float(result[0][1]),method='whole-crop'))
        except Exception as exc:
            # A failed OCR stays unresolved; no image or text is written to logs.
            candidates.append(dict(text='',score=0.0,method='ocr-error',error=type(exc).__name__))
        candidates.sort(key=lambda c:(valid_plate(c['text']),c['score']),reverse=True)
        best=candidates[0] if candidates else {'text':'','score':0.0}
        metadata['records'].append(dict(id=row['id'],crop=f"crops/{row['id']}.jpg",
            plate=best['text'],score=best['score'],candidates=candidates,approved=False))
        atomic_data(workspace/'proposals.json',metadata)
        if len(metadata['records'])%50==0:
            print(f"OCR {len(metadata['records'])}/{len(cars)}",flush=True)
    summary=dict(images=len(cars),proposed_valid=sum(valid_plate(r['plate']) for r in metadata['records']),
                 pending=len(cars),approved=0,automatic_approval=False)
    atomic_data(workspace/'ocr-summary.json',summary)
    print(json.dumps(summary),flush=True)
    return metadata


class CarGroupStore:
    def __init__(self, workspace):
        self.root=Path(workspace).resolve()
        self.metadata=json.loads((self.root/'proposals.json').read_text(encoding='utf-8'))
        self.dataset=Path(self.metadata['dataset']).resolve()
        if hashlib.sha256((self.dataset/'manifest.json').read_bytes()).hexdigest()!=self.metadata['manifest_sha256']:
            raise ValueError('Source manifest changed')
        source=json.loads((self.dataset/'manifest.json').read_text(encoding='utf-8'))
        cars={r['id']:r for r in source['records'] if 'car' in r['source_groups']}
        self.rows={r['id']:r for r in self.metadata['records']}
        if set(cars)!=set(self.rows):
            raise ValueError('OCR must cover every car before review starts')
        self.assets={}
        for image_id,row in cars.items():
            image=(self.dataset/row['image']).resolve()
            crop=(self.root/self.rows[image_id]['crop']).resolve()
            if self.dataset not in image.parents or (self.root/'crops') not in crop.parents:
                raise ValueError('Unsafe image path')
            self.assets['/image/'+image_id+'.jpg']=image
            self.assets['/crop/'+image_id+'.jpg']=crop
        self.state_dir=self.root/'group-reviews'
        self.state_dir.mkdir(exist_ok=True)
        self.lock=threading.Lock()
        self.edits={}
        for path in self.state_dir.glob('*.json'):
            saved=json.loads(path.read_text(encoding='utf-8'))
            if saved['manifest_sha256']!=self.metadata['manifest_sha256'] or saved['id'] not in self.rows or path.stem!=saved['id']:
                raise ValueError('Saved group reviews belong to another dataset')
            self.edits[saved['id']]=saved

    def snapshot(self):
        with self.lock:
            rows=[]
            for image_id,row in self.rows.items():
                edit=self.edits.get(image_id,{})
                rows.append({**row,'plate':edit.get('plate',row['plate']),
                    'manual_group':edit.get('manual_group',''),'reviewer':edit.get('reviewer',''),
                    'approved':edit.get('approved',False),'version':edit.get('version',0),
                    'crop':'/crop/'+image_id+'.jpg','image':'/image/'+image_id+'.jpg'})
            return dict(schema=1,manifest_sha256=self.metadata['manifest_sha256'],records=rows)

    def save(self, request):
        with self.lock:
            image_id=request.get('id')
            if image_id not in self.rows or type(request.get('version')) is not int:
                raise ValueError('Unknown image or invalid revision')
            previous=self.edits.get(image_id,{})
            if request['version']!=previous.get('version',0):
                raise FileExistsError('Ảnh đã được sửa ở cửa sổ khác. Tải lại trang trước khi tiếp tục.')
            if set(request)!={'id','version','plate','manual_group','reviewer','approved'}:
                raise ValueError('Invalid review fields')
            if not all(isinstance(request[k],str) for k in ('plate','manual_group','reviewer')) or type(request['approved']) is not bool:
                raise ValueError('Invalid review types')
            plate=normalize_plate(request['plate']); manual=request['manual_group'].strip(); reviewer=request['reviewer'].strip()
            if len(plate)>16 or len(manual)>100 or len(reviewer)>100:
                raise ValueError('Review field too long')
            if request['approved'] and (not valid_plate(plate) or not reviewer):
                raise ValueError('Cần biển số đúng dạng và tên người duyệt. Ảnh chưa đọc rõ chỉ Lưu, chưa Duyệt.')
            state=dict(id=image_id,manifest_sha256=self.metadata['manifest_sha256'],version=request['version']+1,
                plate=plate,manual_group=manual,reviewer=reviewer,approved=request['approved'])
            # Keep previous revisions for recovery and audit.
            history=self.root/'group-history';history.mkdir(exist_ok=True)
            if previous:
                atomic_data(history/f"{image_id}-{previous['version']}.json",previous)
            atomic_data(self.state_dir/(image_id+'.json'),state)
            self.edits[image_id]=state
            return dict(version=state['version'])


class CarGroupHandler(ReviewHandler):
    def do_GET(self):
        if not self.authenticated(): return
        path=urlsplit(self.path).path
        if path=='/':
            self.respond(200,Path(__file__).with_name('review_car_groups.html').read_bytes(),'text/html; charset=utf-8')
        elif path in ('/api/state','/api/export'):
            headers={'Content-Disposition':'attachment; filename="car-group-reviews.json"'} if path.endswith('export') else None
            self.respond(200,self.server.store.snapshot(),headers=headers)
        elif path in self.server.store.assets:
            self.respond(200,self.server.store.assets[path].read_bytes(),'image/jpeg')
        else: self.respond(404,{'error':'Not found'})

    def do_POST(self):
        if not self.authenticated(): return
        if self.path!='/api/review':
            self.respond(404,{'error':'Not found'});return
        if (self.headers.get('X-Review-Request')!='1' or self.headers.get('Content-Type','').split(';')[0]!='application/json' or
            self.headers.get('Origin','http://'+self.headers.get('Host',''))!='http://'+self.headers.get('Host','')):
            self.respond(403,{'error':'Invalid request origin'});return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=8192: raise ValueError('Invalid request size')
            request=json.loads(self.rfile.read(length))
            if not isinstance(request,dict): raise ValueError('Invalid request')
            self.respond(200,self.server.store.save(request))
        except FileExistsError as exc: self.respond(409,{'error':str(exc)})
        except (ValueError,KeyError,TypeError) as exc: self.respond(400,{'error':str(exc)})


class CarGroupServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,workspace,credentials,networks):
        self.state_lock=(Path(workspace)/'group-server.lock').open('a+b')
        self.state_lock.seek(0,2)
        if self.state_lock.tell()==0:
            self.state_lock.write(b'0');self.state_lock.flush()
        self.state_lock.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.state_lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.state_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.store=CarGroupStore(workspace)
            self.networks=[ipaddress.ip_network(n) for n in networks]
            self.expected_auth='Basic '+base64.b64encode((credentials['username']+':'+credentials['password']).encode()).decode()
            super().__init__(address,CarGroupHandler)
        except Exception:
            self.state_lock.close();raise
    def server_close(self):
        super().server_close();self.state_lock.close()


def reviewed_split_plan(workspace, car_val_fraction=.2, seed=42):
    store=CarGroupStore(workspace)
    state=store.snapshot()
    edits={r['id']:r for r in state['records']}
    if any(not r['approved'] or not valid_plate(r['plate']) or not r['reviewer'] for r in edits.values()):
        raise ValueError('Phải duyệt đủ mọi ảnh ô tô trước khi chia train/val')
    if not 0<car_val_fraction<1: raise ValueError('Invalid car validation fraction')
    source=json.loads((store.dataset/'manifest.json').read_text(encoding='utf-8'))
    rows=source['records'];parent=list(range(len(rows)));owners={}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j): parent[root(j)]=root(i)
    for i,row in enumerate(rows):
        if row['id'] in edits:
            edit=edits[row['id']]
            # Identical plates always stay together even if manual group differs.
            keys=[('plate',normalize_plate(edit['plate']))]
            if edit['manual_group']: keys.append(('manual',edit['manual_group']))
        else:
            keys=[('source',s) for s in row['source_groups']]+[('old-group',row['group']),('old-component',row['component'])]
        keys += [('pixel',row['pixel_sha256']),('bytes',row['sha256'])]
        # Cross-source aliases must retain links to non-car sources.
        keys += [('source',s) for s in row['source_groups'] if s!='car']
        for key in keys:
            if key in owners: union(i,owners[key])
            owners[key]=i
        for j in range(i):
            if (row['dhash']^rows[j]['dhash']).bit_count()<=source['dhash_distance']: union(i,j)
    components=defaultdict(list)
    for i,row in enumerate(rows): components[root(i)].append(row)
    chosen={};free=[];fixed_car_val=0
    for key,items in sorted(components.items(),key=lambda pair:min(r['id'] for r in pair[1])):
        motor={s for r in items for s in r['source_groups'] if s!='car'}
        forced=set()
        if 'green' in motor: forced.add('train')
        if motor & {'bv1','cam'}: forced.add('val')
        if len(forced)>1: raise ValueError('Near-duplicate edge joins both motorcycle source splits')
        car_count=sum(r['id'] in edits for r in items)
        if forced:
            chosen[key]=forced.pop()
            if chosen[key]=='val': fixed_car_val+=car_count
        else: free.append((key,car_count))
    random.Random(seed).shuffle(free)
    possible={0:[]}
    for key,count in free:
        for total,keys in list(possible.items()): possible.setdefault(total+count,keys+[key])
    eligible=[n for n in possible if 0<n+fixed_car_val<len(edits)]
    if not eligible: raise ValueError('Không đủ nhóm ô tô độc lập cho cả train và val')
    total=min(eligible,key=lambda n:(abs(n+fixed_car_val-car_val_fraction*len(edits)),n))
    val=set(possible[total])
    for key,_ in free: chosen[key]='val' if key in val else 'train'
    assignments=[]
    for key,items in components.items():
        component=min(r['id'] for r in items)
        for row in items:
            is_car=row['id'] in edits
            group='car-vehicle-'+hashlib.sha256(edits[row['id']]['plate'].encode()).hexdigest()[:24] if is_car else row['group']
            assignments.append(dict(id=row['id'],split=chosen[key],component=component,group=group,vehicle_type='car' if is_car else 'motorcycle'))
    counts=Counter((r['split'],r['vehicle_type']) for r in assignments)
    if any(not counts[(split,kind)] for split in ('train','val') for kind in ('car','motorcycle')):
        raise ValueError('Both vehicle types required in both splits')
    return dict(schema=1,ready_for_packaging=True,parent_manifest_sha256=state['manifest_sha256'],
        review_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest(),seed=seed,
        dhash_distance=source['dhash_distance'],requested_car_val_fraction=car_val_fraction,
        counts={s:dict(Counter(r['vehicle_type'] for r in assignments if r['split']==s)) for s in ('train','val')},
        policy='Reviewed car plate/manual groups plus exact/pixel/dHash components; whole motorcycle sources held out',
        records=assignments)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('ocr');p.add_argument('--dataset',required=True);p.add_argument('--workspace',required=True)
    p=commands.add_parser('serve');p.add_argument('--workspace',required=True);p.add_argument('--credentials',required=True)
    p.add_argument('--bind',default='127.0.0.1');p.add_argument('--port',type=int,default=8766)
    p.add_argument('--allow-network',action='append',default=['127.0.0.0/8'])
    p=commands.add_parser('plan');p.add_argument('--workspace',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='ocr': propose_groups(args.dataset,args.workspace)
    elif args.command=='plan':
        target=Path(args.out)
        if target.exists(): raise ValueError('Output already exists')
        atomic_data(target,reviewed_split_plan(args.workspace))
        print('Reviewed split plan saved; package only with matching reviewed source.')
    else:
        credentials=json.loads(Path(args.credentials).read_text(encoding='utf-8'))
        server=CarGroupServer((args.bind,args.port),args.workspace,credentials,args.allow_network)
        print(f'Car group review: http://{args.bind}:{args.port}',flush=True)
        try: server.serve_forever()
        finally: server.server_close()
