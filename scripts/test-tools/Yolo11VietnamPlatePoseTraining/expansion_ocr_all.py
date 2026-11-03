"""Complete resumable local OCR, including uncertain regions, for expansion triage."""
import hashlib
import json
from pathlib import Path
import time

from expansion_triage import OCR_SCHEMA, triage_plate, triage_valid
from prepare_pose_dataset import valid_points


def expansion_ocr_regions(workspace,row):
    import cv2
    import numpy as np
    from PIL import Image,ImageOps
    workspace=Path(workspace)
    p,e=json.loads(row['proposal']),json.loads(row['edit'])
    boxes=p.get('source_detection_boxes',[])
    regions=[];full_frame=False;truncated=False
    points=e.get('points') if e.get('corners_reviewed') else p.get('points',[])
    if len(boxes)>1:
        for cx,cy,w,h in boxes:
            regions.append([[max(0,cx-w*.52),max(0,cy-h*.52)],
                            [min(1,cx+w*.52),max(0,cy-h*.52)],
                            [min(1,cx+w*.52),min(1,cy+h*.52)],
                            [max(0,cx-w*.52),min(1,cy+h*.52)]])
    elif row['stage']=='multiple' and p.get('detections'):
        regions=[d['points'] for d in p['detections']]
    elif points:
        regions=[points]
    elif len(boxes)==1:
        cx,cy,w,h=boxes[0]
        regions=[[[max(0,cx-w*.52),max(0,cy-h*.52)],
                  [min(1,cx+w*.52),max(0,cy-h*.52)],
                  [min(1,cx+w*.52),min(1,cy+h*.52)],
                  [max(0,cx-w*.52),min(1,cy+h*.52)]]]
    if len(regions)>8: regions=regions[:8];truncated=True
    crop=workspace/'crops'/(row['id']+'.jpg')
    if len(regions)==1 and points==p.get('points') and crop.exists():
        return [(str(crop),False)],False,truncated
    if not regions:
        preview=workspace/'previews'/(row['id']+'.jpg')
        if preview.exists(): return [(str(preview),True)],True,False
        full_frame=True
    data=Path(row['path']).read_bytes()
    if row['sha'] and hashlib.sha256(data).hexdigest()!=row['sha']:
        raise ValueError('Source image differs from its recorded fingerprint')
    image=ImageOps.exif_transpose(Image.open(__import__('io').BytesIO(data))).convert('RGB')
    array=np.array(image)[:,:,::-1].copy();height,width=array.shape[:2]
    if full_frame:
        if max(height,width)>1280:
            array=cv2.resize(array,(round(width*1280/max(width,height)),round(height*1280/max(width,height))))
        return [(array,True)],True,False
    outputs=[]
    for points in regions:
        valid_points(points)
        pts=np.array(points,dtype=np.float32)*[width,height]
        w=max(np.linalg.norm(pts[1]-pts[0]),np.linalg.norm(pts[2]-pts[3]))
        h=max(np.linalg.norm(pts[3]-pts[0]),np.linalg.norm(pts[2]-pts[1]))
        cw,ch=480,int(np.clip(480*h/max(w,1),64,480))
        transform=cv2.getPerspectiveTransform(pts.astype('float32'),
            np.array([[0,0],[cw-1,0],[cw-1,ch-1],[0,ch-1]],dtype='float32'))
        outputs.append((cv2.warpPerspective(array,transform,(cw,ch)),False))
    return outputs,False,truncated


def expansion_recognize_all(engine,regions):
    candidates=[];errors=[]
    for region,(image,full_frame) in enumerate(regions):
        try:
            results,_=engine(image,use_det=True,use_cls=False,use_rec=True)
            lines=sorted(results or [],key=lambda x:(sum(p[1] for p in x[0])/4,min(p[0] for p in x[0])))
            if lines:
                candidates.append(dict(plate=triage_plate(''.join(str(x[1]) for x in lines)),
                    confidence=min(float(x[2]) for x in lines),method='det-lines',region=region))
            # Full-scene text is never automatically trusted as a plate. Preserve
            # individual/adjacent reads to detect possible validation matches too.
            if full_frame:
                for i,line in enumerate(lines):
                    for group in (lines[i:i+1],lines[i:i+2]):
                        text=triage_plate(''.join(str(x[1]) for x in group))
                        if triage_valid(text): candidates.append(dict(plate=text,
                            confidence=min(float(x[2]) for x in group),method='full-frame-lines',region=region))
            else:
                split=expansion_split_line_read(engine,image,lines)
                if split is not None:
                    candidates.append(dict(plate=split[0],confidence=split[1],method='split-lines',region=region))
                else:
                    whole,_=engine(image,use_det=False,use_cls=False,use_rec=True)
                    if whole: candidates.append(dict(plate=triage_plate(whole[0][0]),
                        confidence=float(whole[0][1]),method='whole-crop',region=region))
        except Exception as exc:
            errors.append(type(exc).__name__)
    candidates.sort(key=lambda c:(triage_valid(c['plate']),c['confidence']),reverse=True)
    best=candidates[0] if candidates else dict(plate='',confidence=0.0,region=0)
    return dict(schema=OCR_SCHEMA,layout_aware=True,plate=best['plate'],confidence=best['confidence'],region=best['region'],
        valid_format=triage_valid(best['plate']),candidates=candidates,
        error=','.join(sorted(set(errors))) if errors else None,engine='RapidOCR 1.4.4; local two-pass')


def expansion_split_line_read(engine,image,lines):
    import cv2
    import numpy as np
    if isinstance(image,str):
        path=Path(image)
        if not path.is_file(): return None
        array=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    else: array=image
    if array is None or array.shape[0]/max(1,array.shape[1])<.42: return None
    height=array.shape[0];middle=height//2
    centers=sorted(sum(p[1] for p in line[0])/4 for line in lines)
    if len(centers)>1:
        gap,index=max((centers[i+1]-centers[i],i) for i in range(len(centers)-1))
        if gap>height*.15: middle=round((centers[index]+centers[index+1])/2)
    middle=max(round(height*.25),min(round(height*.75),middle));margin=round(height*.025)
    reads=[]
    for part in (array[:middle+margin],array[middle-margin:]):
        result,_=engine(part,use_det=False,use_cls=False,use_rec=True)
        if not result: return ('',0.0)
        reads.append((triage_plate(result[0][0]),float(result[0][1])))
    return ''.join(r[0] for r in reads),min(r[1] for r in reads)


def expansion_ocr_all(workspace,follow=False,limit=0,engine=None):
    from curate_acs_training_pool import expansion_db,expansion_meta
    if engine is None:
        from rapidocr_onnxruntime import RapidOCR
        engine=RapidOCR(intra_op_num_threads=2,inter_op_num_threads=1,text_score=0.0)
    count=0
    with expansion_db(workspace) as db: expansion_meta(db,'ocr_all_complete',False)
    while not limit or count<limit:
        with expansion_db(workspace) as db:
            rows=db.execute("SELECT * FROM images WHERE stage NOT IN ('pending','duplicate','error') "
                "AND (coalesce(json_extract(proposal,'$.ocr.schema'),0)<>? OR coalesce(json_extract(proposal,'$.ocr.layout_aware'),0)<>1) ORDER BY "
                "CASE kind WHEN 'validation' THEN 0 WHEN 'roboflow' THEN 1 WHEN 'acs' THEN 2 ELSE 3 END,rowid LIMIT 50",(OCR_SCHEMA,)).fetchall()
            if not rows:
                if expansion_meta(db,'processing_complete') and not db.execute("SELECT 1 FROM images WHERE stage='pending' LIMIT 1").fetchone():
                    expansion_meta(db,'ocr_all_complete',True);break
                if not follow: break
                time.sleep(3);continue
            for row in rows:
                if limit and count>=limit: break
                try:
                    regions,full,truncated=expansion_ocr_regions(workspace,row)
                    result=expansion_recognize_all(engine,regions)
                    result.update(full_frame=full,regions_truncated=truncated,region_count=len(regions))
                except Exception as exc:
                    result=dict(schema=OCR_SCHEMA,layout_aware=True,plate='',confidence=0,valid_format=False,
                                candidates=[],error=type(exc).__name__)
                # Never write an old proposal over work committed by another worker.
                db.execute('BEGIN IMMEDIATE')
                latest=db.execute('SELECT proposal FROM images WHERE id=?',(row['id'],)).fetchone()
                p=json.loads(latest[0]);p['ocr']=result
                db.execute('UPDATE images SET proposal=?,ocr_done=1 WHERE id=?',(json.dumps(p),row['id']))
                db.commit();count+=1
                if count%50==0: print(json.dumps(dict(ocr_all_this_run=count)),flush=True)
            expansion_meta(db,'ocr_all_progress',dict(processed_this_run=count,last_update=time.time()))
    return count
