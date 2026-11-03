"""Local OCR comparison queues; approval is a separate owner-authorized action."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time

OCR_SCHEMA = 2
MIN_CONFIDENCE = .90


def triage_plate(value):
    return re.sub('[^A-Z0-9]', '', str(value).upper())


def triage_valid(value):
    return bool(re.fullmatch(r'[0-9]{2}[A-Z]{1,2}[0-9]{4,6}', value))


def triage_score(value):
    try: number=float(value)
    except (TypeError,ValueError): return 0.0
    return max(0.0,min(1.0,number)) if math.isfinite(number) else 0.0


def triage_candidates(proposal, edit):
    ocr=proposal.get('ocr') or {}
    values=[]
    if ocr.get('plate'):
        values.append(dict(plate=triage_plate(ocr['plate']),confidence=triage_score(ocr.get('confidence',0)),region=ocr.get('region',0)))
    for item in ocr.get('candidates',[]) + ocr.get('comparison_candidates',[]):
        text=triage_plate(item.get('plate',item.get('text','')))
        if text: values.append(dict(plate=text,confidence=triage_score(item.get('confidence',item.get('score',0))),region=item.get('region',0)))
    if edit.get('identity_reviewed') and triage_valid(edit.get('plate','')):
        return [dict(plate=edit['plate'],confidence=1.0,region=0,human=True)]
    return values


def triage_validation_index(rows):
    exact=defaultdict(list); deleted=defaultdict(set)
    by_id={};hashes={};chunks=defaultdict(set)
    coverage=Counter()
    for row in rows:
        if row['kind']!='validation': continue
        p,e=json.loads(row['proposal']),json.loads(row['edit'])
        candidates=triage_candidates(p,e)
        valid={c['plate']:c for c in sorted(candidates,key=lambda c:c['confidence']) if triage_valid(c['plate'])}
        coverage['total']+=1
        if e.get('identity_reviewed'):
            coverage['auto_confirmed' if e.get('identity_method')=='auto-ocr-v1' else 'human_confirmed']+=1
        if valid: coverage['has_format_valid_candidate']+=1
        else: coverage['unresolved_ocr']+=1
        if p.get('ocr',{}).get('schema')==OCR_SCHEMA: coverage['ocr_v2_done']+=1
        if p.get('ocr',{}).get('layout_aware'): coverage['layout_aware_done']+=1
        primary=e.get('plate','') if e.get('identity_reviewed') else p.get('ocr',{}).get('plate','')
        by_id[row['id']]=dict(id=row['id'],reference_plate=primary,confidence=max((c['confidence'] for c in candidates),default=0),human=bool(e.get('identity_reviewed')))
        dhash=row['dhash'] if 'dhash' in row.keys() else None
        if dhash is not None:
            number=int(dhash,16) if isinstance(dhash,str) else int(dhash)
            hashes[row['id']]=number
            for part in range(5): chunks[(part,(number>>(part*13))&8191)].add(row['id'])
        for plate,c in valid.items():
            exact[plate].append(dict(id=row['id'],confidence=c['confidence'],human=bool(c.get('human'))))
            for i in range(len(plate)): deleted[plate[:i]+plate[i+1:]].add(plate)
    return dict(exact=dict(exact),deleted=deleted,coverage=dict(coverage),by_id=by_id,hashes=hashes,chunks=chunks)


def triage_distance_one(a,b):
    if abs(len(a)-len(b))>1: return False
    if len(a)==len(b): return sum(x!=y for x,y in zip(a,b))<=1
    if len(a)>len(b): a,b=b,a
    i=j=errors=0
    while i<len(a) and j<len(b):
        if a[i]==b[j]: i+=1;j+=1
        else:
            errors+=1;j+=1
            if errors>1: return False
    return True


def triage_matches(candidates,index):
    found={}
    for c in candidates:
        plate=c['plate']
        if len(plate)<6 or len(plate)>11: continue
        possible={plate} if plate in index['exact'] else set()
        possible.update(index['deleted'].get(plate,()))
        for i in range(len(plate)):
            key=plate[:i]+plate[i+1:]
            possible.update(index['deleted'].get(key,()))
            if key in index['exact']: possible.add(key)
        for reference in possible:
            if not triage_distance_one(plate,reference): continue
            for ref in index['exact'][reference]:
                method='exact-plate' if plate==reference else 'one-character'
                key=(ref['id'],method)
                found[key]=dict(**ref,method=method,candidate_plate=plate,reference_plate=reference)
    return sorted(found.values(),key=lambda m:(m['method']!='exact-plate',m['id'],m['reference_plate'],m['candidate_plate']))


def triage_match_digest(matches):
    values=sorted((m['id'],m['method'],m['candidate_plate'],m['reference_plate'],m['human']) for m in matches)
    return hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()


def triage_image_matches(row,proposal,index,plate):
    ids=set(proposal.get('near',[]))&index['by_id'].keys()
    dhash=row['dhash'] if 'dhash' in row.keys() else None
    if dhash is not None:
        number=int(dhash,16) if isinstance(dhash,str) else int(dhash)
        possible=set()
        for part in range(5): possible.update(index['chunks'].get((part,(number>>(part*13))&8191),()))
        ids.update(image_id for image_id in possible if (number^index['hashes'][image_id]).bit_count()<=4)
    return [dict(**index['by_id'][image_id],method='near-image',candidate_plate=plate) for image_id in sorted(ids)]


def triage_record(row,index):
    p,e=json.loads(row['proposal']),json.loads(row['edit'])
    ocr=p.get('ocr') or {};reasons=[];matches=[]
    category='ocr_pending'
    if e.get('exclude'): category='excluded'
    elif row['stage']=='pending': category='pending'
    elif row['stage']=='duplicate': category='duplicate'
    elif row['stage']=='error': category='file_error'
    else:
        candidates=triage_candidates(p,e)
        best=triage_plate(e['plate'] if e.get('identity_reviewed') else ocr.get('plate',''))
        human=bool(e.get('identity_reviewed'))
        if row['kind']!='validation':
            matches=triage_matches(candidates,index)+triage_image_matches(row,p,index,best)
        resolved=(bool(matches) and human and e.get('validation_comparison')=='different'
                  and e.get('validation_comparison_digest')==triage_match_digest(matches)
                  and all(m['method'] in ('one-character','near-image') for m in matches))
        if e.get('validation_comparison')=='same' and human:
            category='confirmed_overlap';reasons.append('Người duyệt xác nhận cùng xe với validation')
        elif matches and not resolved:
            confirmed=human and any(m['human'] and m['method']=='exact-plate' for m in matches)
            category='confirmed_overlap' if confirmed else 'suspected_overlap'
            reasons.append('Ảnh hoặc biển số gần giống validation; cần so ảnh tham chiếu')
        elif (ocr.get('schema')!=OCR_SCHEMA or not ocr.get('layout_aware')) and not human:
            category='ocr_pending';reasons.append('Chờ lượt OCR đối chiếu đầy đủ')
        else:
            hold=p.get('auto_identity_review',{})
            if not human and hold.get('status')=='needs_review' and hold.get('reason') in (
                    'ambiguous-validation-reference','conflicting-plate-evidence','saved-manual-plate-differs',
                    'two-high-confidence-methods-required','low-confidence-or-invalid-plate'):
                reasons.append('Định danh tự động chưa đủ chắc hoặc nghi nhầm biển; cần người duyệt')
            check=ocr.get('crop_check',{})
            if not human and check and check.get('status')!='pass':
                reasons.append(check.get('reason') or 'Crop chưa đạt kiểm tra trước OCR')
            if not triage_valid(best): reasons.append('Chưa đọc được biển số đúng định dạng')
            if not human and triage_score(ocr.get('confidence'))<MIN_CONFIDENCE:
                reasons.append('Điểm tin cậy OCR dưới 0,90')
            support={c.get('method') for c in ocr.get('candidates',[])
                     if triage_plate(c.get('plate',''))==best
                     and triage_score(c.get('confidence'))>=MIN_CONFIDENCE
                     and c.get('region',0)==ocr.get('region',0)}
            if not human and not ('det-lines' in support and ('whole-crop' in support or 'split-lines' in support)):
                reasons.append('Chưa có hai cách OCR cùng đọc rõ một biển số')
            by_region=defaultdict(set)
            # Historical reads remain leakage evidence, not independent votes
            # about the newly checked crop. Only the current pass may support
            # or contradict its recognition confidence.
            current_candidates=triage_candidates(dict(p,ocr=dict(ocr,comparison_candidates=[])),e)
            for c in current_candidates:
                if triage_valid(c['plate']) and c['confidence']>=.70:
                    by_region[c['region']].add(c['plate'])
            if not human and any(len(values)>1 for values in by_region.values()):
                reasons.append('Các cách OCR đọc khác nhau trên cùng một biển')
            if ocr.get('full_frame') or ocr.get('regions_truncated') or row['stage']=='multiple':
                reasons.append('OCR toàn cảnh hoặc ảnh có nhiều vùng nghi biển số')
            if ocr.get('error') and not human: reasons.append('OCR gặp lỗi; cần kiểm tra ảnh')
            if reasons: category='ocr_uncertain'
            elif row['kind']=='validation': category='reference_ready'
            elif row['stage']=='crop_only': category='ocr_only'
            elif row['stage']!='ready' or len(e.get('points',p.get('points',[])))!=4: category='corners_pending'
            else: category='ocr_distinct'
    return dict(category=category,reasons=reasons,matches=matches[:20],
                match_count=len(matches),confidence=triage_score(ocr.get('confidence')),
                has_exact_match=any(m['method']=='exact-plate' for m in matches),
                match_digest=triage_match_digest(matches),
                comparison_is_provisional=True,training_ready=False)


def triage_snapshot(workspace):
    workspace=Path(workspace)
    with closing(sqlite3.connect('file:'+str(workspace/'curation.sqlite')+'?mode=ro',uri=True,timeout=60)) as db:
        db.row_factory=sqlite3.Row
        db.execute('BEGIN')
        rows=db.execute('SELECT id,kind,stage,proposal,edit,version,dhash FROM images').fetchall()
        meta={r[0]:json.loads(r[1]) for r in db.execute('SELECT key,value FROM meta')}
    index=triage_validation_index(rows)
    counts=Counter();queues=defaultdict(list)
    for row in rows:
        if row['kind'] not in ('acs','roboflow','validation','baseline'): continue
        result=triage_record(row,index)
        counts[row['kind']+':'+result['category']]+=1
        if result['category'] not in ('pending','duplicate','excluded'):
            queues[result['category']].append(dict(id=row['id'],kind=row['kind'],version=row['version'],**result))
    pending=sum(r['stage']=='pending' for r in rows)
    ocr_remaining=0
    for r in rows:
        if r['stage'] in ('pending','duplicate','error'): continue
        p,e=json.loads(r['proposal']),json.loads(r['edit'])
        if e.get('exclude') or p.get('source_deleted'): continue
        o=p.get('ocr',{})
        if (o.get('schema')!=OCR_SCHEMA or not o.get('layout_aware')
                or (meta.get('crop_pipeline') and o.get('crop_pipeline')!=meta['crop_pipeline'])): ocr_remaining+=1
    return dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat(),
        counts=dict(counts),validation_coverage=index['coverage'],minimum_ocr_confidence=MIN_CONFIDENCE,
        processing_complete=bool(meta.get('processing_complete')) and pending==0,
        ocr_complete=bool(meta.get('ocr_all_complete')) and ocr_remaining==0,
        pending_images=pending,ocr_remaining_after_processing=ocr_remaining,training_ready=False),queues


def triage_write_snapshot(workspace):
    workspace=Path(workspace).resolve()
    repo=Path(__file__).resolve().parents[3]
    target=workspace/'triage'
    if workspace==repo or repo in workspace.parents or target.is_symlink() or workspace not in target.resolve().parents:
        raise ValueError('Private triage output must stay outside Git and inside its workspace')
    summary,queues=triage_snapshot(workspace)
    target.mkdir(exist_ok=True)
    files={'summary.json':json.dumps(summary,ensure_ascii=False,indent=2)}
    for category in ('suspected_overlap','confirmed_overlap','ocr_uncertain','ocr_distinct',
                     'ocr_pending','corners_pending','ocr_only','reference_ready','file_error'):
        files[category+'.jsonl']=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in queues.get(category,[]))
    # Summary is replaced last and signals a completed snapshot.
    for name in [n for n in files if n!='summary.json']+['summary.json']:
        output=target/name;temporary=target/(name+'.tmp')
        temporary.write_text(files[name],encoding='utf-8',newline='\n')
        for attempt in range(8):
            try: os.replace(temporary,output);break
            except PermissionError:
                if attempt==7: raise
                time.sleep(.05*(attempt+1))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True);parser.add_argument('--follow',action='store_true')
    args=parser.parse_args()
    while True:
        result=triage_write_snapshot(args.workspace)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        if not args.follow or (result['processing_complete'] and result['ocr_complete']): break
        time.sleep(30)
