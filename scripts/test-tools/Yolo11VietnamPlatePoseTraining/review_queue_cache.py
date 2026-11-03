"""One bounded queue snapshot, invalidated by SQLite or corner-index changes."""
from pathlib import Path


def queue_signature(workspace, kind, status, vehicle):
    if kind not in ('acs','roboflow') or status not in ('corners','corners_done','pending'):
        return None
    root=Path(workspace)
    values=[]
    for name in ('curation.sqlite','curation.sqlite-wal','corner-review-index.json'):
        try:
            stat=(root/name).stat(); values.append((stat.st_mtime_ns,stat.st_size))
        except FileNotFoundError: values.append(None)
    return (kind,status,vehicle,tuple(values))


def cached_page(store,key,offset,anchor):
    cached=getattr(store,'_review_queue_snapshot',None)
    if key is None or not cached or cached[0]!=key: return None
    _,candidates,plates,missing,suggestions,reference,result=cached
    from expansion_triage import triage_record
    from collections import Counter
    if anchor: offset=next((n for n,(row,_) in enumerate(candidates) if row['id']==anchor),offset)
    offset=min(offset,max(0,len(candidates)-1))
    records=[store.public_row(row,plates,missing,suggestions,triage or triage_record(row,reference)) for row,triage in candidates[offset:offset+40]]
    return dict(result,records=records,offset=offset,triage_counts=dict(Counter(r['triage']['category'] for r in records)))
