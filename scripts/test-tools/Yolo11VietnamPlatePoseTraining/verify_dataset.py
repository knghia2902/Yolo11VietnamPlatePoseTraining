"""Verify the immutable 10000/1500 pose training snapshot before use."""
from pathlib import Path
import hashlib,json,math

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['counts'] in ({'train':9996,'validation':1500},{'train':10000,'validation':1500}),'Unexpected dataset counts'
    hashes={'train':set(),'validation':set()};pixels={'train':set(),'validation':set()};ids=set();counts={'train':0,'validation':0}
    for row in manifest['records']:
        assert row['id'] not in ids,'Repeated image ID';ids.add(row['id'])
        split=row['split'];assert split in counts
        for key,checksum in [('image','sha256'),('label','label_sha256')]:
            relative=Path(row[key]);assert not relative.is_absolute() and '..' not in relative.parts
            path=root/relative;assert path.resolve().is_relative_to(root)
            assert sha(path)==row[checksum],f'Changed file: {relative}'
        values=list(map(float,(root/row['label']).read_text(encoding='utf-8').split()))
        assert len(values)==17 and values[0]==0 and all(math.isfinite(v) for v in values)
        assert all(0<=v<=1 for v in values[1:5]) and values[3]>0 and values[4]>0
        points=[[values[i],values[i+1]] for i in (5,8,11,14)]
        assert all(0<=v<=1 for p in points for v in p) and all(v==2 for v in values[7::3])
        cross=[]
        for i in range(4):
            a,b,c=points[i],points[(i+1)%4],points[(i+2)%4]
            cross.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
        assert min(cross)>1e-8
        assert points[0][1]+points[1][1]<points[2][1]+points[3][1]
        assert points[0][0]+points[3][0]<points[1][0]+points[2][0]
        counts[split]+=1;hashes[split].add(row['sha256'])
        if row.get('pixel'):pixels[split].add(row['pixel'])
    assert counts==manifest['counts']
    assert not hashes['train'].intersection(hashes['validation']),'Exact train/validation overlap'
    assert not pixels['train'].intersection(pixels['validation']),'Pixel train/validation overlap'
    assert sha(root/'weights'/'yolo11n-pose.pt')==manifest['initial_weights_sha256']
    print('Verified dataset:',counts,flush=True)
    return manifest

if __name__=='__main__':
    import sys
    verify(Path(sys.argv[1]))
