"""Open the sealed predictions only after exhaustive review of every frozen frame."""
import argparse
import hashlib
import json
import math
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from analyze_firc_tiny import evaluate

OUT=Path(__file__).resolve().parents[1]/'artifacts/verifier88_reserved_v1c_20261109'
MODES=('voting','verifier88','equal_budget')


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def validate(export,cohort):
    assert export['schema']=='verifier88-blind-v1' and export['cohort_sha256']==sha(OUT/'cohort.json')
    rows={r['id']:r for r in export['frames']}
    assert len(rows)==len(export['frames'])==51 and set(rows)=={c['id'] for c in cohort}
    pending=[]
    for c in cohort:
        r=rows[c['id']];assert r['image_sha256']==c['sha256']
        if r['status']!='reviewed':
            pending.append(c['id']);continue
        p=r['points']
        assert isinstance(p,list) and all(isinstance(x,list) and len(x)==2 and all(isinstance(n,(int,float)) and not isinstance(n,bool) and math.isfinite(n) for n in x) for x in p)
        assert all(0<=x<c['size'][0] and 0<=y<c['size'][1] for x,y in p)
        assert len(set(map(tuple,p)))==len(p)==r['exact_count'] and (len(p)>0 or r.get('confirmed_empty') is True)
    if pending:raise ValueError(f'{len(pending)} unfinished frames. Predictions remain sealed: '+', '.join(pending))
    return rows


def summary(records):
    if not records:return {'frames':0,'mae':None,'f1_at8':None,'reason':'No qualifying reviewed frames'}
    n=len(records);tp=sum(r['tp'] for r in records);fp=sum(r['fp'] for r in records);fn=sum(r['fn'] for r in records)
    return dict(frames=n,truth=tp+fn,predictions=tp+fp,tp=tp,fp=fp,fn=fn,
                mae=sum(r['ae'] for r in records)/n,rmse=math.sqrt(sum(r['error']**2 for r in records)/n),bias=sum(r['error'] for r in records)/n,
                f1_at8=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,game1=sum(r['game1'] for r in records)/n,game2=sum(r['game2'] for r in records)/n,
                positive_frames_without_match=sum(r['truth']>0 and r['tp']==0 for r in records),empty_frames_with_fp=sum(r['truth']==0 and r['fp']>0 for r in records))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--review',type=Path,required=True);args=ap.parse_args()
    cohort=read(OUT/'cohort.json');export=read(args.review);labels=validate(export,cohort)
    receipt=read(OUT/'prediction_receipt.json');archive=OUT/'sealed_predictions.tar.gz'
    assert sha(archive)==receipt['sha256'] and receipt['held_out']==51
    dest=OUT/'final_score';dest.mkdir(exist_ok=False)
    with (dest/'locked_labels.json').open('x',encoding='utf8') as f:
        json.dump({'locked_at':datetime.now().astimezone().isoformat(),'export_sha256':sha(args.review),'review':export},f,indent=2)
    metrics={k:[] for k in MODES}
    with tarfile.open(archive) as z:
        report=json.load(z.extractfile('report.json'))
        for c in cohort:
            filename=c['id']+'.json';raw=z.extractfile(filename).read()
            assert hashlib.sha256(raw).hexdigest()==report['hashes'][filename]
            result=json.loads(raw);assert result['id']==c['id'] and result['input_sha256']==c['sha256']
            row=dict(image_id=c['id'],dependency_group=c['group'],width=c['size'][0],height=c['size'][1],roi_xyxy=c['roi'])
            gt=np.asarray(labels[c['id']]['points'],float).reshape(-1,2)
            pred=result['predictions'];assert set(pred)==set(MODES) and len(pred['verifier88'])==len(pred['equal_budget'])
            seed=pred['voting'];assert pred['verifier88'][:len(seed)]==pred['equal_budget'][:len(seed)]==seed
            for mode in MODES:
                xy=np.asarray(pred[mode],float).reshape(-1,2);m=evaluate(xy,gt,row)
                dist=np.linalg.norm(xy[:,None,:]-gt[None,:,:],axis=2)*1024/max(c['size'])
                hits=0
                if dist.size:
                    i,j=linear_sum_assignment(np.where(dist<=8,dist/8,max(dist.shape)+1))
                    hits=int((dist[i,j]<=8).sum())
                assert hits==m['tp'] and np.isfinite(xy).all()
                metrics[mode].append(m)
    result={}
    for mode,rs in metrics.items():
        result[mode]={'overall':summary(rs),'low_1_to_11':summary([r for r in rs if 1<=r['truth']<=11]),
                      'empty':summary([r for r in rs if r['truth']==0]),
                      'groups':{g:summary([r for r in rs if r['group']==g]) for g in sorted({r['group'] for r in rs})}}
    assert result['verifier88']['overall']['mae']==result['equal_budget']['overall']['mae']
    with (dest/'result.json').open('x',encoding='utf8') as f:json.dump(dict(summary=result,per_frame=metrics,labels_sha256=sha(dest/'locked_labels.json'),sealed_sha256=sha(archive)),f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
