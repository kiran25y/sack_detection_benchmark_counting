"""Score one locked sack-only transfer experiment on development and the reused test."""
import json
import math
import tarfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from analyze_firc_tiny import evaluate
from score_verifier88_holdout import summary

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/sack_auxiliary_transfer_v1_20261110'
HELD=ROOT/'artifacts/verifier88_reserved_v1c_20261109'
OLD=ROOT/'artifacts/sack_recording_excluded_v1_20261109/recovery'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def additions(parent,bank,quota,radius):
    chosen=list(parent)
    for k in np.argsort(-np.array(bank['scores']),kind='stable'):
        if len(chosen)-len(parent)>=quota:break
        pt=np.array(bank['points'][k],dtype=float)
        if chosen and np.min(np.linalg.norm(np.asarray(chosen)-pt,axis=1))<=radius:continue
        chosen.append(pt.tolist())
    assert len(chosen)-len(parent)==quota
    return chosen


def partition(records):
    return dict(overall=summary(records),low=summary([r for r in records if 0<r['truth']<12]),
                empty=summary([r for r in records if not r['truth']]),
                groups={g:summary([r for r in records if r['group']==g]) for g in sorted({r['group'] for r in records})})


def main():
    assert read(OUT/'verification.json')['passed']
    plan=read(OUT/'plan.json');candidates=read(OUT/'collected/candidates.json');freeze=read(OUT/'collected/freeze.json')
    parent={r['id']:r['points'] for r in read(OLD/'predictions.json') if r['method']=='included'}
    with tarfile.open(HELD/'sealed_predictions.tar.gz') as z:
        for c in read(HELD/'cohort.json'):parent[c['id']]=json.load(z.extractfile(c['id']+'.json'))['predictions']['verifier88']
    # Generate every point set before reading evaluation truth.
    pred={}
    for r in plan['benchmark']:
        if r['split']=='train':continue
        k=r['id'];base=parent[k];radius=8*max(r['size'])/1024
        modes={'verifier':base}
        for arm in plan['arms']:
            b=candidates[arm][k]
            n=math.floor(b['mass']+.5) if b['mass']>freeze[arm]['empty_floor'] else 0
            quota=max(0,min(64,n-len(base)))
            modes[arm]=additions(base,b,quota,radius)
        modes['equal_budget_points_only']=additions(base,candidates['points'][k],len(modes['both'])-len(base),radius)
        pred[k]=modes
    with (OUT/'predictions.json').open('x',encoding='utf8') as f:json.dump(pred,f,allow_nan=False)
    truth={r['image_id']:r['points'] for r in read(ROOT/'artifacts/ipic_transfer_pilot_data_v1_20261104/sack_manifest.json') if r['split']=='development'}
    truth.update({r['id']:r['points'] for r in read(HELD/'final_score/locked_labels.json')['review']['frames']})
    assert set(truth)==set(pred)
    scoring={}
    for split in ('development','test'):
        methods={}
        for mode in ['verifier',*plan['arms'],'equal_budget_points_only']:
            records=[]
            for r in plan['benchmark']:
                if r['split']!=split:continue
                key=r['id'];gt=np.array(truth[key],float).reshape(-1,2);xy=np.array(pred[key][mode],float).reshape(-1,2)
                m=evaluate(xy,gt,dict(image_id=key,width=r['size'][0],height=r['size'][1],dependency_group=r['group'],roi_xyxy=r['roi']))
                d=np.linalg.norm(xy[:,None]-gt[None,:],axis=2)*1024/max(r['size']);hits=0
                if d.size:
                    i,j=linear_sum_assignment(np.where(d<=8,d/8,max(d.shape)+1));hits=int(sum(d[i,j]<=8))
                assert hits==m['tp'];records.append(m)
            methods[mode]=dict(**partition(records),per_frame=records)
        scoring[split]=methods
    historic=read(HELD/'final_score/result.json')['summary']['verifier88']
    assert scoring['test']['verifier']['overall']==historic['overall']
    dev=scoring['development']['verifier']['overall'];assert dev['tp']==88 and round(dev['mae'],2)==6.35
    checks={}
    for split,m in scoring.items():
        p=m['both'];old=m['verifier'];ab=m['points'];eq=m['equal_budget_points_only']
        assert p['overall']['mae']==eq['overall']['mae']
        checks[split]=dict(lower_mae=p['overall']['mae']<old['overall']['mae'],higher_f1=p['overall']['f1_at8']>old['overall']['f1_at8'],
            beats_points_only_mae=p['overall']['mae']<ab['overall']['mae'],beats_points_only_f1=p['overall']['f1_at8']>ab['overall']['f1_at8'],
            better_equal_budget_placement=p['overall']['tp']>eq['overall']['tp'],empty_zero=p['empty']['fp']==0,
            low_mae_protected=p['low']['mae']<=old['low']['mae'],low_matches_protected=p['low']['tp']>=old['low']['tp'],
            game1_protected=p['overall']['game1']<=old['overall']['game1'],game2_protected=p['overall']['game2']<=old['overall']['game2'])
    with (OUT/'result.json').open('x',encoding='utf8') as f:json.dump(dict(results=scoring,checks=checks),f,indent=2,allow_nan=False)
    for split,m in scoring.items():
        print(split)
        for k,v in m.items():
            print(k,*(round(v['overall'][q],4) for q in ('mae','tp','f1_at8','fp','game1','game2')),v['low']['fn'],v['empty']['fp'])
    print(checks)


if __name__=='__main__':main()
