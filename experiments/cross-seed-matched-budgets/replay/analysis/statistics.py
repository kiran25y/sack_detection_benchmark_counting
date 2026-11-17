"""Recompute descriptive errors and exact paired recording bootstrap distributions."""
import argparse
from itertools import product
import json
from math import comb
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def arr(points):
    return np.asarray(points,float).reshape(-1,2)


def game(points,truth,roi,level):
    roi=np.array(roi);side=2**level
    def bins(xy):
        xy=np.clip(np.floor((arr(xy)+.5-roi[:2])/(roi[2:]-roi[:2])*side).astype(int),0,side-1)
        return np.bincount(xy[:,1]*side+xy[:,0],minlength=side**2)
    return int(abs(bins(points)-bins(truth)).sum())


def interval(x):
    return np.quantile(x,[.025,.975],method='inverted_cdf').tolist()


def signtest(x):
    n=sum(v!=0 for v in x);k=sum(v>0 for v in x)
    return min(1,2*sum(comb(n,i) for i in range(min(k,n-k)+1))/2**n)


def contrast(rows,first,second):
    groups=sorted({r['group'] for r in rows})
    vals=[];delta_err=[];delta_hits=[]
    for g in groups:
        rr=[r for r in rows if r['group']==g]
        e=[r[second]['ae']-r[first]['ae'] for r in rr]
        h=[r[second]['tp']-r[first]['tp'] for r in rr]
        delta_err.extend(e);delta_hits.extend(h)
        vals.append([len(rr),sum(e),sum(h),sum(r[first]['tp'] for r in rr),sum(r[second]['tp'] for r in rr),
                     sum(r[first]['count']+r['truth'] for r in rr),sum(r[second]['count']+r['truth'] for r in rr)])
    v=np.array(vals,dtype=float);g=len(groups)
    draw=np.array(list(product(range(g),repeat=g)))
    draws=v[draw].sum(axis=1)
    f1=np.divide(2*draws[:,4],draws[:,6],out=np.full(len(draws),np.nan),where=draws[:,6]>0)-np.divide(2*draws[:,3],draws[:,5],out=np.full(len(draws),np.nan),where=draws[:,5]>0)
    rng=np.random.default_rng(20261110)
    fdraw=rng.integers(0,len(delta_err),(20000,len(delta_err)))
    return dict(groups=groups,group_effects=[dict(frames=int(s[0]),mean_ae=s[1]/s[0],matches=int(s[2])) for s in v],
                delta_pooled_mae=float(np.mean(delta_err)),group_bootstrap_pooled_mae=interval(draws[:,1]/draws[:,0]),
                delta_group_mean_mae=float(np.mean(v[:,1]/v[:,0])),group_bootstrap_group_mean_mae=interval((v[:,1]/v[:,0])[draw].mean(1)),
                delta_total_matches=sum(delta_hits),group_bootstrap_total_matches=interval(draws[:,2]),group_bootstrap_mean_matches_per_frame=interval(draws[:,2]/draws[:,0]),
                group_bootstrap_delta_f1=interval(f1[np.isfinite(f1)]) if np.isfinite(f1).any() else None,
                frame_bootstrap_pooled_mae=interval(np.asarray(delta_err)[fdraw].mean(1)),frame_bootstrap_matches=interval(np.asarray(delta_hits)[fdraw].sum(1)),
                count_improved=sum(x<0 for x in delta_err),count_worse=sum(x>0 for x in delta_err),count_tied=delta_err.count(0),
                ranking_improved=sum(x>0 for x in delta_hits),ranking_worse=sum(x<0 for x in delta_hits),ranking_tied=delta_hits.count(0),
                frame_sign_rank_p=signtest(delta_hits),group_sign_mae_two_sided=signtest(v[:,1].tolist()))


def main(base):
    replay=read(base/'expected.json')
    result={}
    for split in ('development','test'):
        labels=read(base/'labels'/f'{split}.json');banks=read(base/'banks'/f'{split}.json')
        pred=read(base/'predictions'/f'{split}.json');new=read(base/'predictions'/f'{split}_calibrated.json')
        methods={**pred[next(iter(pred))],**new[next(iter(new))]}
        scores=replay['splits'][split]['scores']
        records=[dict(id=k,truth=len(gt),group=banks[k]['group']) for k,gt in labels.items()]
        statistics={}
        for mode in methods:
            local={x['id']:x for x in scores[mode]['8']['per_frame']}
            for r in records:
                k=r['id'];xy=(new[k] if mode in new[k] else pred[k])[mode];n=len(xy)
                assert local[k]['truth']==r['truth'] and local[k]['pred']==n
                r[mode]=dict(count=n,ae=abs(n-r['truth']),error=n-r['truth'],tp=local[k]['tp'],
                             game1=game(xy,labels[k],banks[k]['roi'],1),game2=game(xy,labels[k],banks[k]['roi'],2))
            ae=np.array([r[mode]['ae'] for r in records]);y=np.array([r['truth'] for r in records]);e=np.array([r[mode]['error'] for r in records])
            statistics[mode]=dict(mae=float(ae.mean()),bias=float(e.mean()),positive_mape=float(100*np.mean(ae[y>0]/y[y>0])),
                                  wape=float(100*sum(ae)/sum(y)),over=int(e[e>0].sum()),under=int(-e[e<0].sum()),
                                  game1=float(np.mean([r[mode]['game1'] for r in records])),game2=float(np.mean([r[mode]['game2'] for r in records])))
        methods2=['A_calibrated_raw','B_learned_count_raw_rank','C_learned_count_learned_rank']
        pairs={tag:contrast(records,l,r) for tag,l,r in [('blind_count','voting','verifier88'),('blind_rank','equal_budget','verifier88'),
               ('calibrated_count',*methods2[:2]),('calibrated_rank',*methods2[1:])]}
        y=np.array([r['truth'] for r in records]);low=(y>0)&(y<12)
        result[split]=dict(summary=statistics,contrasts=pairs,distribution=dict(frames=len(y),positives=int((y>0).sum()),low=int(low.sum()),empty=int((y==0).sum()),positive_mean=float(y[y>0].mean()),mean=float(y.mean()),median=float(np.median(y)),max=int(y.max())))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reviewer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x',encoding='utf8') as f:json.dump(main(args.reviewer),f,indent=2,allow_nan=False)
