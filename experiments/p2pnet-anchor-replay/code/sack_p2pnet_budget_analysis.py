"""Inspect unthresholded P2PNet banks. Calibrate on development totals only."""
import argparse
from collections import Counter
from io import BytesIO
import hashlib
from itertools import product
import json
from math import lcm
from pathlib import Path
import tarfile

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

from sack_factorial_replay import module


ARMS=('pretrained','1729','1730','1731')


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def load_bank(tar, name, row):
    with np.load(BytesIO(tar.extractfile(name+'_'+row['id']+'.npz').read()),allow_pickle=False) as v:
        xy=(v['xy']+.5)/v['scale']+np.asarray(row['roi'][:2])-.5
        return xy,v['score']


def calibration(scores,groups,totals):
    """Sweep tied score events exactly, weighting both development groups equally."""
    sizes=Counter(groups.values());unit=lcm(*sizes.values())
    events=[];changes=[];empty_changes=[];loss0=0
    for k,s in scores.items():
        y=totals[k];w=unit//sizes[groups[k]]
        desc=np.sort(s)[::-1]
        events.append(desc);changes.append(w*np.where(np.arange(len(desc))<y,-1,1))
        empty_changes.append(np.full(len(desc),int(y==0),dtype=np.int32))
        loss0+=w*y
    e=np.concatenate(events);order=np.argsort(-e,kind='stable');e=e[order]
    ends=np.r_[np.flatnonzero(e[1:]!=e[:-1]),len(e)-1]
    thresholds=np.r_[1.000001,e[ends].astype(float)]
    error=np.r_[loss0,loss0+np.cumsum(np.concatenate(changes)[order],dtype=np.int64)[ends]]
    zeros=np.r_[0,np.cumsum(np.concatenate(empty_changes)[order],dtype=np.int64)[ends]]
    best=np.lexsort((-thresholds,zeros,error))[0]
    t=float(thresholds[best])
    counts={k:int(np.sum(s.astype(float)>=t)) for k,s in scores.items()}
    actual=sum(abs(n-totals[k])*(unit//sizes[groups[k]]) for k,n in counts.items())
    assert actual==error[best] and sum(v for k,v in counts.items() if not totals[k])==zeros[best]
    return dict(threshold=t,equal_group_mae=actual/(unit*len(sizes)),empty_points=int(zeros[best]),operating_points=len(thresholds))


def coverage(xy,conf,gt,radius):
    if not len(gt):return 0,[],0
    neighbors=cKDTree(xy).query_ball_point(gt,radius)
    rr=[];cc=[];best=[]
    for j,near in enumerate(neighbors):
        rr.extend([j]*len(near));cc.extend(near)
        best.append(float(conf[near].max()) if len(near) else None)
    if not cc:return 0,best,0
    graph=csr_matrix((np.ones(len(rr),np.uint8),(rr,cc)),shape=(len(gt),len(xy)))
    return int((maximum_bipartite_matching(graph,perm_type='column')>=0).sum()),best,sum(x is not None for x in best)


def quantiles(values):
    q=np.quantile(values,[0,.25,.5,.75,.95,.99,.999,1])
    return dict(zip(['min','p25','p50','p75','p95','p99','p999','max'],map(float,q)))


def run(tarpath,reviewer,design,verify):
    assert verify['report']['verified_native_predictions']==296
    stats=module('paired_stats',reviewer/'analysis/statistics.py')
    replay=module('bank_replay',reviewer/'replay.py')
    rows={r['id']:r for r in design['rows']};assert len(rows)==74
    dev={r['id']:r['group'] for r in rows.values() if r['split']=='development'}
    ydev=read(reviewer/'labels/development.json')
    thresholds={};score_summaries={}
    with tarfile.open(tarpath) as tar:
        for n,h in verify['report']['hashes'].items():
            assert hashlib.sha256(tar.extractfile(n).read()).hexdigest()==h
        for arm in ARMS:
            scores={k:load_bank(tar,arm,rows[k])[1] for k in dev}
            thresholds[arm]=calibration(scores,dev,{k:len(v) for k,v in ydev.items()})
            score_summaries[arm]=quantiles(np.concatenate(list(scores.values())))
        output={}
        for split in ('development','test'):
            labels=read(reviewer/'labels'/f'{split}.json')
            locs={};per_frame={};cov={a:[] for a in ARMS}
            for k,points in labels.items():
                r=rows[k];gt=np.asarray(points,float).reshape(-1,2);radius=8*max(r['size'])/1024
                b={a:load_bank(tar,a,r) for a in ARMS}
                assert len({len(v[0]) for v in b.values()})==1
                rankings={a:np.argsort(-v[1],kind='stable') for a,v in b.items()}
                budgets={a:int((v[1].astype(float)>=thresholds[a]['threshold']).sum()) for a,v in b.items()}
                selections={}
                for a,(xy,s) in b.items():
                    cap,best,any_coverage=coverage(xy,s,gt,radius)
                    cov[a].append(dict(id=k,group=r['group'],truth=len(gt),bank=len(xy),max_matches=cap,nearby_max=best,any_coverage=any_coverage,
                                       oracle_at_calibrated_count=min(budgets[a],cap),calibrated_count=budgets[a]))
                    selections[a+'_native']=xy[s>.5]
                    selections[a+'_calibrated']=xy[rankings[a][:budgets[a]]]
                for geo in ('1729','pretrained'):
                    for cell,count,rank in [('A','pretrained','pretrained'),('D','pretrained','1729'),('B','1729','pretrained'),('C','1729','1729')]:
                        selections[geo+'_fixed_'+cell]=b[geo][0][rankings[rank][:budgets[count]]]
                per_frame[k]=dict(id=k,group=r['group'],truth=len(points))
                for name,xy in selections.items():
                    hits=replay.hits(xy,gt,radius)
                    locs.setdefault(name,{})[k]=xy.tolist()
                    e=len(xy)-len(gt)
                    per_frame[k][name]=dict(count=len(xy),tp=hits,ae=abs(e),error=e,game1=stats.game(xy,gt,r['roi'],1),game2=stats.game(xy,gt,r['roi'],2))
                for geo in ('1729','pretrained'):
                    for l,rr in [('A','D'),('B','C')]:
                        assert len(selections[geo+'_fixed_'+l])==len(selections[geo+'_fixed_'+rr])
            summary={}
            for name in locs:
                m=[dict(id=k,group=r['group'],truth=r['truth'],pred=r[name]['count'],tp=r[name]['tp']) for k,r in per_frame.items()]
                summary[name]=dict(overall=replay.summary(m),low=replay.summary([v for v in m if 0<v['truth']<12]),
                                  groups={g:replay.summary([v for v in m if v['group']==g]) for g in sorted({v['group'] for v in m})})
                for metric in ('error','game1','game2'):
                    summary[name]['overall'][metric]=float(np.mean([r[name][metric] for r in per_frame.values()]))
            csummary={}
            for a,values in cov.items():
                near=[v for r in values for v in r['nearby_max'] if v is not None]
                csummary[a]=dict(truth=sum(v['truth'] for v in values),max_matches=sum(v['max_matches'] for v in values),
                                 oracle_at_calibrated_count=sum(v['oracle_at_calibrated_count'] for v in values),candidates=sum(v['bank'] for v in values),
                                 covered_targets=len(near),nearby_scores=quantiles(near),frames=values)
            paired={geo:{l+'-'+rr:stats.contrast(list(per_frame.values()),geo+'_fixed_'+rr,geo+'_fixed_'+l)
                         for l,rr in [('B','A'),('D','A'),('C','B')]} for geo in ('1729','pretrained')}
            output[split]=dict(summary=summary,coverage=csummary,paired=paired,predictions=locs)
    return dict(calibration=thresholds,all_dev_scores=score_summaries,splits=output)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for key in ('bank','reviewer','experiment','output'):
        ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args()
    result=run(args.bank,args.reviewer,read(args.experiment/'design.json'),read(args.experiment/'capture_verified.json'))
    with args.output.open('x',encoding='utf8') as f:json.dump(result,f,separators=(',',':'),allow_nan=False)
