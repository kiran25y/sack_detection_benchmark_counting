"""Portable reviewer replay. Python 3, NumPy. Run from the extracted review folder."""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def array(x):
    return np.asarray(x, dtype=float).reshape(-1, 2)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hits(points, truth, radius):
    xy, gt = array(points), array(truth)
    neighbors = [np.flatnonzero(((xy-g)**2).sum(1)<=radius**2) for g in gt]
    owner = {}
    def visit(g,seen):
        for j in neighbors[g]:
            if j in seen:continue
            seen.add(j)
            if j not in owner or visit(owner[j],seen):
                owner[j]=g
                return True
        return False
    return sum(visit(g,set()) for g in range(len(gt)))


def eligible(row):
    xy, seeds = array(row['points']), array(row['seeds'])
    assert len(xy)==len(row['raw'])==len(row['learned'])==64
    assert np.isfinite(xy).all() and np.isfinite(seeds).all()
    assert all(np.isfinite(row[k]).all() and np.all((0<=np.array(row[k]))&(np.array(row[k])<=1)) for k in ('raw','learned'))
    radius=8*max(row['size'])/1024
    # The frozen generator has already applied the same exclusion radius within its bank.
    d2=((xy[:,None]-xy[None])**2).sum(2)
    assert np.all(d2[np.triu_indices(len(xy),1)]>radius**2)
    return np.flatnonzero((((xy[:,None]-seeds[None])**2).sum(2)>radius**2).all(1))


def calibrate(bank, truth, field):
    groups=sorted({b['group'] for b in bank.values()})
    curves=[]
    for t in [k/100 for k in range(101)]+[1.000001]:
        errors={g:[] for g in groups};empty=0
        for k,b in bank.items():
            n=len(b['seeds'])+int((np.asarray(b[field])[b['_eligible']]>=t).sum())
            errors[b['group']].append(abs(n-len(truth[k])))
            if not truth[k]:empty+=n
        e=sum(Fraction(sum(v),len(v)) for v in errors.values())/len(groups)
        curves.append((e,empty,-t))
    e,empty,neg=min(curves)
    return dict(threshold=-neg,equal_group_mae=float(e),empty_points=empty)


def ranked(b,field,quota):
    ix=b['_eligible']
    order=ix[np.argsort(-np.array(b[field])[ix],kind='stable')[:quota]]
    return list(b['seeds'])+[b['points'][j] for j in order]


def predict(row,thresholds):
    counts={field:int((np.array(row[field])[row['_eligible']]>=thresholds[field]['threshold']).sum()) for field in ('raw','learned')}
    return {'A_calibrated_raw':ranked(row,'raw',counts['raw']),
            'B_learned_count_raw_rank':ranked(row,'raw',counts['learned']),
            'C_learned_count_learned_rank':ranked(row,'learned',counts['learned'])}


def metrics(points,truth,b,rad):
    tp=hits(points,truth,rad*max(b['size'])/1024)
    return dict(id=b['id'],group=b['group'],truth=len(truth),pred=len(points),tp=tp)


def summary(rows):
    if not rows:return dict(frames=0)
    n=len(rows);g=sum(r['truth'] for r in rows);p=sum(r['pred'] for r in rows);tp=sum(r['tp'] for r in rows)
    empty=[r for r in rows if r['truth']==0]
    return dict(frames=n,truth=g,pred=p,tp=tp,fp=p-tp,fn=g-tp,mae=sum(abs(r['pred']-r['truth']) for r in rows)/n,
                precision=tp/p if p else None,recall=tp/g if g else None,f1=2*tp/(p+g) if p+g else None,
                empty_frames=len(empty),empty_false_points=sum(r['pred'] for r in empty),
                empty_frames_with_fp=sum(r['pred']>0 for r in empty),
                empty_false_alarm_fraction=sum(r['pred']>0 for r in empty)/len(empty) if empty else None)


def evaluate(bank,truth,methods):
    assert set(truth)==set(bank)
    scores={}
    for name,predictions in methods.items():
        assert set(predictions)==set(bank)
        per_radius={}
        for rad in (4,8,16):
            rows=[metrics(predictions[k],truth[k],b,rad) for k,b in bank.items()]
            per_radius[str(rad)]=dict(overall=summary(rows),low=summary([r for r in rows if 0<r['truth']<12]),
                groups={g:summary([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})},
                per_frame=rows)
        scores[name]=per_radius
    return scores


def ceiling(bank,truth,fixed_counts):
    records=[]
    for k,b in bank.items():
        seeds=b['seeds'];xy=[b['points'][i] for i in b['_eligible']]
        quota=fixed_counts[k]-len(seeds)
        assert 0<=quota<=len(xy)
        rad=8*max(b['size'])/1024
        seed_hits=hits(seeds,truth[k],rad);all_hits=hits(seeds+xy,truth[k],rad)
        # A maximum matching of the seeds extends by augmenting paths in the
        # enlarged bipartite graph. Each extra point can add at most one match.
        oracle=min(all_hits,seed_hits+quota)
        records.append(dict(id=k,group=b['group'],truth=len(truth[k]),total_budget=fixed_counts[k],
                            seeds=len(seeds),seed_hits=seed_hits,available_additions=len(xy),quota=quota,
                            all_hits=all_hits,oracle=oracle))
    return dict(total={f:sum(r[f] for r in records) for f in ('truth','total_budget','seed_hits','all_hits','oracle')},per_frame=records)


def prepared(path):
    bank=read(path)
    for k,b in bank.items():
        assert b['id']==k
        b['_eligible']=eligible(b)
    return bank


def run(folder):
    dev=prepared(folder/'banks/development.json')
    dev_y=read(folder/'labels/development.json')
    assert set(dev)==set(dev_y)
    cuts={k:calibrate(dev,dev_y,k) for k in ('raw','learned')}
    out={'calibration':cuts,'splits':{}}
    for split in ('development','test'):
        bank=dev if split=='development' else prepared(folder/'banks/test.json')
        archive=read(folder/'predictions'/f'{split}.json')
        modes={name:{k:v[name] for k,v in archive.items()} for name in next(iter(archive.values()))}
        fresh={}
        for k,b in bank.items():
            historical=int((np.array(b['learned'])[b['_eligible']]>=0.8346997170417895).sum())
            for name,field in [('verifier88','learned'),('equal_budget','raw')]:
                xy=array(ranked(b,field,historical));original=array(modes[name][k])
                assert xy.shape==original.shape and np.allclose(xy,original,atol=1e-5,rtol=0),(k,name)
            for name,xy in predict(b,cuts).items():fresh.setdefault(name,{})[k]=xy
        truth=read(folder/'labels'/f'{split}.json')
        modes.update(fresh)
        scores=evaluate(bank,truth,modes)
        oracle={name:ceiling(bank,truth,{k:len(v) for k,v in modes[name].items()}) for name in ['verifier88','C_learned_count_learned_rank']}
        a,b,c=[scores[n]['8'] for n in ('A_calibrated_raw','B_learned_count_raw_rank','C_learned_count_learned_rank')]
        pairs={}
        for g in a['groups']:
            first,second,third=[x['groups'][g] for x in (a,b,c)]
            pairs[g]=dict(frames=first['frames'],B_minus_A_mae=second['mae']-first['mae'],C_minus_B_matches=third['tp']-second['tp'])
        assert all(len(fresh['B_learned_count_raw_rank'][k])==len(fresh['C_learned_count_learned_rank'][k]) for k in bank)
        out['splits'][split]=dict(scores=scores,oracles=oracle,paired_groups=pairs,
                    equal_group_B_minus_A_mae=float(np.mean([p['B_minus_A_mae'] for p in pairs.values()])))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path,help='Optionally save the full per-frame replay')
    args=parser.parse_args();result=run(args.folder)
    for k in ('voting','verifier88','equal_budget','A_calibrated_raw','B_learned_count_raw_rank','C_learned_count_learned_rank'):
        m=result['splits']['test']['scores'][k]['8']['overall']
        print(f"{k:30s} MAE {m['mae']:.4f}   P {m['precision']:.4f}   R {m['recall']:.4f}   F1 {m['f1']:.4f}   matches {m['tp']}")
    print('Thresholds from development:',result['calibration'])
    expected=args.folder/'expected.json'
    if expected.exists():
        assert result==read(expected),'Saved result differs from replay'
        print('All reference results reproduced.')
    if args.output:
        with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
