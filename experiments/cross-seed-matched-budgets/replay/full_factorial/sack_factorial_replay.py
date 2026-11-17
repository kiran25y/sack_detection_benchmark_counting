"""Complete the frozen verifier crossing and its scalar baseline from saved banks."""
import argparse
from fractions import Fraction
import importlib.util
from itertools import product
import json
from pathlib import Path

import numpy as np


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def run(folder):
    replay=module('bank_replay',folder/'replay.py')
    stats=module('paired_stats',folder/'analysis/statistics.py')
    locked=replay.read(folder/'locked_protocol.json')['calibration']
    scale=Fraction(replay.read(folder/'vote_scale_expected.json')['exact'])
    output={}
    for split in ('development','test'):
        bank=replay.prepared(folder/'banks'/f'{split}.json')
        labels=replay.read(folder/'labels'/f'{split}.json')
        archived=replay.read(folder/'predictions'/f'{split}_calibrated.json')
        preds={k:{} for k in ('A','D','B','C','M')}
        for k,r in bank.items():
            raw=sum(np.array(r['raw'])[r['_eligible']]>=locked['raw']['threshold'])
            learned=sum(np.array(r['learned'])[r['_eligible']]>=locked['learned']['threshold'])
            n=len(r['seeds'])
            mul=min(int(n*scale+Fraction(1,2))-n,len(r['_eligible']))
            for name,ordering,count in [('A','raw',raw),('D','learned',raw),('B','raw',learned),('C','learned',learned),('M','raw',mul)]:
                preds[name][k]=replay.ranked(r,ordering,count)
            for n,old in [('A','A_calibrated_raw'),('B','B_learned_count_raw_rank'),('C','C_learned_count_learned_rank')]:
                assert archived[k][old]==preds[n][k]
            assert len(preds['A'][k])==len(preds['D'][k]) and len(preds['B'][k])==len(preds['C'][k])
        scores=replay.evaluate(bank,labels,preds)
        rows={k:dict(id=k,group=r['group'],truth=len(labels[k])) for k,r in bank.items()}
        summary={}
        for name in preds:
            for m in scores[name]['8']['per_frame']:
                k=m['id'];e=m['pred']-m['truth'];g=labels[k];b=bank[k]
                rows[k][name]=dict(count=m['pred'],tp=m['tp'],ae=abs(e),error=e,
                                   game1=stats.game(preds[name][k],g,b['roi'],1),game2=stats.game(preds[name][k],g,b['roi'],2))
            summary[name]=dict(scores[name]['8']['overall'])
            for metric in ('error','game1','game2'):
                summary[name][metric]=float(np.mean([r[name][metric] for r in rows.values()]))
            summary[name]['positive_mape']=float(np.mean([100*r[name]['ae']/r['truth'] for r in rows.values() if r['truth']]))
        contrasts={f'{r}-{l}':stats.contrast(list(rows.values()),l,r) for l,r in [('A','D'),('B','C'),('A','B'),('D','C'),('M','B')]}
        groups=sorted({r['group'] for r in rows.values()})
        interaction=[sum((r['C']['tp']-r['B']['tp'])-(r['D']['tp']-r['A']['tp']) for r in rows.values() if r['group']==g) for g in groups]
        draws=np.array(list(product(interaction,repeat=len(groups)))).sum(1)
        output[split]=dict(summary=summary,scores=scores,paired=contrasts,interaction=dict(total=sum(interaction),by_group=dict(zip(groups,interaction)),ci=stats.interval(draws)),new_predictions=preds['D'])
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--reviewer',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    with args.output.open('x',encoding='utf8') as f:json.dump(run(args.reviewer),f,indent=2,allow_nan=False)
