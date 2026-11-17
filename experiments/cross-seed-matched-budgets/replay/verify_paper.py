"""Verify supplementary paper statistics against portable archived outputs."""
import json
from pathlib import Path
import sys
import math
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'analysis'))
from statistics import main as compute
from replay import read,hits,array
from sys import stdout
expected=read(BASE/'analysis/statistics.json');got=compute(BASE);assert got==expected
labels_by_split={s:read(BASE/'labels'/f'{s}.json') for s in ('development','test')}
metrics=read(BASE/'p2pnet/evaluation.json')
for split,labels in labels_by_split.items():
 bank=read(BASE/'banks'/f'{split}.json')
 for arm,ref in metrics[split].items():
  forecasts=read(BASE/'p2pnet'/f'{arm}.json')
  count=[];matched=0;empty=0
  for k,y in labels.items():
   x=forecasts[k];count.append(abs(len(x)-len(y)));matched+=hits(x,y,8*max(bank[k]['size'])/1024)
   if not y:empty+=len(x)
  assert np.mean(count)==ref['overall']['mae'] and matched==ref['overall']['tp'] and empty==ref['empty']['fp']
stats=read(BASE/'analysis/scale_oracle_design.json');replay=read(BASE/'expected.json')
for s,info in stats['oracle'].items():
 o=replay['splits'][s]['oracles']['verifier88']['per_frame'];assert sum(r['all_hits'] for r in o)==info['all_hits']
 ok=[r for r in o if 0<=r['truth']-r['seeds']<=r['available_additions']]
 assert len(ok)==info['feasible'] and sum(min(r['all_hits'],r['seed_hits']+r['truth']-r['seeds']) for r in ok)==info['feasible_seed_constrained_matches']
 tf=labels_by_split[s];ext=read(BASE/'predictions'/f'{s}.json');b=read(BASE/'banks'/f'{s}.json')
 assert sum(hits(ext[k]['CountGD_native'],y,8*max(b[k]['size'])/1024) for k,y in tf.items())==info['native_countgd_gt_cap_hits']
d=stats['design']
threshold=next(i for i in range(d['groups']+1) if sum(math.comb(d['groups'],j) for j in range(i,d['groups']+1))/2**d['groups']<=.05)
assert threshold==d['required_positive_groups']
print('All archived P2PNet endpoints, relative errors, spatial metrics, confidence intervals and oracle checks passed.')
