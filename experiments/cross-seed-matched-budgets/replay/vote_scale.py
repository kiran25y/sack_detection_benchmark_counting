"""Secondary count-only calibration of the voting baseline, using development labels."""
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
import sys

if (Path(__file__).resolve().parent/'replay.py').is_file():
    from replay import prepared, read, ranked, evaluate
else:
    from replay_sack_reviewer_comparison import prepared, read, ranked, evaluate


def run(folder):
    dev=prepared(folder/'banks/development.json')
    labels=read(folder/'labels/development.json')
    sizes=Counter(r['group'] for r in dev.values())
    ratios=[]
    for k,r in dev.items():
        n=len(r['seeds'])
        if n:
            ratios.append((Fraction(len(labels[k]),n),Fraction(n,sizes[r['group']])))
    # Weighted median minimizes equal-group continuous absolute count error for a*n.
    weights=sum(w for _,w in ratios)
    acc=0
    for scale,w in sorted(ratios):
        acc+=w
        if 2*acc>=weights:break
    scale=max(Fraction(1),scale)
    out=dict(multiplier=float(scale),exact=str(scale),splits={})
    for s in ('development','test'):
        bank=prepared(folder/'banks'/f'{s}.json')
        preds={}
        for k,r in bank.items():
            n=len(r['seeds'])
            target=int(scale*n+Fraction(1,2))
            preds[k]=ranked(r,'raw',min(target-n,len(r['_eligible'])))
        scores=evaluate(bank,read(folder/'labels'/f'{s}.json'),{'vote_scale':preds})
        out['splits'][s]=scores['vote_scale']
    return out


if __name__=='__main__':
    base=Path(__file__).resolve().parent
    result=run(base)
    if len(sys.argv)==2:
        with Path(sys.argv[1]).open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    else:
        assert result==read(base/'vote_scale_expected.json')
    print(result['multiplier'],result['splits']['test']['8']['overall'])
