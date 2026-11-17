"""Replay fixed-count rankings and candidate budgets. All selections precede scoring."""
import argparse
from collections import defaultdict
from itertools import product
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from sack_factorial_replay import module
from sack_p2pnet_budget_analysis import ARMS, load_bank, read


def scored(replay, points, truth, radius):
    return {'n': len(points), 'tp': replay.hits(points, truth, radius),
            'ae': abs(len(points) - len(truth))}


def paired(rows, before, after):
    groups = sorted({row['group'] for row in rows})
    changes = {g: sum(row[after]['tp'] - row[before]['tp'] for row in rows
                      if row['group'] == g) for g in groups}
    totals = [sum(x) for x in product(changes.values(), repeat=len(groups))]
    bounds = np.quantile(totals, [.025, .975], method='inverted_cdf')
    return {'delta': sum(changes.values()), 'groups': changes, 'interval': bounds.tolist()}


def summarize(rows):
    return {key: {'pred': sum(r[key]['n'] for r in rows),
                  'matches': sum(r[key]['tp'] for r in rows),
                  'mae': float(np.mean([r[key]['ae'] for r in rows]))}
            for key in rows[0] if key not in {'id', 'group', 'truth'}}


def run(root, bank_dir):
    replay = module('review_base', root / 'replay.py')
    source = read(bank_dir / 'analysis.json')
    locked = read(root / 'locked_protocol.json')['calibration']
    design = {r['id']: r for r in read(bank_dir / 'design.json')['rows']}
    identities = read(bank_dir / 'capture_verified.json')['report']['hashes']
    assert len(identities) == 296 and len(design) == 74
    results = {}
    with tarfile.open(bank_dir / 'banks.tar') as tar:
        assert {n for n in tar.getnames() if n.endswith('.npz')} == set(identities)
        for name, digest in identities.items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == digest
        for split in ('development', 'test'):
            labels = read(root / 'labels' / (split+'.json'))
            verifier = replay.prepared(root / 'banks' / (split+'.json'))
            assert labels.keys() == verifier.keys() == {k for k,r in design.items() if r['split']==split}
            crosses = defaultdict(list)
            pools = defaultdict(list)
            for key, truth in labels.items():
                row = design[key]
                v = verifier[key]
                assert row['group'] == v['group'] and row['size'] == v['size'] and row['roi']==v['roi']
                radius = 8 * max(row['size']) / 1024
                base = dict(id=key, group=row['group'], truth=len(truth))
                b = {a: load_bank(tar, a, row) for a in ARMS}
                assert all(len(xy)==len(score)==len(b['pretrained'][0]) for xy,score in b.values())
                order = {a: np.argsort(-s, kind='stable') for a, (_,s) in b.items()}
                counts = {a: int(np.sum(s.astype(float)>=source['calibration'][a]['threshold']))
                          for a, (_,s) in b.items()}
                for seed in ARMS[1:]:
                    for geo in (seed, 'pretrained'):
                        xy = b[geo][0]
                        frame = base.copy()
                        for cell, count, rank in [('A','pretrained','pretrained'),('D','pretrained',seed),
                                                  ('B',seed,'pretrained'),('C',seed,seed)]:
                            frame[cell] = scored(replay, xy[order[rank][:counts[count]]], truth, radius)
                        assert frame['A']['n']==frame['D']['n'] and frame['B']['n']==frame['C']['n']
                        crosses[seed+'_'+geo].append(frame)
                ix = v['_eligible']
                for setting, quota in [('A',sum(np.asarray(v['raw'])[ix]>=locked['raw']['threshold'])),
                                       ('C',sum(np.asarray(v['learned'])[ix]>=locked['learned']['threshold'])),
                                       ('full',len(ix))]:
                    n = len(v['seeds']) + int(quota)
                    frame = base.copy()
                    for rank in ('raw','learned'):
                        frame['verifier_'+rank] = scored(replay, replay.ranked(v,rank,quota),truth,radius)
                    for geo in ARMS:
                        ranks = ['pretrained'] if geo=='pretrained' else ['pretrained',geo]
                        for rank in ranks:
                            frame[geo+'_'+rank] = scored(replay,b[geo][0][order[rank][:n]],truth,radius)
                    assert all(value['n']==n for k,value in frame.items() if k not in base)
                    pools[setting].append(frame)
            section = {}
            for name,rows in crosses.items():
                seed, geo = name.split('_')
                summary = summarize(rows)
                if seed == geo:
                    earlier = source['splits'][split]['summary'][seed+'_calibrated']['overall']
                    assert (summary['C']['matches'],summary['C']['pred']) == (earlier['tp'],earlier['pred'])
                if seed=='1729':
                    for cell, metrics in summary.items():
                        earlier = source['splits'][split]['summary'][geo+'_fixed_'+cell]['overall']
                        assert (metrics['matches'],metrics['pred'])==(earlier['tp'],earlier['pred'])
                section[name] = dict(summary=summary, crowd_count=paired(rows,'A','D'),
                                     adapted_count=paired(rows,'B','C'),frames=rows)
            results[split] = dict(crossings=section,budgets={
                k: dict(summary=summarize(rows), range=[min(r['verifier_raw']['n'] for r in rows),
                       max(r['verifier_raw']['n'] for r in rows)],frames=rows) for k,rows in pools.items()})
    assert results['test']['budgets']['full']['summary']['verifier_raw']['matches']==368
    return results


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewer',type=Path,required=True)
    parser.add_argument('--banks',type=Path,required=True)
    out = parser.add_mutually_exclusive_group(required=True)
    out.add_argument('--write',type=Path)
    out.add_argument('--check',type=Path)
    args=parser.parse_args()
    results=run(args.reviewer,args.banks)
    if args.check:
        assert results==read(args.check), 'Saved results differ from replay'
        print('Verified all seeds, both coordinate choices, exact frame budgets, and every reported match.')
    else:
        with args.write.open('x',encoding='utf8') as f:
            json.dump(results,f,allow_nan=False,separators=(',',':'))



