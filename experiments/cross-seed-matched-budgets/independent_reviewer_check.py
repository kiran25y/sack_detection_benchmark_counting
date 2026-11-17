"""Independent CPU verification of the reviewer's fixed-count and candidate-budget tests."""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile
from zipfile import ZipFile

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def measure(xy, truth, radius):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    # Minimum-cost rectangular assignment with binary costs maximizes valid matches.
    cost = cdist(xy, truth) > radius
    i, j = linear_sum_assignment(cost.astype(np.uint8))
    return dict(n=len(xy), tp=int((~cost[i, j]).sum()), ae=abs(len(xy) - len(truth)))


def ordering(scores):
    return np.lexsort((np.arange(len(scores)), -np.asarray(scores, dtype=float)))


def check(package):
    with ZipFile(package) as archive:
        def read(name):
            return json.loads(archive.read(name))
        expected = read('cross_seed_results.json')['test']
        labels = read('labels/test.json')
        sparse = read('banks/test.json')
        info = {r['id']: r for r in read('full_factorial/design.json')['rows'] if r['split']=='test'}
        cuts = read('full_factorial/analysis.json')['calibration']
        verifier_cut = read('locked_protocol.json')['calibration']
        ids = read('full_factorial/capture_verified.json')['report']['hashes']
        tar = tarfile.open(fileobj=BytesIO(archive.read('full_factorial/banks.tar')))
    assert labels.keys()==sparse.keys()==info.keys() and len(labels)==51
    seeds = ('1729', '1730', '1731')
    crosses = {s:{c: {} for c in 'ADBC'} for s in seeds}
    retained = {k:{} for k in ('A','C','full')}
    checks = 0
    with tar:
        for key, annotation in labels.items():
            truth = np.asarray(annotation, dtype=float).reshape(-1,2)
            frame = info[key]
            radius = 8 * max(frame['size']) / 1024
            xy, rank, count = {}, {}, {}
            for model in ('pretrained', *seeds):
                name = f'{model}_{key}.npz'
                payload = tar.extractfile(name).read()
                assert hashlib.sha256(payload).hexdigest()==ids[name]
                with np.load(BytesIO(payload),allow_pickle=False) as bank:
                    xy[model] = (bank['xy'] + .5) / bank['scale'] + np.asarray(frame['roi'][:2]) - .5
                    score = bank['score'].astype(float)
                rank[model] = ordering(score)
                count[model] = np.count_nonzero(score>=cuts[model]['threshold'])
            for seed in seeds:
                cells = [('A','pretrained','pretrained'),('D','pretrained',seed),('B',seed,'pretrained'),('C',seed,seed)]
                for cell,budget,score in cells:
                    crosses[seed][cell][key] = measure(xy[seed][rank[score][:count[budget]]],truth,radius)
            v = sparse[key]
            seeds_xy = np.asarray(v['seeds'],dtype=float).reshape(-1,2)
            extra = np.asarray(v['points'],dtype=float)
            eligible = np.flatnonzero((cdist(extra,seeds_xy)>radius).all(axis=1))
            raw,learned = (np.asarray(v[k],dtype=float)[eligible] for k in ('raw','learned'))
            for k,q in [('A',int(sum(raw>=verifier_cut['raw']['threshold']))),
                        ('C',int(sum(learned>=verifier_cut['learned']['threshold']))),('full',len(eligible))]:
                cap = len(seeds_xy)+q
                selected = np.vstack([seeds_xy, extra[eligible[ordering(learned)[:q]]]])
                result = {'verifier_learned':measure(selected,truth,radius)}
                for seed in seeds:
                    result[f'{seed}_{seed}'] = measure(xy[seed][rank[seed][:cap]],truth,radius)
                assert all(v['n']==cap for v in result.values())
                retained[k][key] = result

    summary = {}
    for seed,cells in crosses.items():
        reference = {r['id']:r for r in expected['crossings'][seed+'_'+seed]['frames']}
        for cell, rows in cells.items():
            assert all(result==reference[key][cell] for key,result in rows.items())
            checks += len(rows)
        summary[seed] = {}
        for name,before,after in [('crowd_counts','A','D'),('adapted_counts','B','C')]:
            changes = Counter()
            for key in labels:
                a,b = cells[before][key],cells[after][key]
                assert a['n']==b['n']
                changes[info[key]['group']] += b['tp']-a['tp']
            summary[seed][name] = dict(gain=sum(changes.values()),by_group=changes)
    budgets = {}
    for k,frames in retained.items():
        reference = {r['id']:r for r in expected['budgets'][k]['frames']}
        totals = Counter()
        for key,methods in frames.items():
            for name,m in methods.items():
                assert m==reference[key][name]
                totals[name] += m['tp']
                checks += 1
        budgets[k] = dict(candidates=sum(m['verifier_learned']['n'] for m in frames.values()),matches=totals)
    return dict(frames=len(labels),truth=sum(map(len,labels.values())),recording_groups=len(set(r['group'] for r in info.values())),
                verified_model_frame_arrays=4*len(labels),verified_frame_method_cells=checks,training_updates=0,
                independent_matching='minimum-cost rectangular assignment on a binary spatial-distance matrix',
                seed_rank_gains=summary,matched_candidate_budgets=budgets)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',type=Path)
    result=check(parser.parse_args().package)
    print(json.dumps(result,indent=2))
