"""Independently replay the frozen decisions, then score both recording splits."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from analyze_firc_tiny import evaluate
from score_sack_auxiliary import HELD, OUT as PRIOR, ROOT, partition, read


OUT = ROOT/'artifacts/sack_candidate_refinement_v1_20261110'


def main():
    assert read(OUT/'verification.json')['passed']
    plan = read(OUT/'plan.json')
    saved = read(OUT/'collected/predictions.json')
    calibration = read(OUT/'collected/freeze.json')['calibration']
    parent = read(PRIOR/'predictions.json')
    historic = read(PRIOR/'result.json')['results']
    truth = {r['image_id']:r['points'] for r in read(ROOT/'artifacts/ipic_transfer_pilot_data_v1_20261104/sack_manifest.json') if r['split']=='development'}
    truth.update({r['id']:r['points'] for r in read(HELD/'final_score/locked_labels.json')['review']['frames']})
    assert set(truth)==set(saved)==set(parent)
    results, checks = {}, {}
    for split in ('development', 'test'):
        records = {}
        for row in plan['rows']:
            if row['split']!=split: continue
            k = row['id']
            sets = saved[k]['predictions']
            base = parent[k]['verifier']
            joint = parent[k]['both']
            extra = joint[len(base):]
            assert sets['verifier']==base and sets['joint']==joint
            scale = 1024/max(row['size'])
            for arm in plan['arms']:
                e = saved[k]['estimates'][arm]
                assert len(e['scores'])==len(e['shifted'])==len(extra) and np.isfinite(e['scores']).all()
                selected = list(base)
                original = list(base)
                for i, (score, shifted) in enumerate(zip(e['scores'], e['shifted'])):
                    assert np.isfinite(shifted).all() and np.linalg.norm(np.subtract(shifted,extra[i]))*scale<=16.001
                    point = shifted if arm=='refined' else extra[i]
                    if score<calibration[arm]['best']['threshold']: continue
                    if any(np.linalg.norm(np.subtract(point,p))*scale<=8 for p in selected): continue
                    selected.append(point)
                    original.append(extra[i])
                assert selected==sets[arm]
                if arm=='refined':
                    assert original==sets['same_selected_unmoved'] and sets['equal_budget_raw']==joint[:len(selected)]
            for mode, points in sets.items():
                xy = np.asarray(points,float).reshape(-1,2)
                gt = np.asarray(truth[k],float).reshape(-1,2)
                m = evaluate(xy,gt,dict(image_id=k, width=row['size'][0],height=row['size'][1],dependency_group=row['group'],roi_xyxy=row['roi']))
                d = np.linalg.norm(xy[:,None]-gt[None],axis=2)*scale
                hits = 0
                if d.size:
                    i,j = linear_sum_assignment(np.where(d<=8,d/8,max(d.shape)+1))
                    hits = int((d[i,j]<=8).sum())
                assert hits==m['tp']
                records.setdefault(mode,[]).append(m)
        scores = {k:dict(**partition(v),per_frame=v) for k,v in records.items()}
        for k,old in [('joint','both'),('verifier','verifier')]:
            assert scores[k]==historic[split][old]
        p = scores['refined']
        c = {}
        for name in ('verifier','joint','confidence_only'):
            other = scores[name]
            c['lower_mae_vs_'+name] = p['overall']['mae']<other['overall']['mae']
            c['higher_f1_vs_'+name] = p['overall']['f1_at8']>other['overall']['f1_at8']
        for k in ('same_selected_unmoved','equal_budget_raw'):
            assert p['overall']['mae']==scores[k]['overall']['mae']
            c['more_matches_vs_'+k] = p['overall']['tp']>scores[k]['overall']['tp']
        c['empty_zero'] = p['empty']['fp']==0
        c['low_matches_retained'] = p['low']['tp']>=scores['joint']['low']['tp']
        c['low_mae_protected'] = p['low']['mae']<=scores['verifier']['low']['mae']
        for game in ('game1','game2'):
            c[game+'_protected'] = p['overall'][game]<=scores['verifier']['overall'][game]
        results[split],checks[split] = scores,c
    with (OUT/'result.json').open('x',encoding='utf8') as f:
        json.dump(dict(results=results,checks=checks,passed=all(all(v.values()) for v in checks.values())),f,indent=2,allow_nan=False)
    for split,methods in results.items():
        for name,v in methods.items():
            print(split,name,*(round(v['overall'][q],4) for q in ('mae','tp','f1_at8','fp','game1','game2')),v['low']['tp'],v['empty']['fp'])
    print(checks)


if __name__=='__main__':
    main()
