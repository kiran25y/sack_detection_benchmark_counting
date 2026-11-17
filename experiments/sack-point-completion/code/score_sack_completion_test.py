"""Independent image-by-image scoring on frozen thresholds and group identities."""
import json
import math
from pathlib import Path

import numpy as np

from analyze_firc_tiny import evaluate, summarize
from run_sack_completion_test import CONTROLS, MANIFEST, OUT, read, save, sha


def points(raw):
    p=np.asarray(raw,dtype=float).reshape(-1,2)
    assert np.isfinite(p).all()
    return p


def group_scores(rows):
    if not rows:
        raise ValueError('Empty evaluation cohort')
    if any(r['truth'] for r in rows):
        values=summarize(rows)
    else:
        values={'empty_false_positives':sum(r['count'] for r in rows)}
    values.update(frames=len(rows),predicted=sum(r['count'] for r in rows),truth=sum(r['truth'] for r in rows),
                  retained=sum(r['retained'] for r in rows),born=sum(r['born'] for r in rows),
                  matches_with_retained_only=sum(r['retained_tp'] for r in rows))
    if values['truth']:
        values['precision']=values['tp']/values['predicted'] if values['predicted'] else 0.
        values['recall']=values['tp']/values['truth']
    return values


def main():
    freeze=read(OUT/'freeze.json')
    assert read(OUT/'verification.json')['state']=='verified'
    assert sha(MANIFEST)==freeze['manifest'] and sha(CONTROLS)==freeze['controls']
    labels=read(MANIFEST)
    baseline={r['image_id']:r for r in map(json.loads,CONTROLS.read_text().splitlines())}
    report=read(OUT/'collected/report.json')
    raw=read(OUT/'collected/predictions.json')
    pred={(r['arm'],r['image_id']):r for r in raw}
    assert len(pred)==len(raw)==130 and len(labels)==65
    sets={'march12':'ipic_dependency_483475312362','february27':'ipic_dependency_b8c3dda6893f'}
    summary, details={},{}
    for arm in ('unchanged','confidence_vote','ordinary','paired'):
        rec=[]
        for row in labels:
            ident=row['image_id']
            if arm in ('ordinary','paired'):
                observation=pred[(arm,ident)]
                p=points(observation['points'])
                retained=observation['retained']
            else:
                p=points(baseline[ident]['native_points'][arm]);retained=len(p)
            gt=points(row['points'])
            r=evaluate(p,gt,row)
            retained_tp=evaluate(p[:retained],gt,row)['tp']
            r.update(split=row['split'],retained=retained,born=len(p)-retained,retained_tp=retained_tp)
            rec.append(r)
        dev=[r for r in rec if r['split']=='development']
        fitting=[r for r in rec if r['split']=='train']
        assert len(dev)==23 and sum(r['truth'] for r in dev)==355 and len(fitting)==42
        cohorts={'fitting':fitting,'development':dev,'low':[r for r in dev if 0<r['truth']<12],
                 'empty':[r for r in dev if r['truth']==0],
                 **{name:[r for r in dev if r['group']==key] for name,key in sets.items()}}
        scores={k:group_scores(v) for k,v in cohorts.items()}
        if arm in ('ordinary','paired'):
            calibration=report['arms'][arm]['best']
            assert math.isclose(scores['fitting']['mae'],calibration['mae']) and math.isclose(scores['fitting']['positive_macro_f1_at8'],calibration['positive_f1'])
            assert scores['fitting']['empty_false_positives']==calibration['empty_fp']
        summary[arm]=scores;details[arm]=rec
    expected={'unchanged':[71,62,9],'confidence_vote':[75,68,7]}
    for arm,counts in expected.items():
        assert [summary[arm][s]['tp'] for s in ('development','march12','february27')]==counts
    p=summary['paired'];gates={}
    for arm in expected:
        ref=summary[arm]
        tests={}
        for side in sets:
            tests[side+'_matches']=p[side]['tp']>ref[side]['tp']
            tests[side+'_F1']=p[side]['pooled_f1_at8']>ref[side]['pooled_f1_at8']
        tests['low_MAE']=p['low']['mae']<ref['low']['mae']
        tests['low_matches']=p['low']['tp']>ref['low']['tp']
        tests['pooled_F1']=p['development']['pooled_f1_at8']>=ref['development']['pooled_f1_at8']+.02
        for m in ('mae','game1','game2'):
            tests[m]=p['development'][m]<=ref['development'][m]
        gates[arm]=tests
    gates['absolute']={'zero_empty':p['empty']['empty_false_positives']==0}
    gates['matched_head']={m:p['development'][m]>summary['ordinary']['development'][m] for m in ('tp','pooled_f1_at8')}
    result=dict(summary=summary,per_image=details,gates=gates,passed=all(all(v.values()) for v in gates.values()),
                verification=sha(OUT/'verification.json'),predictions=sha(OUT/'collected/predictions.json'))
    save(OUT/'result.json',result)
    for arm,s in summary.items():
        print(arm, {k:s['development'][k] for k in ('tp','fp','fn','pooled_f1_at8','mae','game1','game2')},
              'March,Feb,low,empty', [s[k]['tp'] for k in ('march12','february27','low')],s['empty']['predicted'])
    print('PASS',result['passed'])


if __name__=='__main__':
    main()
