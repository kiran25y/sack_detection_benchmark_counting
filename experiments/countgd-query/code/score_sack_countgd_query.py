"""Score sealed CountGD++ query predictions against the same native sack labels."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from analyze_firc_tiny import evaluate
from score_sack_auxiliary import HELD, OUT as JOINT, ROOT, partition, read


A=ROOT/'artifacts'
OUT=A/'sack_countgd_query_v1_20261110'
REFINER=A/'sack_candidate_refinement_v1_20261110'


def main():
    assert read(OUT/'verification.json')['passed']
    plan=read(OUT/'plan.json')
    pred=read(OUT/'collected/predictions.json')
    truth={r['image_id']:r['points'] for r in read(A/'ipic_transfer_pilot_data_v1_20261104/sack_manifest.json') if r['split']=='development'}
    truth.update({r['id']:r['points'] for r in read(HELD/'final_score/locked_labels.json')['review']['frames']})
    assert set(truth)==set(pred)
    results,decisions={},{}
    joint=read(JOINT/'result.json')['results'];refined=read(REFINER/'result.json')['results']
    for split in ('development','test'):
        scores={}
        for mode in ('native','geometry','decoder','same_selected_unmoved','equal_budget_native'):
            metrics=[]
            for r in plan['rows']:
                if r['split']!=split:continue
                k=r['id']
                assert len(pred[k]['decoder'])==len(pred[k]['same_selected_unmoved'])==len(pred[k]['equal_budget_native'])
                xy=np.asarray(pred[k][mode],float).reshape(-1,2);gt=np.asarray(truth[k],float).reshape(-1,2)
                assert np.isfinite(xy).all()
                m=evaluate(xy,gt,dict(image_id=k,width=r['size'][0],height=r['size'][1],dependency_group=r['group'],roi_xyxy=r['roi']))
                d=np.linalg.norm(xy[:,None]-gt[None],axis=2)*1024/max(r['size'])
                a,b=linear_sum_assignment(np.where(d<=8,d/8,max(d.shape)+1))
                assert int((d[a,b]<=8).sum())==m['tp']
                metrics.append(m)
            scores[mode]=dict(**partition(metrics),per_frame=metrics)
        old={**{'aux_'+k:v['overall'] for k,v in joint[split].items()},**{'refine_'+k:v['overall'] for k,v in refined[split].items()}}
        if split=='test':
            old.update({'holdout_'+k:v['overall'] for k,v in read(HELD/'final_score/result.json')['summary'].items()})
        else:
            for experiment,key in [('sack_recording_excluded_v1_20261109/recovery','per_frame'),('sack_point_completion_v1b_20261109','per_image'),('sack_context_detector_v1_20261109','per_frame')]:
                for k,records in read(A/experiment/'result.json')[key].items():
                    if isinstance(records,list) and len(records)==23 and all('group' in m and 'truth' in m and 'tp' in m and 'error' in m and 'game2' in m for m in records):
                        old[experiment+'/'+k]=partition(records)['overall']
        sources={**old,**{k:scores[k]['overall'] for k in ('geometry','native')}}
        p=scores['decoder'];v=joint[split]['verifier']
        checks=dict(lowest_mae=p['overall']['mae']<min(s['mae'] for s in sources.values()),
                    highest_f1=p['overall']['f1_at8']>max(s['f1_at8'] or 0 for s in sources.values()),
                    empty_zero=p['empty']['fp']==0,low_mae=p['low']['mae']<=v['low']['mae'],low_matches=p['low']['tp']>=v['low']['tp'],
                    game1=p['overall']['game1']<=v['overall']['game1'],game2=p['overall']['game2']<=v['overall']['game2'])
        for name in ('native','geometry'):
            checks[name+'_mae']=p['overall']['mae']<scores[name]['overall']['mae']
            checks[name+'_f1']=p['overall']['f1_at8']>scores[name]['overall']['f1_at8']
        for name in ('same_selected_unmoved','equal_budget_native'):
            checks[name+'_matches']=p['overall']['tp']>scores[name]['overall']['tp']
        results[split]=dict(methods=scores,comparators=sources)
        decisions[split]=checks
        for k,v in scores.items():
            print(split,k,*[round(v['overall'][q],4) for q in ('mae','f1_at8','tp','fp','game1','game2')],v['low']['tp'],v['empty']['fp'])
    with (OUT/'result.json').open('x',encoding='utf8') as f:
        json.dump(dict(results=results,checks=decisions,pass_all=all(all(v.values()) for v in decisions.values())),f,indent=2,allow_nan=False)
    print(decisions)


if __name__=='__main__':
    main()
