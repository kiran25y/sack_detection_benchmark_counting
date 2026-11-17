"""Independent saved-output calibration, count and localization analysis."""
import json,math,pathlib
import numpy as np
from analyze_firc_tiny import sha,evaluate,summarize
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def main():
    out=ROOT/'artifacts/firc_ordinary_evaluation_v1_20261106';result=out/'collected/result';p=read(out/'data/plan.json');r=read(result/'report.json')
    launch=read(out/'launch_receipt.json');v=read(out/'collected/verification.json');f=read(result/'threshold_frozen.json');c,d=v['controller_status'],v['docker_state']
    truth=read(ROOT/p['local_truth_manifest']);by_id={row['image_id']:row for row in truth};events=[json.loads(s) for s in (result/'events.jsonl').read_text().splitlines()]
    checks={'controller_completed':not v['controller_alive'] and c['state']=='completed' and c['exit_code']==0 and c['guard_stop'] is None,
      'docker_completed':d['Status']=='exited' and d['ExitCode']==0 and not d['OOMKilled'],
      'local_release':all(sha(out/k)==h for k,h in launch['release_hashes'].items()),'remote_release':v['release_hashes_match'],
      'analyzer_frozen':sha(pathlib.Path(__file__))==launch['release_hashes']['analysis/analyze_firc_ordinary_evaluation.py'],
      'helper_frozen':sha(ROOT/'scripts/analyze_firc_tiny.py')==launch['release_hashes']['analysis/analyze_firc_tiny.py'],
      'plan':sha(out/'data/plan.json')==launch['plan_sha256']==r['plan_sha256'],
      'result_hashes':all(sha(result/k)==h for k,h in read(result/'result_hashes.json').items()),
      'checkpoint_identity':sha(out/'data/adaptation.pt')==p['checkpoint_sha256']==r['checkpoint_sha256'],
      'unchanged_model':r['head_initial_sha256']==r['head_final_sha256']==p['head_sha256'] and r['encoder_initial_sha256']==r['encoder_final_sha256']==p['encoder_sha256'],
      'budget':r['state']=='completed' and r['query_forwards']==65 and r['reference_image_encodings']==2 and r['optimizer_calls']==r['persistent_steps']==0,
      'truth_identity':sha(ROOT/p['local_truth_manifest'])==p['truth_manifest_sha256'],
      'labels_absent':all('points' not in row and 'count' not in row for row in p['rows']),
      'counts_train_only':read(out/'data/training_counts.json')=={row['image_id']:row['count'] for row in truth if row['split']=='train'},
      'threshold_receipt':sha(result/'threshold_frozen.json')==r['threshold_receipt_sha256'] and sha(out/'data/training_counts.json')==f['training_count_sha256'],
      'coverage':len(r['predictions'])==len({e['image_id'] for e in r['predictions']})==65 and {e['image_id'] for e in r['predictions']}==set(by_id),
      'chronology':[e['sequence'] for e in events]==list(range(1,67)) and events[42]['event']=='threshold_frozen' and all(e.get('split')=='train' for e in events[:42]) and all(e.get('split')=='development' for e in events[43:]),
      'worker_finite':r['finite_model']}
    arrays={}
    for e in r['predictions']:
        row=by_id[e['image_id']];t=e['transform'];assert sha(result/e['file'])==e['sha256'] and t['roi']==row['roi_xyxy'] and e['split']==row['split']
        rw,rh=t['roi'][2]-t['roi'][0],t['roi'][3]-t['roi'][1];w,h=round(rw*896/max(rw,rh)),round(rh*896/max(rw,rh))
        assert t['valid_hw']==[h,w] and t['padded_hw']==[math.ceil(h/112)*112,math.ceil(w/112)*112] and t['scale_xy']==[w/rw,h/rh]
        with np.load(result/e['file'],allow_pickle=False) as z:
            xy,q=z['points'].copy(),z['probabilities'].copy();assert set(z.files)=={'points','probabilities'}
        assert xy.shape==(len(q),2) and len(q)==len(np.arange(2,w,4))*len(np.arange(2,h,4))
        assert np.isfinite(xy).all() and np.isfinite(q).all() and ((q>=0)&(q<=1)).all() and (xy>=0).all() and (xy[:,0]<=w).all() and (xy[:,1]<=h).all()
        arrays[e['image_id']]=(xy,q,t)
    scores=[(sum(abs(int((arrays[row['image_id']][1]>=threshold).sum())-row['count']) for row in truth if row['split']=='train'),threshold) for threshold in p['threshold_grid']]
    error,tau=min(scores,key=lambda pair:(pair[0],-pair[1]))
    checks['training_selection']=tau==f['selection']['threshold'] and error/42==f['selection']['training_mae'] and f['selection']['grid_scores']==[{'threshold':t,'total_absolute_error':e} for e,t in scores]
    checks['training_hashes_before_freeze']=f['training_prediction_hashes']=={e['image_id']:e['sha256'] for e in r['predictions'] if e['split']=='train'} and f['query_forwards']==42 and f['development_forwards']==0
    metrics={}
    for mode,threshold in [('fixed',.5),('calibrated',tau)]:
        per=[]
        for row in truth:
            xy,q,t=arrays[row['image_id']];pred=xy[q>=threshold].astype(float)/np.asarray(t['scale_xy'])+np.asarray(t['roi'][:2])-.5
            m=evaluate(pred,np.asarray(row['points'],float).reshape(-1,2),row);m['split']=row['split'];per.append(m)
        tr=[m for m in per if m['split']=='train'];dev=[m for m in per if m['split']=='development']
        metrics[mode]={'threshold':threshold,'training':summarize(tr),'development':summarize(dev),'low':summarize([m for m in dev if 1<=m['truth']<=11]),'higher':summarize([m for m in dev if m['truth']>=12]),
          'groups':{g:summarize([m for m in dev if m['group']==g]) for g in sorted({m['group'] for m in dev})},'positive_frames_tp':sum(m['truth']>0 and m['tp']>0 for m in dev),'per_image':per}
    m=metrics['calibrated'];s=m['development'];b=metrics['fixed'];gate={'engineering':all(checks.values()),'mae':s['mae']<=7.71739130435,'pooled_f1':s['pooled_f1_at8']>=.5,'positive_macro_f1':s['positive_macro_f1_at8']>=.5,
      'positive_frames_tp':m['positive_frames_tp']>=14,'training_positive_macro_f1':m['training']['positive_macro_f1_at8']>=.8,'low_mae':m['low']['mae']<=2,'low_f1':m['low']['positive_macro_f1_at8']>=.5,'empty_fp':s['empty_false_positives']==0}
    delta={scope:{k:m[scope][k]-b[scope][k] for k in ['mae','rmse','bias','pooled_f1_at8','positive_macro_f1_at8','game1','game2','empty_false_positives']} for scope in ['training','development','low','higher']}
    output={'engineering_complete':all(checks.values()),'checks':checks,'metrics':metrics,'calibrated_minus_fixed':delta,'readiness_components':gate,'readiness_passed':all(gate.values()),'reserved_scored':0,'final_deployable_accuracy':None,'limits':p['limitations']}
    with (out/'analysis.json').open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({'engineering_complete':output['engineering_complete'],'threshold':tau,'development':s,'readiness':gate,'readiness_passed':output['readiness_passed']},indent=2))
if __name__=='__main__':main()
