"""Independent fixed-inference metrics and matched-budget verification for the two LR arms."""
import hashlib,json,math,pathlib
import numpy as np
from analyze_firc_tiny import evaluate,summarize
ROOT=pathlib.Path(__file__).resolve().parents[1]
ATTEMPT='firc_lr_continuation_v1_20261106'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(v):
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    if isinstance(v,list):return all(finite(x) for x in v)
    return not isinstance(v,(int,float)) or math.isfinite(v)
def main():
    out=ROOT/'artifacts'/ATTEMPT;result=out/'collected/result'
    plan=json.loads((out/'data/plan.json').read_text());report=json.loads((result/'report.json').read_text())
    launch=json.loads((out/'launch_receipt.json').read_text());v=json.loads((out/'collected/verification.json').read_text())
    logs=[json.loads(s) for s in (result/'updates.jsonl').read_text().splitlines()]
    checks={'local_release':all(sha(out/k)==h for k,h in launch['release_hashes'].items()),'remote_release':v['release_hashes_match'],
      'metric_helper_frozen':sha(ROOT/'scripts/analyze_firc_tiny.py')==launch['release_hashes']['analysis/analyze_firc_tiny.py'],
      'controller_exited':not v['controller_alive'],'controller_completed':v['controller_status']['state']=='completed' and v['controller_status']['exit_code']==0 and v['controller_status']['guard_stop'] is None,
      'docker_completed':v['docker_state']['Status']=='exited' and v['docker_state']['ExitCode']==0 and not v['docker_state']['OOMKilled'],
      'plan':sha(out/'data/plan.json')==report['plan_sha256']==launch['plan_sha256'],
      'result_hashes':all(sha(result/k)==h for k,h in json.loads((result/'result_hashes.json').read_text()).items()),
      'budget':report['optimizer_calls']==report['saved_updates']==300 and report['query_presentations']==2400,
      'log_calls':[r['step'] for r in logs]==list(range(1,301)),'finite_logs':finite(logs),
      'presentations':all(r['query_presentations']==8 and len(r['bundles'])==2 and all(len(b)==4 for b in r['bundles']) for r in logs),
      'worker_frozen_encoder':report['encoder_initial_sha256']==report['encoder_final_sha256'],
      'source_checkpoint':sha(out/'data/initial.pt')==report['initial_checkpoint_sha256']==plan['initial_checkpoint_sha256']}
    metrics={};control=json.loads((out/'data/control/report.json').read_text())
    for arm in plan['arms']:
        a=report['arms'][arm];al=[r for r in logs if r['arm']==arm]
        checks[arm+'_budget']=a['saved_updates']==a['optimizer_calls']==150 and a['matching_calls']==1208 and [x['arm_step'] for x in al]==list(range(1,151))
        checks[arm+'_finite_state']=a['finite_model_adam'] and a['completed']
        checks[arm+'_initial_identity']=a['initial_adam_exact'] and a['initial_head_sha256']==control['head_final_sha256']
        checks[arm+'_checkpoint_hash']=sha(result/arm/'diagnostic.pt')==a['checkpoint_sha256']
        checks[arm+'_initial_arrays']=True
        for e,b in zip(a['initial_predictions'],control['final_predictions']):
            with np.load(result/arm/e['file'],allow_pickle=False) as x,np.load(out/'data/control'/b['file'],allow_pickle=False) as y:
                checks[arm+'_initial_arrays'] &= x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
        checks[arm+'_dynamic_used']=all(x['used_candidate_ids']==[s['candidate_ids'] for s in x['signals_before_update']] for x in al)
        checks[arm+'_loss']=all(math.isclose(z['total'],z['focal']+5*z['point'],rel_tol=2e-6,abs_tol=2e-6) and z['matched']==row['count'] for x in al for b in x['bundles'] for z,row in zip(b,plan['rows']))
        checks[arm+'_schedule']=all(x['lr']==2e-4*(i/15 if i<=15 else .01+.99*(1+math.cos(math.pi*(i-15)/135))/2)*plan['lr_scales'][arm] for i,x in enumerate(al,1))
        details=[]
        assert [e['image_id'] for e in a['final_predictions']]==[r['image_id'] for r in plan['rows']]
        for e,r in zip(a['final_predictions'],plan['rows']):
            assert e['sha256']==sha(result/arm/e['file'])
            with np.load(result/arm/e['file'],allow_pickle=False) as z:
                xy,q,gt=z['points'],z['probabilities'],z['truth_native'];assert np.isfinite(xy).all() and np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
                assert np.array_equal(gt,np.asarray(r['points']).reshape(-1,2));t=e['transform']
                assert (xy>=0).all() and (xy[:,0]<=t['valid_hw'][1]).all() and (xy[:,1]<=t['valid_hw'][0]).all()
                pred=xy[q>=.5].astype(float)/np.asarray(t['scale_xy'])+np.asarray(t['roi'][:2])-.5
                m=evaluate(pred,gt,r);m['maximum_confidence']=float(q.max());details.append(m)
        metrics[arm]={'per_image':details,'summary':summarize(details)}
    probe=ROOT/'artifacts/firc_assignment_probe_v2_20261106/collected/result'
    old=[json.loads(s) for s in (probe/'updates.jsonl').read_text().splitlines()][1:]
    actual=[x for x in logs if x['arm']=='reference_lr'][:12]
    checks['dependent_probe_replay']=all(x['bundles']==y['bundles'] and x['signals_before_update']==y['signals_before_update'] and x['lr']==y['lr'] for x,y in zip(actual,old))
    absolute={}
    for arm,m in metrics.items():
        s=m['summary'];single=next(r for r in m['per_image'] if r['truth']==1)
        absolute[arm]={'engineering_complete':all(checks.values()),'mae':s['mae']<=1,'pooled_f1':s['pooled_f1_at8']>=.8,'positive_macro_f1':s['positive_macro_f1_at8']>=.8,'empty_fp':s['empty_false_positives']==0,'single_sack':single['ae']==0}
    c=metrics['reference_lr'];p=metrics['lower_lr'];protections={};deltas=[]
    for x,y in zip(p['per_image'],c['per_image']):
        delta={k:x[k]-y[k] for k in ['count','ae','game1','game2','tp','fp','fn']};delta['f1']=x['f1']-y['f1'] if x['f1'] is not None and y['f1'] is not None else None
        deltas.append({'image_id':x['image_id'],'lower_minus_reference':delta})
        for k in ['ae','game1','game2']:protections[x['image_id']+'_'+k]=x[k]<=y[k]
        if x['truth']==1:protections[x['image_id']+'_f1']=x['f1']>=y['f1']
    comparison={'mae_gain':c['summary']['mae']-p['summary']['mae']>=1.,'positive_f1_gain':p['summary']['positive_macro_f1_at8']-c['summary']['positive_macro_f1_at8']>=.10,'all_protections':all(protections.values()),'primary_absolute':all(absolute['lower_lr'].values())}
    candidate='lower_lr' if all(absolute['lower_lr'].values()) and all(protections.values()) else ('reference_lr' if all(absolute['reference_lr'].values()) else None)
    result_data={'checks':checks,'metrics':metrics,'absolute_gate_components':absolute,'absolute_passed':{k:all(v.values()) for k,v in absolute.items()},'comparison_components':comparison,'comparison_passed':all(comparison.values()),'protections':protections,'deltas':deltas,'ordinary_pilot_candidate':candidate,'full_training_started':False,'paper_final_accuracy':None,'limits':'Four repeatedly fitted images, seed1729, no development or reserved scoring. Authentic Adam continuation from same150-step diagnostic. This is a standard LR engineering test, not novelty or generalization.'}
    with (out/'analysis.json').open('x') as f:json.dump(result_data,f,indent=2)
    print(json.dumps(result_data,indent=2))
if __name__=='__main__':main()
