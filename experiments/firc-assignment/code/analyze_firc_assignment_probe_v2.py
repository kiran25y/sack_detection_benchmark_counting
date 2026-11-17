"""Independent assignment-history verification; no model, threshold search or accuracy promotion."""
import json, pathlib, hashlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
ATTEMPT='firc_assignment_probe_v2_20261106'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/'artifacts'/ATTEMPT;result=out/'collected/result'
    plan=json.loads((out/'data/plan.json').read_text());report=json.loads((result/'report.json').read_text())
    launch=json.loads((out/'launch_receipt.json').read_text());v=json.loads((out/'collected/verification.json').read_text())
    history=json.loads((result/'assignment_history.json').read_text());logs=[json.loads(s) for s in (result/'updates.jsonl').read_text().splitlines()]
    checks={'local_release':all(sha(out/k)==h for k,h in launch['release_hashes'].items()),'remote_release':v['release_hashes_match'],
      'controller_exited':not v['controller_alive'],'controller_completed':v['controller_status']['state']=='completed' and v['controller_status']['exit_code']==0 and v['controller_status']['guard_stop'] is None,
      'docker_completed':v['docker_state']['Status']=='exited' and v['docker_state']['ExitCode']==0 and not v['docker_state']['OOMKilled'],
      'plan':sha(out/'data/plan.json')==report['plan_sha256']==launch['plan_sha256'],
      'result_hashes':all(sha(result/k)==h for k,h in json.loads((result/'result_hashes.json').read_text()).items()),
      'budget':report['optimizer_calls']==13 and report['diagnostic_updates']==12 and report['saved_updates']==report['persistent_steps']==0,
      'log_calls':[r['optimizer_call'] for r in logs]==list(range(1,14)),
      'history_steps':[r['state_step'] for r in history]==list(range(13)),
      'parity_and_recovery':report['exact_checkpoint_forward_parity'] and report['exact_loss_gradient_parity'] and report['exact_nonempty_adam_recovery'],
      'worker_finite_frozen':report['finite_state'] and report['initial_encoder_sha256']==report['encoder_final_sha256']}
    details=[]
    for i,row in enumerate(plan['rows']):
        seq=[h['images'][i] for h in history]
        assert all(x['image_id']==row['image_id'] and x['truth']==row['count'] and len(x['candidate_ids'])==row['count'] and len(set(x['candidate_ids']))==row['count'] for x in seq)
        ids=np.array([x['candidate_ids'] for x in seq],dtype=int).reshape(13,row['count'])
        swaps=(ids[1:]!=ids[:-1]);per_gt=swaps.sum(0)
        d={'image_id':row['image_id'],'truth':row['count'],'switch_events':int(swaps.sum()),
           'targets_switching_at_least_once':int((per_gt>=1).sum()),'targets_switching_at_least_twice':int((per_gt>=2).sum()),
           'fraction_switching_twice':float((per_gt>=2).mean()) if row['count'] else None,
           'switches_per_target':per_gt.tolist(),'hard_counts':[s['hard_count'] for s in seq],
           'max_confidences':[s['max_q'] for s in seq]}
        details.append(d)
    checks['coverage']=len(details)==4
    motivated=any(d['truth'] in (21,34) and d['fraction_switching_twice']>=.10 for d in details)
    analysis={'checks':checks,'complete':all(checks.values()),'per_image':details,'assignment_stability_comparison_justified':all(checks.values()) and motivated,
      'rule':'At least 10% of targets in either higher-count fitting image switch assigned candidate at least twice across 12 genuine updates. This motivates a causal matched continuation, not a claim churn caused the failure.',
      'full_training_ready':False,'deployable_accuracy_established':False,'limits':'One checkpoint, four fitting images. All 13 optimizer calls disposable. Exact model/Adam recovery includes one replay reference call. No development or reserved evaluation.'}
    with (out/'analysis.json').open('x') as f:json.dump(analysis,f,indent=2)
    print(json.dumps(analysis,indent=2))
if __name__=='__main__':main()
