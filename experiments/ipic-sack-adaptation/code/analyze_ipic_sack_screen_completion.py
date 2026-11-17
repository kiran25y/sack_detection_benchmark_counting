"""Independent numpy/stdlib verification of final saved predictions; no model."""
import argparse,collections,datetime,hashlib,json,math
from pathlib import Path
import numpy as np
from verify_ipic_assignment_continuation import calculate,compare_objects
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def finite(v):
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 if isinstance(v,list):return all(finite(x) for x in v)
 return math.isfinite(v) if isinstance(v,(float,int)) else True
def summary(records,threshold):
 results=[]
 for r in records:
  points=np.asarray(r['points']).reshape(-1,2);probs=np.asarray(r['probabilities'])
  selected=points[probs>=threshold]
  # Existing independent spatial hashing matcher; no scipy/worker metric code.
  arrays=dict(points=selected,valid=np.ones(len(selected),bool),probabilities=np.ones(len(selected)),truth=np.asarray(r['truth']).reshape(-1,2),width=r['width'],height=r['height'])
  results.append(dict(query_id=r['query_id'],**calculate(arrays)))
 result=dict(threshold=threshold,MAE=float(np.mean([r['AE'] for r in results])),RMSE=float(np.sqrt(np.mean([r['error']**2 for r in results]))),bias=float(np.mean([r['error'] for r in results])),GAME1=float(np.mean([r['GAME1'] for r in results])),GAME2=float(np.mean([r['GAME2'] for r in results])),F1_8=float(np.mean([r['point_8']['F1'] for r in results])),rows=results)
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--attempt',default='ipic_sack_screen_v1_20261103');a=ap.parse_args()
 if not a.attempt.replace('_','').isalnum():raise ValueError('Invalid attempt')
 folder=ROOT/'artifacts'/a.attempt;col=folder/'collected';dest=folder/'completion_analysis.json'
 if dest.exists():raise ValueError('Analysis already exists; inspect and reuse')
 verification=read(col/'collection_verification.json');checks={};controller=read(col/'controller_status.json')
 checks['terminal_controller']=verification['controller_exited'] and controller['state']=='completed' and controller['completed_arms']==list('ABCDEF')
 for n,h in verification['collected_hashes'].items():assert sha(col/n)==h
 checks['collection_hashes']=True
 frozen=read(col/'frozen_hashes.json');checks['release_hashes']=verification['source_actual']==frozen
 spec=read(col/'data/screen_plan.json');dev=read(col/'data/development_manifest.json');truth={r['image_id']:r for r in dev}
 checks['frozen_plan']=sha(col/'data/screen_plan.json')=='e7f867213d671cc7d9bddddf248fe4ef4887a09b374aea04709360e9bd9603c1'
 for n,h in spec['input_hashes'].items():assert sha(col/'data'/n)==h
 draws=rows(ROOT/'artifacts/ipic_sack_pair_audit_v5_20261103/draws.jsonl');assert sha(ROOT/'artifacts/ipic_sack_pair_audit_v5_20261103/draws.jsonl')==spec['draws_sha256']
 guard=read(ROOT/'study_protocols/ipic_sack_adaptation_cuda_verified_20261103.json')
 arm_results={};initial=None
 for arm in 'ABCDEF':
  path=col/'result'/arm;status=read(path/'status.json');plan=read(path/'plan.json');exit_state=read(col/('exit_'+arm+'.json'));docker=verification['docker'][arm]
  assert exit_state['returncode']==docker['state']['ExitCode']==0 and not exit_state['guard_stop'] and not docker['state']['OOMKilled'] and docker['id']==exit_state['container_id']
  assert status['state']=='completed' and status['durable_optimizer_steps']==status['logged_updates']==1000 and status['finite_model_optimizer']
  assert plan['arm']==arm and plan['updates']==1000 and plan['seed']==1729 and plan['identities']==guard['identities'] and status['plan_sha256']==sha(path/'plan.json')
  for k,v in spec.items():assert plan[k]==v
  actual=verification['result_actual'][arm];assert read(path/'result_hashes.json')==actual and status['checkpoint_sha256']==actual['counter.pt'] and status['resume_sha256']==actual['resume.pt']
  updates=rows(path/'updates.jsonl');ledger=rows(path/'augmentation_ledger.jsonl')
  assert len(updates)==1000 and [r['update'] for r in updates]==list(range(1,1001)) and finite(updates)
  assert len(ledger)==8000 and [r['draw'] for r in ledger]==draws
  evaluations={}
  for step in [0,250,500,750,1000]:
   records=rows(path/f'development_predictions_{step:04}.jsonl');saved=read(path/f'development_{step:04}.json')
   assert len(records)==11 and len({r['query_id'] for r in records})==11 and {r['query_id'] for r in records}==set(truth) and finite(records)
   for r in records:
    t=truth[r['query_id']];assert r['truth']==t['points'] and r['count']==t['count'] and r['dependency_group']==t['dependency_group'] and r['width']==t['width'] and r['height']==t['height']
    assert len(r['points'])==len(r['probabilities']) and all(0<=v<=1 for v in r['probabilities'])
   if step==0:
    if initial is None:initial=records
    else:assert records==initial
   scores=[(float(np.mean([abs(sum(p>=t for p in r['probabilities'])-r['count']) for r in records])),-t) for t in [x/100 for x in range(10,91,5)]]
   mae,minus_t=min(scores);threshold=-minus_t
   compare_objects(saved['threshold_selection'],dict(threshold=threshold,development_MAE=mae,common_threshold=.5))
   computed={name:summary(records,t) for name,t in [('common_0.50',.5),('development_selected',threshold)]}
   for name,result in computed.items():compare_objects(saved['scores'][name],result)
   evaluations[str(step)]=computed
  arm_results[arm]=dict(final=evaluations['1000'],initial=evaluations['0'],evaluated_steps=list(evaluations),total_update_seconds=sum(r['seconds'] for r in updates),optimizer_updates=1000,augmentation_pairs=8000)
  checks['arm_'+arm]=True
 effects=[]
 for candidate,control in [('B','A'),('D','C'),('B','E'),('D','F'),('C','A'),('E','A'),('F','C')]:
  for threshold in ['common_0.50','development_selected']:
   c=arm_results[control]['final'][threshold];v=arm_results[candidate]['final'][threshold]
   effects.append(dict(candidate=candidate,control=control,threshold_policy=threshold,signed_candidate_minus_control={k:v[k]-c[k] for k in ['MAE','RMSE','bias','GAME1','GAME2','F1_8']},relative_MAE_reduction=(c['MAE']-v['MAE'])/c['MAE'] if c['MAE'] else None,per_image_AE_differences=[dict(query_id=x['query_id'],difference=y['AE']-x['AE']) for x,y in zip(c['rows'],v['rows'])]))
 checks['same_initial_predictions']=True
 assert all(checks.values())
 result=dict(state='verified_complete',client_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),checks=checks,arms=arm_results,effects=effects,scientific_gate=dict(full_gate_evaluable=False,full_gate_passed=False,missing='independently verified physical intervention pairs and independent final test',natural_5percent_components=[dict(candidate=e['candidate'],control=e['control'],threshold_policy=e['threshold_policy'],passed=e['relative_MAE_reduction'] is not None and e['relative_MAE_reduction']>=.05) for e in effects if (e['candidate'],e['control']) in [('B','A'),('D','C')]],dense_extreme_evaluable=False,confirmatory_training_authorized=False),limitations='One recording group per split; all 11 development frames are dependent. No physical-pair accuracy, population MAE, dense/extreme sack gain, novel mechanism or independent final generalization established. Finite tensors checked by frozen worker; local verification uses hashes/logs/predictions without model execution.')
 with dest.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
 print(json.dumps(dict(state=result['state'],checks=checks,effects=effects,scientific_gate=result['scientific_gate']),indent=2))
if __name__=='__main__':main()
