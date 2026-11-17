"""Independent saved-prediction scoring on the unchanged reviewed11 only."""
import datetime,hashlib,json,math
from pathlib import Path
import numpy as np
from verify_ipic_assignment_continuation import calculate
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/ipic_pretrained_reference_v1_20261104';COL=OUT/'collected'
OLD=ROOT/'artifacts/ipic_sack_data_admission_v1_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def finite(x):
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return math.isfinite(x) if isinstance(x,(int,float)) else True
def metrics(points,truth):
 p=np.asarray(points,float).reshape(-1,2);gt=np.asarray(truth['points'],float).reshape(-1,2)
 return dict(query_id=truth['image_id'],dependency_group=truth['dependency_group'],**calculate(dict(points=p,truth=gt,probabilities=np.ones(len(p)),valid=np.ones(len(p),bool),width=truth['width'],height=truth['height'])))
def summary(rows):
 def agg(rs):
  if not rs:return None
  tp=sum(r['point_8']['TP'] for r in rs);fp=sum(r['point_8']['FP'] for r in rs);fn=sum(r['point_8']['FN'] for r in rs)
  return dict(n=len(rs),MAE=float(np.mean([r['AE'] for r in rs])),RMSE=float(np.sqrt(np.mean([r['error']**2 for r in rs]))),bias=float(np.mean([r['error'] for r in rs])),GAME1=float(np.mean([r['GAME1'] for r in rs])),GAME2=float(np.mean([r['GAME2'] for r in rs])),macro_F1_8=float(np.mean([r['point_8']['F1'] for r in rs])),pooled_F1_8=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,TP=tp,FP=fp,FN=fn)
 return dict(all=agg(rows),positive=agg([r for r in rows if r['truth']>0]),empty=agg([r for r in rows if r['truth']==0]),low_1_11=agg([r for r in rows if 1<=r['truth']<=11]),higher_12_plus=agg([r for r in rows if r['truth']>=12]),per_image=rows)
def main():
 if (OUT/'analysis.json').exists():raise FileExistsError('Reuse saved analysis')
 v=read(COL/'collection_verification.json');assert v['all_current_identities_match'] and all(v['current_identity_checks'].values())
 assert v['controller_exited'] and v['controller']['state']=='completed' and v['controller_exit']['exit_code']==0
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==sha(COL/'protocol.json')
 assert p['max_model_calls']==58 and not p['truth_transferred']
 for name,h in read(OUT/'input_hashes.json').items():assert sha(OUT/name)==h,name
 assert read(OUT/'release_hashes.json')==read(COL/'release_hashes.json')
 inputs={r['query_id']:r for r in read(OUT/'inputs.json')};total=0;checks=[];banks={}
 for stage,calls in p['calls'].items():
  folder=COL/stage;rep=read(folder/'report.json');state=v['stages'][stage]['docker'];exitreceipt=read(COL/(stage+'_exit.json'))
  assert state['Status']=='exited' and state['ExitCode']==0 and not state['OOMKilled']
  assert exitreceipt['ExitCode']==0 and not exitreceipt['OOMKilled']
  assert not (COL/(stage+'_guard_stop.json')).exists()
  assert rep['state']=='completed' and rep['calls']==calls and rep['optimizer_steps']==rep['persistent_steps']==0
  assert rep['parameters_unmodified'] and rep['finite_parameters'] and rep['initial_state_sha256']==rep['final_state_sha256']
  assert rep['protocol_sha256']==sha(OUT/'protocol.json')
  for name,h in read(folder/'result_hashes.json').items():assert sha(folder/name)==h
  records=lines(folder/'predictions.jsonl');assert len(records)==calls and finite(records);total+=calls
  if stage.endswith('bank'):
   assert len(records)==23 and {r['query_id'] for r in records}==set(inputs)
   for r in records:
    q=inputs[r['query_id']];assert r['roi_xyxy']==q['roi_xyxy'];bb=np.asarray(r['boxes'],float).reshape(-1,4);ss=np.asarray(r['scores'],float);assert ss.shape==(len(bb),)
    centers=(bb[:,:2]+bb[:,2:])/2;x0,y0,x1,y1=q['roi_xyxy'];keep=(centers[:,0]>=x0)&(centers[:,0]<x1)&(centers[:,1]>=y0)&(centers[:,1]<y1)
    assert keep.tolist()==r['roi_keep'];np.testing.assert_array_equal(centers[keep],np.asarray(r['points']).reshape(-1,2))
   banks[stage.removesuffix('_bank')]={r['query_id']:r for r in records}
  else:
   bykey={(r['query_id'],r['kind']):[] for r in records}
   for r in records:bykey[(r['query_id'],r['kind'])].append(r)
   for (qid,kind),rs in bykey.items():
    if kind=='external':
     assert len(rs)==2
     for key in ['boxes','scores','points','transformed_boxes']:np.testing.assert_allclose(rs[0][key],rs[1][key],rtol=1e-5,atol=1e-6)
   if stage.startswith('geco'):
    for q in read(OUT/'compatibility_inputs.json'):
     for key in ['boxes','scores']:np.testing.assert_allclose(bykey[(q['query_id'],'native')][0][key],bykey[(q['query_id'],'identity')][0][key],rtol=1e-5,atol=1e-6)
  checks.append(dict(stage=stage,calls=calls,backbone_forward_calls=rep['backbone_forward_calls'],seconds_shared_GPU=rep['elapsed_seconds_shared_GPU'],unchanged_parameters=True))
 assert total==58
 expected='a69c8f976172c3315135feb082c4e11fac26669b53dd56286862b2c80847c1fd';assert sha(OLD/'development_manifest.json')==expected
 truth=read(OLD/'development_manifest.json');assert len(truth)==11 and sum(t['count'] for t in truth)==244
 for t in truth:assert t['encoded_sha256']==inputs[t['image_id']]['image_sha256'] and t['roi_xyxy']==inputs[t['image_id']]['roi_xyxy'] and t['exhaustive_reviewed']
 results={m:summary([metrics(bank[t['image_id']]['points'],t) for t in truth]) for m,bank in banks.items()}
 prior={};priorhashes={}
 for arm in 'ABCDEF':
  folder=ROOT/'artifacts/ipic_sack_screen_v1_20261103/collected/result'/arm;path=folder/'development_predictions_1000.jsonl';h=sha(path);assert h==read(folder/'result_hashes.json')[path.name];priorhashes[arm]=h
  records={r['query_id']:r for r in lines(path)};assert set(records)=={t['image_id'] for t in truth}
  for threshold in [.5,.1]:
   rs=[]
   for t in truth:
    r=records[t['image_id']];assert r['truth']==t['points'] and r['count']==t['count'] and (r['width'],r['height'])==(t['width'],t['height'])
    pt=np.asarray(r['points']).reshape(-1,2);pr=np.asarray(r['probabilities']);rs.append(metrics(pt[pr>=threshold],t))
   prior[f'{arm}_t{threshold:.2f}']=summary(rs)
 # Separate count-only arithmetic check, independent of the point-matching utility.
 for m,b in banks.items():
  err=[len(b[t['image_id']]['points'])-t['count'] for t in truth]
  assert abs(sum(abs(e) for e in err)/11-results[m]['all']['MAE'])<1e-12
  assert abs(math.sqrt(sum(e*e for e in err)/11)-results[m]['all']['RMSE'])<1e-12
 result=dict(verified_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),engineering_passed=True,checks=checks,model_calls=total,optimizer_steps=0,
  coverage_per_model=23,scored_queries=11,scored_true_count=244,full23_metrics_withheld=True,unscored_new_queries=[q for q in inputs if q not in {t['image_id'] for t in truth}],
  results=results,reused_prior_IPIC_endpoints=prior,reused_prediction_hashes=priorhashes,source_label_sha256=expected,
  caveat='Original11 reused development images in one group. CountGD native external prompts; GeCo2 disclosed external adapter, not native official result. Different pretraining/capacity/target supervision/compute. Author thresholds versus fixed IPIC thresholds, no calibration or winning-arm selection. No sparse1-11 or dense/extreme conclusion; no novelty claim.',
  new_training_started=False,remaining_partial_additional_labels=25)
 with (OUT/'analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(dict(verified_kst=result['verified_kst'],checks_passed=True,results={k:{s:v for s,v in d.items() if s!='per_image'} for k,d in results.items()},prior={k:d['all'] for k,d in prior.items()}),indent=2))
if __name__=='__main__':main()
