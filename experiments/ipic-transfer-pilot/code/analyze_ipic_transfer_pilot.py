"""Independent saved-output verification. Numpy/stdlib only, no model execution."""
import argparse,datetime,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from counting.ipic.transfer_protocol import CELLS,gate,validate_manifest
from verify_ipic_assignment_continuation import calculate,compare_objects
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(v):
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 if isinstance(v,list):return all(finite(x) for x in v)
 return math.isfinite(v) if isinstance(v,(float,int)) else True
def aggregate(rows):
 if not rows:return None
 return dict(queries=len(rows),MAE=float(np.mean([r['AE'] for r in rows])),RMSE=float(np.sqrt(np.mean([r['error']**2 for r in rows]))),bias=float(np.mean([r['error'] for r in rows])),GAME1=float(np.mean([r['GAME1'] for r in rows])),GAME2=float(np.mean([r['GAME2'] for r in rows])),F1_8=float(np.mean([r['point_8']['F1'] for r in rows])),FP_total=sum(r['point_8']['FP'] for r in rows),rows=rows)
def summarize(records,threshold):
 rows=[]
 for r in records:
  p=np.asarray(r['points'],float).reshape(-1,2);prob=np.asarray(r['probabilities'],float);selected=p[prob>=threshold]
  arrays=dict(points=selected,probabilities=np.ones(len(selected)),valid=np.ones(len(selected),bool),truth=np.asarray(r['truth'],float).reshape(-1,2),width=r['width'],height=r['height'])
  rows.append(dict(query_id=r['query_id'],dependency_group=r['dependency_group'],**calculate(arrays)))
 return dict(all=aggregate(rows),zero=aggregate([r for r in rows if r['truth']==0]),low_1_11=aggregate([r for r in rows if 1<=r['truth']<=11]),higher_12_plus=aggregate([r for r in rows if r['truth']>=12]),groups={g:aggregate([r for r in rows if r['dependency_group']==g]) for g in sorted({r['dependency_group'] for r in rows})})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--collected',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();col=a.collected
 if a.output.exists():raise ValueError('Analysis exists; inspect and reuse')
 ver=read(col/'collection_verification.json');controller=read(col/'controller_status.json');assert ver['controller_exited'] and controller['state']=='completed' and controller['completed_arms']==list(CELLS)
 frozen=read(col/'frozen_hashes.json');assert ver['source_actual']==frozen
 for n,h in ver['collected_hashes'].items():assert sha(col/n)==h
 spec=read(col/'data/pilot_plan.json');assert spec['cells']==CELLS and spec['updates_per_cell']==200 and spec['total_updates']==800 and not spec['threshold_selection']
 for n,h in spec['input_hashes'].items():
  # Collection may omit large unchanged image payloads; remote verification must include them.
  assert ver['source_actual']['data/'+n]==h
  if (col/'data'/n).exists():assert sha(col/'data'/n)==h
 validate_manifest(read(col/'data/sack_manifest.json'),read(col/'data/approved_group_allocation.json'))
 truth={r['image_id']:r for r in read(col/'data/development_manifest.json')};results={};initial=None;checks={}
 for cell in CELLS:
  folder=col/'result'/cell;status=read(folder/'status.json');plan=read(folder/'plan.json');ex=read(col/f'exit_{cell}.json');live=ver['docker'][cell]
  assert live['id']==ex['container_id'] and live['state']['Status']=='exited' and live['state']['ExitCode']==ex['returncode']==0 and not live['state']['OOMKilled'] and not ex['guard_stop']
  assert status['state']=='completed' and status['durable_optimizer_steps']==status['logged_updates']==200 and status['finite_model_optimizer']
  for k,v in spec.items():assert plan[k]==v
  assert plan['cell']==cell and plan['identities']==status['identities']
  hashes=read(folder/'result_hashes.json');assert hashes==ver['result_actual'][cell]
  assert hashes['counter.pt']==status['checkpoint_sha256'] and hashes['resume.pt']==status['resume_sha256']
  updates=lines(folder/'updates.jsonl');ledger=lines(folder/'augmentation_ledger.jsonl');draws=lines(col/f'data/draws_{cell}.jsonl')
  assert len(updates)==200 and [r['update'] for r in updates]==list(range(1,201)) and finite(updates)
  assert len(ledger)==1600 and [r['draw'] for r in ledger]==draws
  for update in updates:
   assert update['optimizer_updates']==1 and len(update['microbatch_losses'])==8
   for r in update['microbatch_losses']:
    base=sum(r[v]['focal']+5*r[v]['point'] for v in ('view_a','view_b'))/2
    assert math.isclose(r['total'],base,rel_tol=2e-5,abs_tol=2e-5)
  for name,meta in ver['checkpoints'][cell].items():
   m=meta['literal_metadata_without_deserialization'];assert m['optimizer_steps']==200 and m['seed']==1729 and m['cell']==cell and m['plan_sha256']==sha(folder/'plan.json') and meta['sha256']==hashes[name]
  final=None
  for step in (0,200):
   records=lines(folder/f'development_predictions_{step:04}.jsonl');saved=read(folder/f'development_{step:04}.json')
   assert len(records)==len(truth)==len({r['query_id'] for r in records}) and {r['query_id'] for r in records}==set(truth) and finite(records)
   for r in records:
    t=truth[r['query_id']];assert r['truth']==t['points'] and r['count']==t['count'] and r['dependency_group']==t['dependency_group'] and (r['width'],r['height'])==(t['width'],t['height'])
    assert len(r['points'])==len(r['probabilities']) and all(0<=p<=1 for p in r['probabilities'])
   if step==0:
    if initial is None:initial=records
    else:assert records==initial
   computed={str(t):summarize(records,t) for t in (.5,.1)};compare_objects(saved['scores'],computed)
   assert saved['threshold_selection'] is False
   if step==200:final=computed
  results[cell]=dict(final=final,optimizer_updates=200,augmentation_pairs=1600,update_seconds=sum(r['seconds'] for r in updates));checks[cell]=True
 primary=gate(results['P11']['final']['0.5'],{c:results[c]['final']['0.5'] for c in ('P00','P10','P01')})
 diagnostic=gate(results['P11']['final']['0.1'],{c:results[c]['final']['0.1'] for c in ('P00','P10','P01')})
 result=dict(state='verified_complete',client_kst=datetime.datetime.now().astimezone().isoformat(),checks=checks,cells=results,primary_gate=primary,secondary_diagnostic_cannot_rescue_primary=diagnostic,training_budget=800,full_training_authorized=False,limitations='Two repeatedly used development dependence groups, one seed, fixed200 updates; not independent sites, convergence, FSC dense/extreme accuracy, novelty or paper acceptance. Finite tensors were checked by worker; this verifier checks saved metadata/hashes and predictions without tensor deserialization.')
 with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(dict(state=result['state'],primary_gate=primary),indent=2))
if __name__=='__main__':main()
