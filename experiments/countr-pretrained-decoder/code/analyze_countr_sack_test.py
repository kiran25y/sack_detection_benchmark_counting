"""Independently recompute saved density metrics and gates; no model loading."""
import argparse,datetime,hashlib,json,math
from pathlib import Path
import numpy as np
from countr_sack_metrics import score,summary,gate
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def compare(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:compare(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):compare(x,y)
 elif isinstance(a,(float,int)) and not isinstance(a,bool):assert math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-7),(a,b)
 else:assert a==b,(a,b)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--attempt',required=True);a=ap.parse_args();folder=ROOT/'artifacts'/a.attempt;col=folder/'collected';assert not (folder/'analysis.json').exists()
 ver=read(col/'collection_verification.json');spec=read(col/'data/plan.json');ctl=read(col/'controller_status.json');assert ctl['state']=='completed' and ctl['completed_arms']==['R','P'] and ver['controller_exited']
 assert ver['source_actual']==read(folder/'frozen_hashes.json') and ver['weights_actual']==spec['weights_sha256']
 for n,h in ver['collected_hashes'].items():assert sha(col/n)==h,n
 for n,h in spec['source_hashes'].items():assert ver['source_actual'][n]==h
 local=ROOT/'artifacts/countr_sack_recipe_v1_20261105'
 for n,h in spec['data_hashes'].items():assert sha(local/n)==h,n
 rows=read(col/'data/manifest.json');draws=read(col/'data/draws.json');results={};first=None
 for arm in ['R','P']:
  f=col/'result'/arm;s=read(f/'status.json');ex=read(col/f'exit_{arm}.json');d=ver['docker'][arm]['state']
  assert d['Status']=='exited' and d['ExitCode']==ex['returncode']==0 and not d['OOMKilled'] and not ex['guard_stop']
  assert s['state']=='completed' and s['durable_steps']==300 and s['finite_model_optimizer'] and s['initial']['encoder']==s['final']['encoder'] and s['plan_sha256']==sha(col/'data/plan.json')
  meta=ver['checkpoints'][arm]['resume.pt'];m=meta['literal_metadata_without_deserialization'];assert meta['sha256']==s['resume_sha256'] and m['steps']==300 and m['arm']==arm and m['seed']==1729 and m['plan_sha256']==s['plan_sha256']
  updates=[json.loads(l) for l in (f/'updates.jsonl').read_text().splitlines()];ledger=[json.loads(l) for l in (f/'inputs.jsonl').read_text().splitlines()]
  assert len(updates)==300 and [r['update'] for r in updates]==list(range(1,301)) and len(ledger)==1200 and [r['draw'] for r in ledger]==draws
  for u in updates:assert all(math.isfinite(u[k]) for k in ['loss','gradient_norm','lr','seconds']) and len(u['micro_losses'])==4
  if first is None:first=ledger
  else:assert ledger==first
  for name,pred in [(arm,f/'predictions')]+([('B',f/'baseline_B')] if arm=='P' else []):
   scored=[];assert len(list(pred.glob('*.npy')))==65
   for r in rows:
    a=np.load(local/r['array']);raw=np.load(pred/(r['image_id']+'.npy'));assert raw.shape==a['mask'].shape and np.isfinite(raw).all()
    scored.append(dict(split=r['split'],**score(raw,a['mask'],r)))
   value={n:summary([r for r in scored if r['split']==sp]) for n,sp in [('fitting','train'),('development','development')]}
   compare(value,read(pred/'metrics.json'));results[name]=value
 assert read(col/'result/P/status.json')['initial']['encoder']==read(col/'result/R/status.json')['initial']['encoder']
 result=dict(state='verified_complete',client_kst=datetime.datetime.now().astimezone().isoformat(),results=results,gate=gate(results),optimizer_calls=600,
  limitations='Reused23-frame development/2groups; one seed; density mass metrics are not point F1; no FSC dense/extreme, independent generalization or novelty claim. Tensor finite checks are worker checks, not independently deserialized.',automatic_full_training=False)
 with (folder/'analysis.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result['gate'],indent=2))
if __name__=='__main__':main()
