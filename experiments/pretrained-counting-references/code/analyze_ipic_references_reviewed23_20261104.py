"""Score existing predictions after owner labels freeze; zero inference calls."""
import datetime,hashlib,json,math
from pathlib import Path
import numpy as np
from analyze_ipic_pretrained_reference import metrics,summary,finite
ROOT=Path(__file__).resolve().parents[1];BANK=ROOT/'artifacts/ipic_pretrained_reference_v1_20261104';DATA=ROOT/'artifacts/ipic_transfer_pilot_data_v1_20261104'
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 output=BANK/'analysis_reviewed23.json';assert not output.exists()
 old=read(BANK/'analysis.json');assert old['engineering_passed'] and old['model_calls']==58 and old['optimizer_steps']==0
 v=read(BANK/'collected/collection_verification.json');assert v['all_current_identities_match'] and all(v['current_identity_checks'].values()) and v['controller_exited']
 truth=read(DATA/'development_manifest.json');assert len(truth)==23 and len({t['image_id']for t in truth})==23 and sum(t['count']for t in truth)==355
 inputs={r['query_id']:r for r in read(BANK/'inputs.json')};assert set(inputs)=={t['image_id']for t in truth}
 for t in truth:
  q=inputs[t['image_id']];assert t['exhaustive_reviewed'] and t['count']==len(t['points'])
  assert (t['width'],t['height'],t['roi_xyxy'],t['dependency_group'])==(q['width'],q['height'],q['roi_xyxy'],q['dependency_group']) and t['encoded_sha256']==q['image_sha256']
 for name in ['references.json']:
  # Query capture uses the same frozen external reference crops; no new exemplar choice.
  assert sha(DATA/name)==sha(ROOT/'artifacts/ipic_sack_data_admission_v1_20261103'/name)
 results={};hashes={}
 for model in ['countgd','geco']:
  folder=BANK/'collected'/(model+'_bank');h=read(folder/'result_hashes.json');p=folder/'predictions.jsonl';assert sha(p)==h[p.name]
  rows=[json.loads(s)for s in p.read_text().splitlines()];assert len(rows)==23 and finite(rows)
  bank={r['query_id']:r for r in rows};assert len(bank)==23 and set(bank)==set(inputs)
  for r in rows:
   q=inputs[r['query_id']];assert r['roi_xyxy']==q['roi_xyxy'];boxes=np.asarray(r['boxes']).reshape(-1,4);centers=(boxes[:,:2]+boxes[:,2:])/2;x0,y0,x1,y1=q['roi_xyxy'];keep=(centers[:,0]>=x0)&(centers[:,0]<x1)&(centers[:,1]>=y0)&(centers[:,1]<y1)
   assert keep.tolist()==r['roi_keep'];np.testing.assert_array_equal(centers[keep],np.asarray(r['points']).reshape(-1,2))
  scores=[metrics(bank[t['image_id']]['points'],t)for t in truth];res=summary(scores)
  res['groups']={g:summary([r for r in scores if r['dependency_group']==g]) for g in sorted({t['dependency_group']for t in truth})}
  errors=[len(bank[t['image_id']]['points'])-t['count']for t in truth]
  for k,x in [('MAE',sum(abs(x)for x in errors)/23),('RMSE',math.sqrt(sum(x*x for x in errors)/23)),('bias',sum(errors)/23)]:assert abs(res['all'][k]-x)<1e-12
  originals=summary([r for r in scores if r['query_id'].startswith('dev_')]);assert originals==old['results'][model]
  results[model]=res;hashes[model]=sha(p)
 receipt=dict(state='verified_saved_prediction_rescore',verified_kst=datetime.datetime.now().astimezone().isoformat(),scored_queries=23,true_sack_points=355,development_groups=2,source_manifest_sha256=sha(DATA/'development_manifest.json'),owner_export_sha256=sha(DATA/'owner_review.json'),prior_analysis_sha256=sha(BANK/'analysis.json'),prediction_hashes=hashes,results=results,new_inference_calls=0,new_optimizer_calls=0,threshold_or_reference_changes=False,prior11_metrics_preserved=True,pilot_plan_sha256=sha(DATA/'pilot_plan.json'),limitations='Repeatedly used development recordings, not independent test. CountGD++ native external prompts; GeCo2 disclosed external-reference adapter, not official native external accuracy. Different training supervision/capacity/compute from IPIC. Do not retune the frozen pilot from these metrics.')
 with output.open('x')as f:json.dump(receipt,f,indent=2,allow_nan=False)
 print(json.dumps({m:{k:v for k,v in r.items()if k not in['per_image','groups']}for m,r in results.items()},indent=2))
if __name__=='__main__':main()
