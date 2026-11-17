"""Read-only independent checks of the fixed800+3000 training completion."""
import datetime,hashlib,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def finite(v):
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    if isinstance(v,list):return all(finite(x) for x in v)
    return not isinstance(v,(int,float)) or math.isfinite(v)
def main():
    out=ROOT/'artifacts/firc_ordinary_training_v1_20261106';result=out/'collected/result'
    assert not (out/'completion_analysis.json').exists(),'Reuse prior analysis'
    v=read(out/'collected/verification.json');r=read(result/'report.json');plan=read(out/'data/plan.json');launch=read(out/'launch_receipt.json')
    pf=read(ROOT/'study_protocols/firc_ordinary_cuda_verified_20261106.json');rows=[json.loads(s) for s in (out/'data/rows.jsonl').read_text().splitlines()]
    logs=[json.loads(s) for s in (result/'updates.jsonl').read_text().splitlines()];c,d=v['controller_status'],v['docker_state']
    checks={'controller_complete':not v['controller_alive'] and c['state']=='completed' and c['exit_code']==0 and c['guard_stop'] is None,
      'docker_complete':d['Status']=='exited' and d['ExitCode']==0 and not d['OOMKilled'],
      'local_release':all(sha(out/k)==h for k,h in launch['release_hashes'].items()),'remote_release':v['release_hashes_match'],
      'plan':sha(out/'data/plan.json')==launch['plan_sha256']==r['plan_sha256']==pf['plan_sha256'],
      'result_hashes':all(sha(result/k)==h for k,h in read(result/'result_hashes.json').items()),
      'fresh_initialization':r['fresh_head_sha256']==pf['fresh_head_sha256'] and not r['diagnostic_weights_loaded'],
      'frozen_encoder':r['encoder_initial_sha256']==r['encoder_final_sha256']==pf['encoder_sha256'],
      'budget':r['optimizer_calls']==r['saved_updates']==3800 and r['query_presentations']==30400 and len(logs)==3800,
      'calls':[e['call'] for e in logs]==list(range(1,3801)),'finite_logs':finite(logs),
      'train_only':len(rows)==8075 and all(row['split']=='train' for row in rows),
      'ordinary_only':r['physical_quartets']==0,'finite_worker':r['finite_model_adam'],
      'shared_foundation':r['stages']['foundation']['head_final_sha256']==r['stages']['adaptation']['head_initial_sha256']}
    scopes={}
    for name,n in [('foundation',800),('adaptation',3000)]:
        a=r['stages'][name];stage=plan['stages'][name];ll=[e for e in logs if e['stage']==name]
        draws=[json.loads(s) for s in (out/'data'/stage['draw_file']).read_text().splitlines()]
        checks[name+'_budget']=a['saved_updates']==a['optimizer_calls']==n and a['completed'] and len(ll)==n and [e['stage_step'] for e in ll]==list(range(1,n+1))
        checks[name+'_checkpoints']=sha(result/(name+'.pt'))==a['checkpoint_sha256'] and sha(result/(name+'_resume.pt'))==a['resume_sha256']
        checks[name+'_rates']=[e['lr'] for e in ll]==stage['learning_rates']
        actual=[q for e in ll for q in e['queries']]
        checks[name+'_draws']=len(actual)==n*8 and [q['draw'] for q in actual]==draws and sha(out/'data'/stage['draw_file'])==stage['draw_sha256']
        checks[name+'_optimizer_reset']=a['optimizer_reset']
        truth_ok=True;counts={};seen=set()
        for q in actual:
            draw=q['draw'];row=rows[draw['row_index']];source=row['dataset'];counts[source]=counts.get(source,0)+1;seen.add(draw['row_index'])
            truth=(row['count'] if source in ['FSC147','sacks'] else sum(o['category_id']==draw['category_id'] for o in row['objects']))
            truth_ok &= q['count']==q['matched']==truth and q['roundtrip_error']<1e-8
        checks[name+'_truth_and_matching']=truth_ok;checks[name+'_source_proportions']=counts==stage['draw_counts']
        scopes[name]={'updates':n,'query_presentations':len(actual),'unique_rows':len(seen),'source_draws':counts,'checkpoint_sha256':a['checkpoint_sha256']}
    output={'verified_kst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'complete':all(checks.values()),'checks':checks,'stages':scopes,
      'training_updates':3800,'query_presentations':30400,'development_scored':False,'reserved_scored':False,'MAE':None,
      'limitations':'Completion is not accuracy, convergence, novelty or paper readiness. Worker checks finite tensors; this verifier hashes checkpoints without deserializing tensors.'}
    with (out/'completion_analysis.json').open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps(output,indent=2));assert output['complete']
if __name__=='__main__':main()
