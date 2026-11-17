"""Collect and verify terminal public-foundation receipts without model execution."""
import datetime, hashlib, io, json, math, pathlib, subprocess, zipfile
from inspect_ipic_foundation_dynamic import ROOT, REMOTE, ATTEMPT, REMOTE_SCRIPT

def main():
    folder=ROOT/'artifacts/ipic_foundation_dynamic_v1_20261031'
    archive=folder/'result_outputs.zip';dest=folder/'collected';analysis=folder/'completion_analysis.json'
    if analysis.exists():
        print(analysis.read_text());return
    if not archive.exists():
        if dest.exists():raise ValueError('Partial collection exists; inspect it before writing')
        program='ROOT='+repr(REMOTE)+'\nATTEMPT='+repr(ATTEMPT)+'\n'+REMOTE_SCRIPT.replace('print(json.dumps(result))','')+r'''
import io,sys
if result.get('docker',{}).get('state',{}).get('Running') or result['controller_process']:raise ValueError('Job not terminal')
if not result['exit']:raise ValueError('No terminal receipt')
buf=io.BytesIO();hashes={}
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(out.iterdir()):
  if p.is_file() and p.suffix in ('.json','.jsonl','.log'):
   raw=p.read_bytes();hashes[p.name]=hashlib.sha256(raw).hexdigest();z.writestr(p.name,raw)
 result['file_hashes']=hashes;z.writestr('collection_verification.json',json.dumps(result,indent=2))
sys.stdout.buffer.write(buf.getvalue())
'''
        p=subprocess.run(['C:/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=15','training-host','python3','-'],input=program.encode(),capture_output=True,timeout=120)
        if p.returncode:raise RuntimeError(p.stderr.decode())
        with archive.open('xb') as f:f.write(p.stdout)
    with zipfile.ZipFile(archive) as z:
        evidence=json.loads(z.read('collection_verification.json'))
        if evidence['source_mismatches']:raise ValueError('Frozen release changed')
        for name,h in evidence['file_hashes'].items():
            if pathlib.PurePosixPath(name).name!=name or hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('Unsafe/changed archive file')
        if not dest.exists():
            dest.mkdir()
            for name in (*evidence['file_hashes'],'collection_verification.json'):(dest/name).write_bytes(z.read(name))
    def read(name):return json.loads((dest/name).read_text())
    def lines(name):return [json.loads(s) for s in (dest/name).read_text().splitlines()]
    state=read('status.json');plan=read('plan.json');exit=read('result.json');controller=read('controller_status.json')
    frozen=json.loads((ROOT/'study_protocols/ipic_public_foundation_dynamic_recipe_20261031.json').read_text())
    checks={
      'successful_exit':exit['returncode']==0 and not exit['guard_stop'] and not exit['docker_state']['OOMKilled'] and not exit['docker_state']['Running'],
      'completed_controller':controller['state']=='completed' and not evidence['controller_process'],
      'completed_2000':state['state']=='completed' and state['durable_optimizer_steps']==2000,
      'release_identity':read('source_hashes.json')==frozen['source_hashes'],
      'plan_identity':state['plan_sha256']==evidence['plan_sha256'],
      'prescribed_recipe':plan['optimizer_updates']==2000 and plan['seed']==1729 and plan['count_loss_coefficient']==0 and plan['relation_losses'] is False,
      'manifest_identity':plan['manifest_sha256']==frozen['inputs']['native_public_manifest_sha256'],
      'audited_input_metadata_identity':not evidence['audited_metadata_mismatches'],
      'finite_worker_check':state.get('finite_model_optimizer') is True,
      'checkpoint_hashes':all(state.get(k+'_sha256')==evidence['checkpoints'].get(k+'.pt',{}).get('sha256') for k in ('resume','foundation')),
      'checkpoint_literal_metadata':all(evidence['checkpoints'].get(k+'.pt',{}).get('literal_metadata_without_deserialization',{}).get('optimizer_steps')==2000 and evidence['checkpoints'][k+'.pt']['literal_metadata_without_deserialization'].get('seed')==1729 and evidence['checkpoints'][k+'.pt']['literal_metadata_without_deserialization'].get('plan_sha256')==evidence['plan_sha256'] for k in ('resume','foundation')),
      'checkpoint_draw_cursor':evidence['checkpoints'].get('resume.pt',{}).get('literal_metadata_without_deserialization',{}).get('draw_cursor')==16000,
      'draw_hash':plan['draw_sha256']==evidence['draws_sha256'],
    }
    if (dest/'updates.jsonl').exists():
        updates=lines('updates.jsonl')
        checks['update_coverage']=[r['update'] for r in updates]==list(range(1,2001))
        checks['finite_update_metrics']=all(math.isfinite(r['loss']) and math.isfinite(r['pre_clip_norm']) and math.isfinite(r['seconds_including_input']) for r in updates)
    else:updates=[];checks['update_coverage']=False;checks['finite_update_metrics']=False
    draws=lines('draws.jsonl');ledger=lines('augmentation_ledger.jsonl') if (dest/'augmentation_ledger.jsonl').exists() else []
    checks['augmentation_draw_alignment']=[r['draw'] for r in ledger]==draws and len(draws)==16000
    checks['source_exposure']=evidence['draw_coverage']['sources']=={'FSC147':8000,'D2S':8000} and evidence['draw_coverage']['all_products']==2000
    checks['same_source_geometry_in_each_pair']=all(r['view_a']['geometry']['original_size']==r['view_b']['geometry']['original_size'] for r in ledger)
    result={'verified_kst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'state':'verified_complete' if all(checks.values()) else 'verification_failed_or_run_incomplete','checks':checks,'saved_updates':state['durable_optimizer_steps'],'elapsed_seconds':exit['elapsed_seconds'],'draw_coverage':evidence['draw_coverage'],'checkpoints':evidence['checkpoints'],'source_validation_accuracy':None,'sack_adaptation_started':False,'limitations':'Receipt/hash and saved-log verification. Finite model/optimizer check was executed by frozen worker at save time, not independently deserialized here. No held-out inference or novel mechanism is established by foundation completion.'}
    with analysis.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
