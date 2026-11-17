"""Fixed public foundation, launched only after the full engineering contract passes.

Fresh seed1729, official context weights, B0, 2000 updates, no relation loss.
There is deliberately no automatic resume or bypass-readiness option.
"""
import argparse,collections,hashlib,json,pathlib,time
import numpy as np
import torch
from . import SPEC_SHA256
from .data_manifest import assert_stage_ready
from .model import IPICNet
from .preflight import WEIGHT_SHA
from .public_data import make_draws,prepare_pair
from .training import make_optimizer,optimizer_update,set_learning_rate

def sha(path):
    with pathlib.Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write_json(path,data):
    path=pathlib.Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)

def validate_admission(root,ready):
    assert_stage_ready(ready,'foundation')
    if ready.get('source_specification_sha256')!=SPEC_SHA256:raise ValueError('Wrong source specification')
    if ready.get('public_manifest_sha256')!='8a70c247df7f304b46cd47b215de23e71c83a1aaeff58def3e5b8522b35e3721':raise ValueError('Public manifest not frozen audited input')
    for r in ready.get('evidence_files',[]):
        path=root/r['path']
        if sha(path)!=r['sha256']:raise ValueError('Readiness evidence changed: '+r['path'])
    if len(ready.get('evidence_files',[]))<4:raise ValueError('Missing detailed readiness evidence')
    if not ready.get('mixed_public_cuda_guard_passed'):raise ValueError('Real D2S/FSC augmented-pair CUDA check missing')
    return True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--readiness',required=True);a=ap.parse_args()
    root=pathlib.Path(a.root);out=pathlib.Path(a.output);out.mkdir(exist_ok=True)
    if any((out/n).exists() for n in ['plan.json','status.json','resume.pt','foundation.pt']):raise RuntimeError('Existing run; explicit recovery only')
    ready=json.loads(pathlib.Path(a.readiness).read_text());validate_admission(root,ready)
    manifest=root/'research-direction-20261012/ipic_checks_20261030/public_v1/public_train_manifest.jsonl'
    if sha(manifest)!=ready['public_manifest_sha256']:raise ValueError('Manifest changed')
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    if len(rows)!=8033 or collections.Counter(r['dataset'] for r in rows)!={'FSC147':3653,'D2S':4380}:raise ValueError('Cohort changed')
    draws=make_draws(rows,2000,1729)
    with (out/'draws.jsonl').open('x') as f:
        for d in draws:f.write(json.dumps(d)+'\n')
    code_root=pathlib.Path(__file__).parent
    source_hashes={p.name:sha(p) for p in sorted(code_root.glob('*.py'))}
    plan={'stage':'public_foundation','architecture':'B0','source_specification_sha256':SPEC_SHA256,
      'seed':1729,'optimizer_updates':2000,'paired_draws':16000,'query_presentations':32000,
      'draw_sha256':sha(out/'draws.jsonl'),'manifest_sha256':sha(manifest),'source_hashes':source_hashes,
      'source_draws':dict(collections.Counter(d['dataset'] for d in draws)),
      'd2s_all_products_draws':sum(d['all_products'] for d in draws),'pretrained_context_sha256':WEIGHT_SHA,
      'relation_losses':False,'ordinary_supervision':'mean of two unchanged per-view losses',
      'warmup_updates':100,'cosine_final_lr_ratio':.01,'gradient_accumulation_pairs':8,
      'pretrained_lr':2e-5,'new_module_lr':2e-4,'weight_decay':.05,'no_decay':'bias and one-dimensional parameters',
      'adam_betas':[.9,.999],'adam_eps':1e-8,'clip_norm':1.,'precision':'BF16 network, FP32 count and loss',
      'tf32':False,'initialization':'official context + fresh common modules; no engineering-fit weights',
      'native_d2s_only_in_foundation':True,'sack_training_images':0,'held_out_query_presentations':0,
      'fresh_optimizer_per_later_adaptation_arm':True,'all_model_comparisons_pending':True,
      'readiness':ready}
    write_json(out/'plan.json',plan);plan_hash=sha(out/'plan.json')
    state={'state':'validating','plan_sha256':plan_hash,'durable_optimizer_steps':0,'total_optimizer_steps':2000,
       'research_stage':'public foundation only','sack_adaptation_started':False,'source_validation_accuracy':None}
    write_json(out/'status.json',state)
    try:
        torch.set_num_threads(2);torch.manual_seed(1729);np.random.seed(1729)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        model=IPICNet(False);weight=root/'research-direction-20261012/ipic_assets_v1_20261030/convnext_tiny_1k_224_ema.pth'
        model.context.load_official(weight,WEIGHT_SHA);model.cuda().train();opt=make_optimizer(model);curves=[]
        state['state']='training';write_json(out/'status.json',state)
        for step in range(2000):
            started=time.monotonic();micro=[];records=[]
            for d in draws[step*8:(step+1)*8]:
                batch,record=prepare_pair(rows[d['row_index']],d,root,'cuda');micro.append(batch);records.append(record)
            set_learning_rate(opt,step,2000,100);m=optimizer_update(model,opt,micro,'A');torch.cuda.synchronize()
            row={'update':step+1,'seconds_including_input':time.monotonic()-started,
              'loss':float(np.mean([v['total'] for v in m['microbatch_losses']])),
              'pre_clip_norm':m['pre_clip_norm'],'learning_rates':[g['lr'] for g in opt.param_groups],
              'microbatch_losses':m['microbatch_losses']}
            curves.append(row)
            with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            with (out/'augmentation_ledger.jsonl').open('a') as f:
                for record in records:f.write(json.dumps(record)+'\n')
            del micro,records,batch
            if (step+1)%25==0 or step==1999:
                temp=out/'resume.tmp'
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'optimizer_steps':step+1,
                  'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                  'plan_sha256':plan_hash,'seed':1729,'draw_cursor':(step+1)*8,'curves':curves},temp)
                temp.replace(out/'resume.pt')
                state.update(durable_optimizer_steps=step+1,last_saved_host_unix=time.time(),
                  recent_seconds_per_update=float(np.median([x['seconds_including_input'] for x in curves[-25:]])))
                write_json(out/'status.json',state)
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise FloatingPointError('Nonfinite final state')
        temp=out/'foundation.tmp';torch.save({'model':model.state_dict(),'plan_sha256':plan_hash,
          'optimizer_steps':2000,'seed':1729,'purpose':'shared public foundation; reset optimizer for each adaptation arm'},temp)
        temp.replace(out/'foundation.pt')
        state.update(state='completed',foundation_sha256=sha(out/'foundation.pt'),
          resume_sha256=sha(out/'resume.pt'),durable_optimizer_steps=2000,finite_model=True)
        write_json(out/'status.json',state)
    except Exception as exc:
        state.update(state='failed',error=repr(exc));write_json(out/'status.json',state);raise

if __name__=='__main__':main()
