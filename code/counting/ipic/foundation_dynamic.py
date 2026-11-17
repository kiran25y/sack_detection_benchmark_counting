"""Owner-authorized amended public B0 foundation; no diagnostic weights promoted.

Original foundation.py and all historical releases remain unchanged. Only the
validated no-soft-count objective and B0 numerical runtime are amended here.
"""
from .stable_resampling import install

import argparse,collections,json,pathlib,time
import numpy as np
import torch
from . import SPEC_SHA256
from .foundation import sha,write_json
from .model import IPICNet
from .preflight import WEIGHT_SHA,tensor_digest
from .point_fit_diagnostic import point_only_pair
from .public_data import make_draws,prepare_pair
from .training import make_optimizer,optimizer_update,set_learning_rate

MANIFEST_SHA='8a70c247df7f304b46cd47b215de23e71c83a1aaeff58def3e5b8522b35e3721'
REQUIRED_EVIDENCE={'data_audit','public_audit','trainability_report','trainability_exit','cpu_exit','guard_report','guard_exit'}

def validate_readiness(root,ready):
    if ready.get('stage')!='public_foundation_dynamic' or ready.get('source_specification_sha256')!=SPEC_SHA256:
        raise ValueError('Wrong foundation recipe/specification')
    if ready.get('count_loss_coefficient')!=0 or ready.get('assignment')!='dynamic' or ready.get('updates')!=2000:
        raise ValueError('Wrong amended objective or budget')
    entries=ready.get('evidence',{})
    if set(entries)!=REQUIRED_EVIDENCE:raise ValueError('Incomplete named readiness evidence')
    evidence={}
    for key,row in entries.items():
        path=root/row['path']
        if sha(path)!=row['sha256']:raise ValueError('Readiness evidence changed: '+key)
        evidence[key]=json.loads(path.read_text())
    data=evidence['data_audit'];public=evidence['public_audit'];fit=evidence['trainability_report'];guard=evidence['guard_report']
    if data['state']!='completed' or data['verified_encoded_images']!=8033 or data['manifest_sha256']!=MANIFEST_SHA:
        raise ValueError('Public data audit incomplete')
    if data['image_hash_mismatches'] or data['exact_encoded_overlap_with_FSC_heldout'] or data['decoded_cross_dataset_duplicates']:
        raise ValueError('Public input identity/partition failure')
    if public['state']!='completed' or public['capacities']!=24099 or not public['all_native_training_images_decoded']:
        raise ValueError('Native decode/capacity audit missing')
    if not fit['decision']['dynamic_trainability_passed'] or fit['optimizer_calls']!=400 or fit['state']!='completed':
        raise ValueError('Dynamic integer-point trainability missing')
    if not guard.get('engineering_guard_passed') or not guard.get('native_chart_trainability_passed') or guard.get('optimizer_calls')!=3:
        raise ValueError('Amended mixed-data CUDA/recovery guard incomplete')
    if guard['count_loss_coefficient']!=0 or guard['assignment']!='dynamic' or guard['manifest_sha256']!=MANIFEST_SHA:
        raise ValueError('Wrong public CUDA recipe')
    if not guard['runtime']['strict_deterministic_algorithms'] or guard['runtime']['warn_only']:
        raise ValueError('Unvalidated numerical runtime')
    for key in ('trainability_exit','cpu_exit','guard_exit'):
        v=evidence[key]
        if v['returncode']!=0 or v['docker_state']['Running'] or v['docker_state']['OOMKilled'] or v['guard_stop']:
            raise ValueError('Failed prerequisite: '+key)
    actual={str(p.relative_to(pathlib.Path(__file__).parents[2])).replace('\\','/'):sha(p) for p in pathlib.Path(__file__).parent.glob('*.py')}
    for name,h in actual.items():
        if evidence['guard_exit']['source_hashes'].get(name)!=h:raise ValueError('Code changed after CUDA guard: '+name)
    return evidence

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--readiness',required=True)
    args=ap.parse_args();root=pathlib.Path(args.root);out=pathlib.Path(args.output);out.mkdir(exist_ok=True)
    if any((out/n).exists() for n in ('plan.json','status.json','resume.pt','foundation.pt')):raise ValueError('Exclusive run; no automatic recovery')
    ready=json.loads(pathlib.Path(args.readiness).read_text());validate_readiness(root,ready)
    manifest=root/'research-direction-20261012/ipic_checks_20261030/public_v1/public_train_manifest.jsonl'
    if sha(manifest)!=MANIFEST_SHA:raise ValueError('Public manifest changed')
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    if len(rows)!=8033 or collections.Counter(r['dataset'] for r in rows)!={'FSC147':3653,'D2S':4380}:raise ValueError('Cohort changed')
    draws=make_draws(rows,2000,1729)
    with (out/'draws.jsonl').open('x') as f:
        for d in draws:f.write(json.dumps(d)+'\n')
    torch.set_num_threads(2);torch.manual_seed(1729);np.random.seed(1729);runtime=install()
    plan={'stage':'public_foundation_dynamic','source_specification_sha256':SPEC_SHA256,'amendment':'dynamic matching; count coefficient0; validated stable B0 runtime',
      'architecture':'B0','seed':1729,'optimizer_updates':2000,'paired_draws':16000,'query_presentations':32000,
      'manifest_sha256':MANIFEST_SHA,'draw_sha256':sha(out/'draws.jsonl'),'source_hashes':{p.name:sha(p) for p in sorted(pathlib.Path(__file__).parent.glob('*.py'))},
      'source_draws':dict(collections.Counter(d['dataset'] for d in draws)),'D2S_all_products_draws':sum(d['all_products'] for d in draws),
      'ordinary_supervision':'mean per-view focal +5*point, native dynamic matching','count_loss_coefficient':0,'relation_losses':False,
      'runtime':runtime,'warmup_updates':100,'cosine_final_lr_ratio':.01,'effective_pairs_per_update':8,
      'pretrained_lr':2e-5,'new_module_lr':2e-4,'weight_decay':.05,'no_decay':'bias and one-dimensional parameters','adam_betas':[.9,.999],'adam_eps':1e-8,'clip_norm':1.,
      'precision':'BF16 network/FP32 losses','pretrained_context_sha256':WEIGHT_SHA,'initialization':'fresh seed1729 official context + fresh common modules; no diagnostic checkpoint',
      'sack_training_images':0,'held_out_query_presentations':0,'native_D2S_only':True,'claim':'joint public data; known D2S-to-FSC held-out category overlap; not strict unseen-category generalization',
      'threshold_selection':False,'evaluation_or_adaptation_auto_launch':False,'fresh_optimizer_per_later_arm':True,'readiness':ready}
    write_json(out/'plan.json',plan);ph=sha(out/'plan.json')
    state={'state':'initializing','plan_sha256':ph,'durable_optimizer_steps':0,'total_optimizer_steps':2000,'stage':plan['stage'],'sack_adaptation_started':False,'source_validation_accuracy':None}
    write_json(out/'status.json',state)
    try:
        model=IPICNet(False);model.context.load_official(root/'research-direction-20261012/ipic_assets_v1_20261030/convnext_tiny_1k_224_ema.pth',WEIGHT_SHA)
        model.cuda().train();opt=make_optimizer(model);curves=[]
        state.update(state='training',initial_model_tensor_digest=tensor_digest(model));write_json(out/'status.json',state)
        for step in range(2000):
            started=time.monotonic();micro=[];records=[]
            for d in draws[step*8:(step+1)*8]:
                batch,record=prepare_pair(rows[d['row_index']],d,root,'cuda');micro.append(batch);records.append(record)
            set_learning_rate(opt,step,2000,100);m=optimizer_update(model,opt,micro,'A',loss_function=point_only_pair);torch.cuda.synchronize()
            row={'update':step+1,'seconds_including_input':time.monotonic()-started,'loss':float(np.mean([x['total'] for x in m['microbatch_losses']])),
                 'pre_clip_norm':m['pre_clip_norm'],'learning_rates':[g['lr'] for g in opt.param_groups],'microbatch_losses':m['microbatch_losses']}
            curves.append(row)
            with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            with (out/'augmentation_ledger.jsonl').open('a') as f:
                for record in records:f.write(json.dumps(record)+'\n')
            del micro,records,batch
            if (step+1)%25==0:
                if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise FloatingPointError('Nonfinite model')
                if not all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)):raise FloatingPointError('Nonfinite optimizer')
                ns=np.random.get_state();npstate={'name':ns[0],'keys':ns[1].tolist(),'position':ns[2],'has_gauss':ns[3],'cached_gaussian':ns[4]}
                tmp=out/'resume.tmp';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'optimizer_steps':step+1,
                  'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':npstate,'plan_sha256':ph,'seed':1729,
                  'draw_cursor':(step+1)*8,'curves':curves,'scheduler':'deterministic set_learning_rate from update index','runtime':runtime},tmp);tmp.replace(out/'resume.pt')
                state.update(durable_optimizer_steps=step+1,last_saved_host_unix=time.time(),recent_seconds_per_update=float(np.median([x['seconds_including_input'] for x in curves[-25:]])))
                write_json(out/'status.json',state)
        tmp=out/'foundation.tmp';torch.save({'model':model.state_dict(),'plan_sha256':ph,'optimizer_steps':2000,'seed':1729,
            'purpose':'shared public foundation; reset optimizer for subsequent matched arms','runtime':runtime},tmp);tmp.replace(out/'foundation.pt')
        state.update(state='completed',foundation_sha256=sha(out/'foundation.pt'),resume_sha256=sha(out/'resume.pt'),finite_model_optimizer=True)
        write_json(out/'status.json',state)
    except Exception as exc:
        state.update(state='failed',error=repr(exc));write_json(out/'status.json',state);raise

if __name__=='__main__':main()
