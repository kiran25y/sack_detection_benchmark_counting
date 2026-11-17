"""Disposable actual-data A-F CUDA/recovery guard. Never admits by estimate."""
import argparse,gc,hashlib,json,time,unittest
from pathlib import Path
import numpy as np
import torch
from .adaptation_core import initialize_from_foundation,adaptation_pair_loss,SCREEN_ARMS,OBJECTIVE
from .adaptation_data import Data,sha
from .assignment_cudnn_preflight import state_digest
from .foundation import write_json
from .stable_adaptation_runtime import install,interpolate
from .training import optimizer_update,set_learning_rate
from .matching import capacity_check

FOUNDATION='research-direction-20261012/ipic_public_foundation_dynamic_v1_20261031/foundation.pt'

def code_digest():
    paths=sorted(Path(__file__).parent.glob('*.py'))
    return hashlib.sha256(json.dumps({p.name:sha(p) for p in paths},sort_keys=True).encode()).hexdigest()

def evidence(root,package):
    spec=json.loads((package/'screen_plan.json').read_text());audit=root/spec['pair_audit_remote']
    report=json.loads((audit/'report.json').read_text());manual=json.loads((package/'pair_manual_review.json').read_text())
    if report['state']!='completed' or report['planned_draws']!=8000 or report['checked']!=report['total'] or len(report['manual_pair_ids'])!=200 or report['optimizer_calls']!=0:raise ValueError('Prescribed input audit incomplete')
    if manual.get('passed') is not True or len(manual.get('reviewed_draw_ids',[]))!=200 or manual['audit_report_sha256']!=sha(audit/'report.json'):raise ValueError('Missing actual manual200-pair review')
    draws=list(map(json.loads,(audit/'draws.jsonl').read_text().splitlines()))
    if sha(audit/'draws.jsonl')!=spec['draws_sha256']:raise ValueError('Frozen draws changed')
    for filename,h in spec['input_hashes'].items():
        if sha(package/filename)!=h:raise ValueError('Input evidence changed: '+filename)
    identities=dict(foundation=sha(root/FOUNDATION),training_manifest=sha(package/'train_manifest.json'),
        development_manifest=sha(package/'development_manifest.json'),draws=sha(audit/'draws.jsonl'),
        pair_audit=sha(package/'pair_manual_review.json'),asset_background_ledger=sha(package/'asset_background_review.json'),source_release=code_digest())
    return spec,audit,draws,identities

def finite(model,opt):
    if not all(torch.isfinite(x).all() for x in model.state_dict().values()):raise ValueError('Nonfinite model')
    if not all(torch.isfinite(x).all() for s in opt.state.values() for x in s.values() if torch.is_tensor(x)):raise ValueError('Nonfinite Adam')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--package',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.root);package=Path(a.package);out=Path(a.output);out.mkdir(exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Exclusive guard')
    torch.set_num_threads(2);runtime=install();spec,audit,draws,identities=evidence(root,package)
    report=dict(state='running',stage='sack_adaptation_cuda',passed=False,objective=OBJECTIVE,assignment='dynamic',arms=list(SCREEN_ARMS),
        identities=identities,optimizer_calls=0,maximum_optimizer_calls=10,persistent_steps=0,runtime_by_arm={},updates=[])
    def save():write_json(out/'report.json',report)
    save()
    try:
        suite=unittest.defaultTestLoader.discover('/code/tests',pattern='test_ipic_adaptation*.py')
        check=unittest.TextTestRunner(verbosity=1).run(suite)
        if not check.wasSuccessful():raise ValueError('CPU contracts failed on final code')
        report['CPU_contract_tests']=check.testsRun
        # Actual CUDA native forward/CPU adjoint checks for all additional detail resizes.
        operator=[]
        for factor in (1,2,4,8):
            torch.manual_seed(1729);x=torch.randn(1,3,5,7,requires_grad=True);g=torch.randn(1,3,5*factor,7*factor)
            native=torch.nn.functional.interpolate(x,scale_factor=factor,mode='bilinear',align_corners=False);(native*g).sum().backward()
            xx=x.detach().cuda().requires_grad_();v=interpolate(xx,scale_factor=factor,mode='bilinear',align_corners=False)
            (v*g.cuda()).sum().backward();torch.testing.assert_close(v.cpu(),native.detach(),rtol=.003,atol=1e-5)
            torch.testing.assert_close(xx.grad.cpu(),x.grad,rtol=.003,atol=1e-5);operator.append(factor)
        data=Data(root,package)
        def find(predicate):return next(d for d in draws if predicate(d))
        maxsack=max(range(len(data.train)),key=lambda i:data.train[i]['count']);maxfsc=max((i for i,r in enumerate(data.public) if r['dataset']=='FSC147'),key=lambda i:data.public[i]['count'])
        choices=[find(lambda d:d['dataset']=='sacks' and d['row_index']==maxsack),find(lambda d:d['dataset']=='sacks'),
            dict(dataset='FSC147',row_index=maxfsc,query_id=data.public[maxfsc]['query_id'],all_products=False,seed=1731,family='ordinary',draw_id='extra_capacity_guard'),
            find(lambda d:d['dataset']=='D2S' and not d['composite'] and d['all_products']),
            find(lambda d:d['dataset']=='D2S' and not d['composite'] and not d['all_products']),
            find(lambda d:d['family']=='replacement' and not d.get('verified_zero_query')),
            find(lambda d:d['family']=='occlusion' and not d.get('verified_zero_query')),
            find(lambda d:d.get('composite') and d.get('verified_zero_query'))]
        micro=[];records=[];capacity=[]
        for d in choices:
            batch,record=data.prepare(d,'cuda');micro.append(batch);records.append(record)
            for side in ('a','b'):
                h,w=batch[side].shape[-2:];yy,xx=torch.meshgrid(torch.arange(h//4,device='cuda')*4+2,torch.arange(w//4,device='cuda')*4+2,indexing='ij')
                anchors=torch.stack((xx.flatten(),yy.flatten()),1).float();valid=batch['valid_'+side][yy.flatten(),xx.flatten()]
                capacity.append(capacity_check(anchors,batch['points_'+side],valid))
        report.update(operator_factors_checked=operator,queries=records,assignment_capacity=capacity);save()
        initial_output=None;shared_digest=None;torch.cuda.reset_peak_memory_stats()
        for arm in SCREEN_ARMS:
            model,opt=initialize_from_foundation(root/FOUNDATION,arm,'cuda');report['runtime_by_arm'][arm]=runtime
            common={k:v for k,v in model.state_dict().items() if not k.startswith('detail.')}
            digest=state_digest(common)
            if shared_digest is None:shared_digest=digest
            elif digest!=shared_digest:raise ValueError('Shared initialization differs')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                b=micro[0];proto=model.encode_references(b['references'],b['reference_boxes']);v=model(b['a'],b['valid_a'],proto)
                values={k:v[k].detach().cpu() for k in ('logits','points','probabilities')}
                if initial_output is None:initial_output=values
                else:
                    for k in values:torch.testing.assert_close(values[k],initial_output[k],rtol=0,atol=0)
            del proto,v,common
            def update(index):
                set_learning_rate(opt,index,1000,100);start=time.monotonic()
                m=optimizer_update(model,opt,micro,arm,loss_function=adaptation_pair_loss);torch.cuda.synchronize();finite(model,opt)
                report['optimizer_calls']+=1;report['updates'].append(dict(arm=arm,schedule_index=index,seconds=time.monotonic()-start,**m));save()
            update(0)
            if arm in ('B','D'):
                snap=dict(model=model.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                original={k:state_digest(v) for k,v in snap.items()};torch.save(snap,out/f'{arm}_recovery.pt');del snap
                update(1);expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                moments={str(i)+':'+k:v.detach().cpu().clone() for i,s in opt.state_dict()['state'].items() for k,v in s.items() if torch.is_tensor(v)}
                ck=torch.load(out/f'{arm}_recovery.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
                torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);del ck
                actual=dict(model=model.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                if {k:state_digest(v) for k,v in actual.items()}!=original:raise ValueError('Pre-update recovery differs')
                del actual;update(1);maximum=0.
                for k,v in model.state_dict().items():
                    val=v.detach().cpu();torch.testing.assert_close(val,expected[k],rtol=.003,atol=1e-5);maximum=max(maximum,float((val-expected[k]).abs().max()))
                for i,s in opt.state_dict()['state'].items():
                    for k,v in s.items():
                        if torch.is_tensor(v):torch.testing.assert_close(v.cpu(),moments[str(i)+':'+k],rtol=.003,atol=1e-5)
                report.setdefault('recovery',{})[arm]=dict(preupdate_exact=True,poststep_model_max_abs=maximum,rtol=.003,atol=1e-5)
                del expected,moments
            del model,opt;gc.collect();torch.cuda.empty_cache()
        assert report['optimizer_calls']==10
        report.update(state='completed',passed=True,full_two_view_backward=True,finite_model_optimizer=True,authentic_recovery=True,
            same_shared_initialization=True,high_count_and_wide_shape=True,regional_loss_backward=True,
            peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved(),shared_initial_digest=shared_digest)
        save()
    except Exception as exc:report.update(state='failed',error=repr(exc));save();raise
if __name__=='__main__':main()
