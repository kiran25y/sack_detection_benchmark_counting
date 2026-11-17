"""Bounded engineering only: artificial fixtures, no research labels/results.

CPU validates operators; CUDA additionally loads the official frozen backbone,
checks repeat recovery (two disposable calls), BF16 (one), and 896 (one).
Every Adam call is charged to the <=50 profiling budget; none is initialization.
"""
import argparse,copy,hashlib,json,os,time,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('XFORMERS_DISABLED','1')
import numpy as np
import torch
from pathlib import Path
from . import reference as ref
from . import torch_ops as ops
from .model import FIRC,load_official_encoder,optimizer_for


def configure():
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1729);np.random.seed(1729)


def operator_checks(device):
    from .geometry import transform,points_to_model,points_to_native
    for side in (672,784,896):
        t=transform(2560,1440,side,[350,350,1450,1440]);p0=np.array([[350,350],[1449,1439],[900,800]])
        assert np.max(abs(points_to_native(points_to_model(p0,t),t)-p0))<1e-10
    f=ref.Field(19,37,112,224);rng=np.random.default_rng(1729)
    p=np.vstack([rng.uniform([0,0],[37,19],(25,2)),[[0,0],[37,19],[.1,18.9],[36.8,.2]]])
    masses=rng.uniform(.01,1,len(p));cot=rng.normal(size=f.shape)
    tp=torch.tensor(p,dtype=torch.float64,device=device,requires_grad=True);tm=torch.tensor(masses,dtype=torch.float64,device=device,requires_grad=True)
    value=ops.splat(tp,tm,f);(value*torch.tensor(cot,device=device)).sum().backward()
    expected=ref.splat(p,masses,f);ep,em=ref.splat_vjp(p,masses,f,cot)
    errors={'splat':float(np.max(abs(value.detach().cpu().numpy()-expected))),
        'point_vjp':float(np.max(abs(tp.grad.cpu().numpy()-ep))),
        'mass_vjp':float(np.max(abs(tm.grad.cpu().numpy()-em)))}
    assert max(errors.values())<1e-10,errors
    partition=np.full((112,224),-1,dtype=np.int32);partition[:19,:37]=3;partition[:8,:12]=0;partition[8:19,:12]=1;partition[:19,12:25]=2
    membership=ref.region_memberships(partition,f);mt={s:torch.tensor(v,device=device) for s,v in membership.items()}
    gt=np.stack([expected,expected*1.1,expected*.6,expected*.8]);pred=gt+rng.uniform(0,.1,gt.shape)
    # Only cells of known coverage are eligible for mass.
    valid=membership[4].sum(0)>0;pred[:,~valid]=0
    target=torch.tensor(gt,device=device);estimate=torch.tensor(pred,device=device,requires_grad=True)
    got=ops.response_terms(estimate,target,mt,[1,2,3,6]);want=ref.response_terms(pred,gt,membership,[1,2,3,6])
    for k,v in got.items():assert abs(v.item()-want[k])<1e-10,(k,v,want)
    total=got['independent']+.25*got['first']+.10*got['mixed'];total.backward();assert torch.isfinite(estimate.grad).all()
    assert abs(value.sum().item()-masses.sum())<1e-10
    for value2 in ops.pyramid(value).values():assert abs(value2.sum().item()-masses.sum())<1e-10
    # Matching includes a Hall-condition failure: two targets and one anchor.
    row={'anchors':torch.tensor([[2.,2.],[6,2]],device=device),'points':torch.tensor([[2.,2.],[6,2]],device=device),'probabilities':torch.tensor([.1,.2],device=device)}
    a,b=ops.match(row,torch.tensor([[2.,2.],[6,2]],device=device));assert len(set(b))==2
    bad={k:v[:1] for k,v in row.items()}
    try:ops.match(bad,torch.tensor([[2.,2.],[3,2]],device=device))
    except ValueError:pass
    else:raise AssertionError('Capacity failure accepted')
    # FP32 production-path comparison and repeatability.
    v1=ops.splat(tp.detach().float(),tm.detach().float(),f);v2=ops.splat(tp.detach().float(),tm.detach().float(),f)
    assert torch.equal(v1,v2)
    assert np.allclose(v1.cpu().numpy(),expected,rtol=3e-3,atol=1e-5)
    return {'errors_float64':errors,'response_parity':True,'mass_conserved':True,'capacity_failure_rejected':True,'fp32_repeat_exact':True}


def fixture(side,device):
    # Bounded artificial pixels; never called natural or physical quartet data.
    gen=torch.Generator(device='cpu').manual_seed(100+side)
    images=torch.randn(4,3,side,side,generator=gen).to(device)
    refs=torch.randn(2,3,224,224,generator=gen).to(device)
    boxes=torch.tensor([[0,0,0,224,224],[1,7,14,210,214]],dtype=torch.float32,device=device)
    points=[[],[[30.,30.]],[[side-35.,side-35.]],[[30.,30.],[side-35.,side-35.]]]
    truth=[torch.tensor(p,device=device).reshape(-1,2) for p in points]
    f=ref.Field(side,side,side,side);partition=np.full((side,side),3,dtype=np.int32)
    partition[:side//2,:side//2]=0;partition[side//2:,side//2:]=1
    memberships={k:torch.tensor(v,device=device,dtype=torch.float32) for k,v in ref.region_memberships(partition,f).items()}
    return images,refs,boxes,truth,f,memberships


def state_diff(a,b):
    if isinstance(a,torch.Tensor):
        if not torch.isfinite(b).all():raise AssertionError('Nonfinite restored state')
        return float((a-b).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):assert a.keys()==b.keys();return max([state_diff(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,(list,tuple)):assert len(a)==len(b);return max([state_diff(x,y) for x,y in zip(a,b)]+[0.])
    assert a==b,(a,b);return 0.


def model_checks(args,report,save):
    checkpoint=Path(args.assets)/'dinov2_vitb14_reg4_pretrain.pth'
    with checkpoint.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    assert digest=='73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71'
    model=FIRC(load_official_encoder(checkpoint)).to('cuda').train()
    assert not model.encoder.training and all(not p.requires_grad for p in model.encoder.parameters())
    report['checkpoint_sha256']=digest
    report['parameters']={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    images,refs,boxes,truth,field,memberships=fixture(224,'cuda')
    taps=model.encode(images);rt=model.encode(refs)
    assert [list(t.shape) for t in taps]==[[4,768,16,16]]*4
    with torch.no_grad():
        a=model(images,refs,boxes,[(224,224)]*4,taps=taps,reference_taps=rt)
        b=model(images,refs,boxes,[(224,224)]*4,detail_enabled=False,taps=taps,reference_taps=rt)
        assert all(torch.equal(x['points'],y['points']) and torch.equal(x['logits'],y['logits']) for x,y in zip(a,b))
    report['zero_detail_forward_exact']=True
    optimizer=optimizer_for(model)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt_initial=copy.deepcopy(optimizer.state_dict())
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    def step(side,bf16=False):
        nonlocal images,refs,boxes,truth,field,memberships,taps,rt
        if side!=images.shape[-1]:
            images,refs,boxes,truth,field,memberships=fixture(side,'cuda');taps=None;rt=None
        optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            rows=model(images,refs,boxes,[(side,side)]*4,taps=taps,reference_taps=rt)
        losses=[ops.view_loss(r,g) for r,g in zip(rows,truth)]
        measures=torch.stack([ops.splat(r['points'],r['probabilities'],field) for r in rows])
        gt=torch.stack([ops.splat(g,torch.ones(len(g),device='cuda'),field) for g in truth])
        terms=ops.response_terms(measures,gt,memberships,[1,1,0,2])
        loss=torch.stack([v['total'] for v in losses]).mean()+.25*terms['first']+.10*terms['mixed']
        assert torch.isfinite(loss);loss.backward()
        grads={n:p.grad for n,p in model.named_parameters() if p.requires_grad}
        assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
        assert all(p.grad is None for p in model.encoder.parameters())
        gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();report['disposable_optimizer_calls']+=1
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
        torch.cuda.synchronize()
        entry={'side':side,'bf16':bf16,'loss':loss.item(),'gradient_norm_before_clip':gradnorm.item(),'seconds':time.perf_counter()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'candidate_counts':[len(r['points']) for r in rows],'soft_counts':[r['probabilities'].sum().item() for r in rows],
            'splat_mass_errors':[abs(measures[i].sum().item()-r['probabilities'].sum().item()) for i,r in enumerate(rows)]}
        assert max(entry['splat_mass_errors'])<=1e-5+3e-3*max(entry['soft_counts'])
        report['steps'].append(entry);save()
        return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},copy.deepcopy(optimizer.state_dict())
    first,ofirst=step(224)
    model.load_state_dict(initial);optimizer.load_state_dict(opt_initial);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    second,osecond=step(224)
    report['recovery_model_max_abs']=state_diff(first,second);report['recovery_adam_max_abs']=state_diff(ofirst,osecond)
    assert report['recovery_model_max_abs']==0 and report['recovery_adam_max_abs']==0
    for name,p in model.encoder.state_dict().items():assert torch.equal(initial['encoder.'+name],p.detach().cpu())
    del first,second,ofirst,osecond
    assert torch.cuda.is_bf16_supported();step(224,bf16=True)
    step(896,bf16=True)
    for name,p in model.encoder.state_dict().items():assert torch.equal(initial['encoder.'+name],p.detach().cpu())
    report['encoder_unchanged']=True;report['persistent_steps']=0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--assets',required=True);ap.add_argument('--cuda',action='store_true');args=ap.parse_args()
    out=Path(args.output);report={'state':'running','device':'cuda' if args.cuda else 'cpu','disposable_optimizer_calls':0,'persistent_steps':0,'steps':[],
        'purpose':'engineering artificial fixtures; no natural labels, training or accuracy claims','rtol':.003,'atol':1e-5}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2))
    save()
    try:
        configure();report['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'strict_determinism':True,'tf32':False}
        report['operators']=operator_checks(report['device']);save()
        if args.cuda:model_checks(args,report,save)
        report['state']='passed';save();print(json.dumps(report,indent=2))
    except Exception:
        report['state']='failed';report['error']=traceback.format_exc();save();raise
if __name__=='__main__':main()
