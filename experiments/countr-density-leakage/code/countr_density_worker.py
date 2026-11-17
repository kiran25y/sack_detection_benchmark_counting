"""One guarded density-loss test. Reuse the unmodified CounTR loading/data/runtime."""
import argparse, copy, gc, json, math, time
from pathlib import Path
import numpy as np
import torch
import countr_sack_worker as base
from countr_density_losses import objective, terms, unit_checks


def update(model,opt,data,draws,step,spec,arm,coefficient):
    lr=spec['learning_rate']*((step+1)/spec['warmup'] if step<spec['warmup'] else .01+.99*.5*(1+math.cos(math.pi*(step-spec['warmup'])/(spec['updates']-spec['warmup']))))
    for g in opt.param_groups:g['lr']=lr
    opt.zero_grad(set_to_none=True);losses=[];parts=[];ledger=[];device=next(model.parameters()).device;refs=data.refs.to(device)
    for d in draws:
        x,t,m,h=data.sample(d,device);pred=model(x,refs,2)[0]
        loss,components=objective(pred,t,m,arm,coefficient);assert torch.isfinite(loss)
        (loss/len(draws)).backward();losses.append(float(loss.detach()));parts.append(components)
        ledger.append(dict(draw=d,hashes=h,target_patch_count=float(t[m].sum()/60)))
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step();torch.cuda.synchronize()
    return dict(loss=float(np.mean(losses)),micro_losses=losses,components=parts,coefficient=coefficient,gradient_norm=float(norm),lr=lr),ledger


def snapshot(model,opt):
    return copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state(),torch.cuda.get_rng_state_all()


def restore(model,opt,s):
    model.load_state_dict(s[0]);opt.load_state_dict(s[1]);torch.set_rng_state(s[2]);torch.cuda.set_rng_state_all(s[3])


def calibrate(model,data,spec):
    """One deterministic train-only rule: equal initial output-gradient L2 norms."""
    before=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all();record=[];squares=dict(mse=0.,C=0.,L=0.)
    for d in spec['calibration_draws']:
        assert data.byid[d['query_id']]['split']=='train'
        x,t,m,h=data.sample(d,'cuda')
        with torch.no_grad():z=model(x,data.refs.cuda(),2)[0]
        z=z.detach().requires_grad_(True);mse,c,l=terms(z,t,m)
        norms={}
        for key,f in [('mse',mse),('C',c),('L',l)]:
            grad=torch.autograd.grad(f,z,retain_graph=True)[0]
            norms[key]=float(grad.double().square().sum());squares[key]+=norms[key]
        record.append(dict(draw=d,input_hashes=h,target_patch_count=float(t[m].sum()/60),gradient_squared_norms=norms))
    torch.set_rng_state(before);torch.cuda.set_rng_state_all(cuda)
    assert all(math.isfinite(v) and v>1e-20 for v in squares.values())
    coefficients={arm:math.sqrt(squares['mse']/squares[arm]) for arm in ['C','L']}
    assert all(math.isfinite(v) and 0<v<1e4 for v in coefficients.values())
    return dict(rule=spec['calibration_rule'],coefficients=coefficients,squared_norm_sums=squares,training_records=record,development_used=False,optimizer_calls=0)


def preflight(root,package,out,spec,data,draws):
    report=dict(state='running',passed=False,optimizer_calls=0,persistent_steps=0,plan_sha256=base.sha(package/'plan.json'),unit_checks=unit_checks())
    model,opt=base.initialize(root,spec,'P');initial=base.identities(model,opt)
    assert initial==spec['reused_initial_identity'];report['initial']=initial
    initial_snapshot=snapshot(model,opt)
    calibration=calibrate(model,data,spec);assert base.identities(model,opt)==initial
    base.write(out/'calibration.json',calibration);report['calibration_sha256']=base.sha(out/'calibration.json');report['calibration']=calibration
    # Exact first-update equality validates reuse of the already-completed original MSE arm.
    old_metrics,old_ledger=base.update(model,opt,data,draws[:4],0,spec);report['optimizer_calls']+=1
    old_identity=base.identities(model,opt);restore(model,opt,initial_snapshot)
    new_metrics,new_ledger=update(model,opt,data,draws[:4],0,spec,'C',0.);report['optimizer_calls']+=1
    assert base.identities(model,opt)==old_identity and old_ledger==new_ledger
    for k in old_metrics:assert old_metrics[k]==new_metrics[k],k
    for k in old_metrics:assert old_metrics[k]==spec['reused_first_update'][k],k
    report['mse_update_exact']=True;report['matches_historical_first_update']=True
    report['arm_checks']={}
    for arm in ['C','L']:
        restore(model,opt,initial_snapshot)
        metrics,_=update(model,opt,data,draws[:4],0,spec,arm,calibration['coefficients'][arm]);report['optimizer_calls']+=1
        base.finite(model,opt);saved=snapshot(model,opt);identity=base.identities(model,opt)
        assert identity['encoder']==initial['encoder'] and identity['model']!=initial['model']
        update(model,opt,data,draws[4:8],1,spec,arm,calibration['coefficients'][arm]);report['optimizer_calls']+=1;expected=base.identities(model,opt)
        restore(model,opt,saved);assert base.identities(model,opt)==identity
        update(model,opt,data,draws[4:8],1,spec,arm,calibration['coefficients'][arm]);report['optimizer_calls']+=1
        base.finite(model,opt);assert base.identities(model,opt)==expected
        report['arm_checks'][arm]=dict(finite=True,exact_recovery=True,encoder_unchanged=expected['encoder']==initial['encoder'],first_metrics=metrics)
        del saved
    assert report['optimizer_calls']==8
    report.update(state='completed',passed=True);base.write(out/'report.json',report);base.write(out/'status.json',report)


def train(root,package,out,spec,data,draws,arm):
    adm=json.loads((package/'cuda_admission.json').read_text());gd=root/adm['remote']
    for filename,key in [('result/report.json','report_sha256'),('result/calibration.json','calibration_sha256'),('exit.json','exit_sha256')]:assert base.sha(gd/filename)==adm[key]
    guard=json.loads((gd/'result/report.json').read_text());ex=json.loads((gd/'exit.json').read_text());calibration=json.loads((gd/'result/calibration.json').read_text())
    assert guard['passed'] and guard['optimizer_calls']==8 and guard['persistent_steps']==0 and guard['plan_sha256']==base.sha(package/'plan.json')
    assert ex['returncode']==0 and not ex['guard_stop'] and not ex['docker_state']['OOMKilled']
    assert calibration==guard['calibration']==adm['calibration']
    model,opt=base.initialize(root,spec,'P');initial=base.identities(model,opt);assert initial==guard['initial']==spec['reused_initial_identity']
    coefficient=calibration['coefficients'][arm]
    state=dict(state='training',arm=arm,durable_steps=0,total_steps=300,initial=initial,plan_sha256=base.sha(package/'plan.json'),calibration_sha256=adm['calibration_sha256'],coefficient=coefficient)
    base.write(out/'status.json',state);base.write(out/'calibration.json',calibration);times=[]
    for step in range(300):
        start=time.monotonic();metrics,ledger=update(model,opt,data,draws[step*4:(step+1)*4],step,spec,arm,coefficient);times.append(time.monotonic()-start)
        with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(update=step+1,seconds=times[-1],**metrics))+'\n')
        with (out/'inputs.jsonl').open('a') as f:
            for record in ledger:f.write(json.dumps(record)+'\n')
        if (step+1)%25==0:
            base.finite(model,opt);assert base.identities(model,opt)['encoder']==initial['encoder']
            ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),steps=step+1,arm=arm,seed=1729,plan_sha256=state['plan_sha256'],calibration_sha256=state['calibration_sha256'],cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            torch.save(ck,out/'resume.tmp');(out/'resume.tmp').replace(out/'resume.pt')
            state.update(durable_steps=step+1,recent_seconds_per_update=float(np.median(times[-25:])),saved_host_time=time.time());base.write(out/'status.json',state)
    base.evaluate(model,data,out/'predictions');base.finite(model,opt)
    state.update(state='completed',finite_model_optimizer=True,final=base.identities(model,opt),resume_sha256=base.sha(out/'resume.pt'));base.write(out/'status.json',state)
    base.write(out/'report.json',dict(state='completed',optimizer_calls=300,arm=arm,scientific_gate_pending=True))


def main():
    ap=argparse.ArgumentParser()
    for n in ['root','package','output','mode']:ap.add_argument('--'+n,required=True)
    ap.add_argument('--arm',choices=['C','L']);args=ap.parse_args();root=Path(args.root);package=Path(args.package);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    base.setup();spec=json.loads((package/'plan.json').read_text());assert spec['arms']==['C','L'] and spec['updates']==300 and spec['microbatch']==4
    for n,h in spec['data_hashes'].items():assert base.sha(package/n)==h,n
    data=base.Data(package);draws=json.loads((package/'draws.json').read_text());assert len(draws)==1200
    try:
        if args.mode=='preflight':preflight(root,package,out,spec,data,draws)
        else:train(root,package,out,spec,data,draws,args.arm)
        base.write(out/'result_hashes.json',{p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*') if p.is_file() and p.name!='result_hashes.json'})
    except Exception as e:
        s=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {};s.update(state='failed',error=repr(e));base.write(out/'status.json',s);raise


if __name__=='__main__':main()
