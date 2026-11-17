"""One matched, bounded continuation. No automatic recovery or model selection."""
import argparse,copy,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from counting.ipic.model import IPICNet
from counting.ipic.training import make_optimizer,optimizer_update,set_learning_rate
from counting.ipic.transfer_data import TransferData
from counting.ipic.transfer_worker import evaluate,move,array_hashes
from counting.ipic.stable_adaptation_runtime import install
from counting.ipic.adaptation_preflight import finite
from counting.ipic.assignment_cudnn_preflight import state_digest
from counting.ipic.foundation import write_json
from ipic_objectness_loss import pair_loss,check_contract

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def initialize(root,spec):
    p=root/spec['resume_path'];assert sha(p)==spec['resume_sha256']
    ck=torch.load(p,map_location='cpu',weights_only=True)
    assert ck['optimizer_steps']==200 and ck['cell']=='P11' and ck['seed']==1729
    model=IPICNet(detail=False);model.load_state_dict(ck['model'],strict=True);model.cuda().train()
    opt=make_optimizer(model);opt.load_state_dict(ck['optimizer'])
    torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);del ck
    finite(model,opt)
    return model,opt

def digest(model,opt):return dict(model=state_digest(model.state_dict()),optimizer=state_digest(opt.state_dict()))

def batch(data,dd):
    micro=[];records=[]
    for d in dd:
        b,r=data.prepare(d,'cpu');r['array_hashes']=array_hashes(b)
        micro.append({k:move(v,'cuda') for k,v in b.items()});records.append(r)
    return micro,records

def update(model,opt,micro,arm,step,spec):
    set_learning_rate(opt,step,spec['updates_per_arm'],spec['warmup'])
    m=optimizer_update(model,opt,micro,arm,loss_function=pair_loss);torch.cuda.synchronize()
    for r in m['microbatch_losses']:
        for v in ['view_a','view_b']:r[v]['classification']=r[v].pop('focal')
    return m

def preflight(root,package,out,spec,draws):
    checks=check_contract();data=TransferData(root,package,'P11');sacks=[d for d in draws if d['dataset']=='sacks']
    panel=[max(sacks,key=lambda d:data.train[d['row_index']]['count']),min(sacks,key=lambda d:data.train[d['row_index']]['count'])]+[d for d in draws if d['dataset']!='sacks'][:6]
    micro,records=batch(data,panel);report=dict(state='running',stage='objectness_preflight',passed=False,optimizer_calls=0,persistent_steps=0,contract=checks,updates=[],spec_sha256=sha(package/'objectness_plan.json'))
    first=None
    for arm in ['F','U','H']:
        model,opt=initialize(root,spec);initial=digest(model,opt)
        if first is None:first=initial
        assert initial==first
        if arm=='H':
            snapshot=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        m=update(model,opt,micro,arm,0,spec);finite(model,opt);report['updates'].append(dict(arm=arm,**m));report['optimizer_calls']+=1
        if arm=='H':
            expected=digest(model,opt)
            model.load_state_dict(snapshot['model']);opt.load_state_dict(snapshot['optimizer']);torch.set_rng_state(snapshot['cpu_rng']);torch.cuda.set_rng_state_all(snapshot['cuda_rng']);assert digest(model,opt)==first
            update(model,opt,micro,arm,0,spec);finite(model,opt);report['optimizer_calls']+=1
            assert digest(model,opt)==expected,'Recovery update differs'
            report['recovery']=dict(preupdate_exact=True,postupdate_exact=True);del snapshot
        del model,opt;gc.collect();torch.cuda.empty_cache()
    report.update(state='completed',passed=True,initial_identity=first,finite_model_optimizer=True,actual_two_view_updates=True,panel=[dict(query_id=r['query_id'],dataset=r['dataset'],count=r['count_a']) for r in records])
    assert report['optimizer_calls']==4
    write_json(out/'report.json',report);write_json(out/'status.json',dict(state='completed',optimizer_calls=4,persistent_steps=0))

def train(root,package,out,spec,draws,arm):
    admission=json.loads((package/'objectness_cuda_admission.json').read_text());guard=root/admission['remote']
    assert sha(guard/'result/report.json')==admission['report_sha256'] and sha(guard/'exit.json')==admission['exit_sha256']
    g=json.loads((guard/'result/report.json').read_text());ex=json.loads((guard/'exit.json').read_text())
    assert g['passed'] and g['spec_sha256']==sha(package/'objectness_plan.json') and g['optimizer_calls']==4 and g['persistent_steps']==0 and g['recovery']['postupdate_exact']
    assert ex['returncode']==0 and not ex['docker_state']['OOMKilled'] and not ex['guard_stop']
    data=TransferData(root,package,'P11');model,opt=initialize(root,spec);assert digest(model,opt)==g['initial_identity']
    state=dict(state='training',arm=arm,new_durable_steps=0,starting_steps=200,total_new_steps=spec['updates_per_arm'],optimizer_moments_preserved=True)
    write_json(out/'status.json',state);write_json(out/'plan.json',dict(spec,arm=arm));ph=sha(out/'plan.json');times=[]
    for step in range(spec['updates_per_arm']):
        start=time.monotonic();micro,records=batch(data,draws[step*8:(step+1)*8]);m=update(model,opt,micro,arm,step,spec);times.append(time.monotonic()-start)
        with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(update=step+1,classification_objective=spec['arms'][arm],seconds=times[-1],**m))+'\n')
        with (out/'augmentation_ledger.jsonl').open('a') as f:
            for r in records:f.write(json.dumps(r)+'\n')
        del micro,records
        if (step+1)%25==0:
            finite(model,opt);temp=out/'resume.tmp';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),optimizer_steps=200+step+1,new_optimizer_steps=step+1,arm=arm,seed=1729,plan_sha256=ph,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp);temp.replace(out/'resume.pt')
            state.update(new_durable_steps=step+1,recent_seconds_per_update=float(np.median(times[-25:])),saved_host_time=time.time());write_json(out/'status.json',state)
    finite(model,opt);torch.save(dict(model=model.state_dict(),optimizer_steps=400,new_optimizer_steps=200,arm=arm,seed=1729,plan_sha256=ph),out/'counter.pt')
    evaluate(model,data,out,400)
    trainout=out/'fitting';trainout.mkdir();data.dev=data.train;evaluate(model,data,trainout,400)
    state.update(state='completed',finite_model_optimizer=True,checkpoint_sha256=sha(out/'counter.pt'),resume_sha256=sha(out/'resume.pt'))
    write_json(out/'status.json',state);write_json(out/'report.json',dict(state='completed',passed=True,arm=arm,optimizer_calls=200,total_cumulative_steps=400,development_queries=23,fitting_queries=42,scientific_gate_evaluated=False))

def main():
    ap=argparse.ArgumentParser()
    for n in ['root','package','output','mode']:ap.add_argument('--'+n,required=True)
    ap.add_argument('--arm',choices=['F','U','H']);a=ap.parse_args();root=Path(a.root);package=Path(a.package);out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(2);runtime=install();assert runtime['strict_deterministic_algorithms']
    spec=json.loads((package/'objectness_plan.json').read_text())
    assert spec['updates_per_arm']==200 and spec['warmup']==20 and spec['thresholds']==[.5,.1]
    for n,h in spec['data_hashes'].items():assert sha(package/n)==h,('Changed data',n)
    draws=list(map(json.loads,(package/'draws_P11.jsonl').read_text().splitlines()));assert len(draws)==1600
    if a.mode=='preflight':preflight(root,package,out,spec,draws)
    elif a.mode=='train' and a.arm:train(root,package,out,spec,draws,a.arm)
    else:raise ValueError('Invalid stage')
    write_json(out/'result_hashes.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='result_hashes.json'})

if __name__=='__main__':main()
