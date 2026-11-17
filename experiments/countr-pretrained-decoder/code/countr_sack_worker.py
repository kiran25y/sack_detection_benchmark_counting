"""One frozen pretrained-density experiment. No threshold or arm selection."""
import argparse,copy,gc,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
import countr_model as cm
from counting.ipic.stable_resampling import bilinear2x, native_functional
from countr_sack_metrics import score,summary
def state_digest(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y=x.detach().cpu().contiguous();h.update(str((y.dtype,tuple(y.shape))).encode());h.update(y.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for y in x:visit(y)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def encoder_key(k):return k.startswith(('patch_embed.','blocks.','norm.')) or k=='pos_embed'
def setup():
    torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    class Proxy:
        @staticmethod
        def interpolate(x,size=None,**kw):return bilinear2x(x,size=(size,size) if isinstance(size,int) else size,**kw)
        def __getattr__(self,name):return getattr(native_functional,name)
    torch.use_deterministic_algorithms(True);cm.F=Proxy()
def initialize(root,spec,arm,device='cuda'):
    torch.manual_seed(1729);model=cm.mae_vit_base_patch16()
    path=root/spec['weights_path'];assert sha(path)==spec['weights_sha256']
    with torch.serialization.safe_globals([argparse.Namespace]):ck=torch.load(path,map_location='cpu',weights_only=True)
    state=ck['model'];assert set(state)==set(model.state_dict()),'Non-strict official architecture'
    random_decoder={k:v.clone() for k,v in model.state_dict().items() if not encoder_key(k)} if arm=='R' else {}
    model.load_state_dict(state,strict=True);del ck,state
    if arm=='R':model.load_state_dict(random_decoder,strict=False)
    del random_decoder
    for k,p in model.named_parameters():p.requires_grad_(not encoder_key(k) and k not in ['decoder_pos_embed','shot_token'])
    model.to(device).train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=spec['learning_rate'],weight_decay=.05)
    return model,opt
def identities(model,opt):
    return dict(model=state_digest(model.state_dict()),encoder=state_digest({k:v for k,v in model.state_dict().items() if encoder_key(k)}),optimizer=state_digest(opt.state_dict()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
def finite(model,opt):
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
class Data:
    def __init__(self,package):
        self.package=package;self.rows=json.loads((package/'manifest.json').read_text());self.byid={r['image_id']:r for r in self.rows};self.arrays={}
        for r in self.rows:
            assert sha(package/r['array'])==r['array_sha256'];a=dict(np.load(package/r['array']));assert abs(a['target'].sum(dtype=np.float64)/60-r['count'])<1e-5;self.arrays[r['image_id']]=a
        self.refs=torch.from_numpy(np.load(package/'references.npy').astype(np.float32).transpose(0,3,1,2)/255)[None]
    def sample(self,d,device):
        a=self.arrays[d['query_id']];l=d['left'];im=a['rgb'][:,l:l+384].astype(np.float32)/255;t=a['target'][:,l:l+384];m=a['mask'][:,l:l+384]
        if d['flip']:im=im[:,::-1];t=t[:,::-1];m=m[:,::-1]
        im=np.clip(im*d['brightness'],0,1).transpose(2,0,1).copy();t=t.copy();m=m.copy()
        h={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('image',im),('target',t),('mask',m)]}
        return torch.from_numpy(im)[None].to(device),torch.from_numpy(t).to(device),torch.from_numpy(m).to(device),h
def update(model,opt,data,draws,step,spec):
    lr=spec['learning_rate']*((step+1)/spec['warmup'] if step<spec['warmup'] else .01+.99*.5*(1+math.cos(math.pi*(step-spec['warmup'])/(spec['updates']-spec['warmup']))))
    for g in opt.param_groups:g['lr']=lr
    opt.zero_grad(set_to_none=True);losses=[];ledger=[];device=next(model.parameters()).device;refs=data.refs.to(device)
    for d in draws:
        x,t,m,h=data.sample(d,device);pred=model(x,refs,2)[0];loss=((pred-t)**2)[m].mean();assert torch.isfinite(loss)
        (loss/len(draws)).backward();losses.append(float(loss.detach()));ledger.append(dict(draw=d,hashes=h,target_patch_count=float(t[m].sum()/60)))
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step();torch.cuda.synchronize();return dict(loss=float(np.mean(losses)),micro_losses=losses,gradient_norm=float(norm),lr=lr),ledger
def evaluate(model,data,out):
    out.mkdir(exist_ok=False);model.eval();device=next(model.parameters()).device;rows=[];refs=data.refs.to(device)
    with torch.inference_mode():
        for r in data.rows:
            a=data.arrays[r['image_id']];H,W=a['mask'].shape;starts=sorted(set(list(range(0,W-383,128))+[W-384]));raw=np.zeros((H,W),np.float64);weight=np.zeros((H,W),np.float64)
            for left in starts:
                x=torch.from_numpy((a['rgb'][:,left:left+384].astype(np.float32)/255).transpose(2,0,1).copy())[None].to(device)
                pred=model(x,refs,2)[0].cpu().numpy();assert np.isfinite(pred).all();raw[:,left:left+384]+=pred;weight[:,left:left+384]+=1
            assert (weight>0).all();raw=(raw/weight).astype(np.float32);np.save(out/(r['image_id']+'.npy'),raw)
            rows.append(dict(split=r['split'],**score(raw,a['mask'],r)))
    scores={name:summary([r for r in rows if r['split']==split]) for name,split in [('fitting','train'),('development','development')]}
    write(out/'metrics.json',scores);model.train();return scores
def preflight(root,package,out,spec,data,draws):
    report=dict(state='running',passed=False,optimizer_calls=0,persistent_steps=0,identities={},plan_sha256=sha(package/'plan.json'))
    train=[r for r in data.rows if r['split']=='train'];ids=[max(train,key=lambda r:r['count'])['image_id'],next(r['image_id'] for r in train if r['count']==0),next(r['image_id'] for r in train if 1<=r['count']<=11),train[0]['image_id']]
    panel=[dict(draw_id=i,query_id=q,left=0,flip=False,brightness=1.) for i,q in enumerate(ids)]
    # CPU/CUDA agreement with original native forward operators; fixed adjoint only alters backward.
    model,opt=initialize(root,spec,'P','cpu');x,_,_,_=data.sample(panel[0],'cpu')
    with torch.inference_mode():cpu=model(x,data.refs,2)
    model.cuda()
    with torch.inference_mode():gpu=model(x.cuda(),data.refs.cuda(),2).cpu()
    torch.testing.assert_close(cpu,gpu,rtol=.003,atol=1e-5);report['forward_parity_max_abs']=float((cpu-gpu).abs().max());del model,opt,cpu,gpu;gc.collect();torch.cuda.empty_cache()
    for arm in ['R','P']:
        model,opt=initialize(root,spec,arm);initial=identities(model,opt);report['identities'][arm]=initial
        m,_=update(model,opt,data,panel,0,spec);report['optimizer_calls']+=1;finite(model,opt)
        assert identities(model,opt)['encoder']==initial['encoder'];assert identities(model,opt)['model']!=initial['model']
        if arm=='P':
            ck=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all());before=identities(model,opt)
            update(model,opt,data,panel,1,spec);report['optimizer_calls']+=1;expected=identities(model,opt)
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);assert identities(model,opt)==before
            update(model,opt,data,panel,1,spec);report['optimizer_calls']+=1;finite(model,opt);assert identities(model,opt)==expected,'Recovery differs'
            report['recovery_exact']=True;del ck
        del model,opt;gc.collect();torch.cuda.empty_cache()
    assert report['identities']['P']['encoder']==report['identities']['R']['encoder'] and report['identities']['P']['trainable']==report['identities']['R']['trainable']
    assert report['optimizer_calls']==4;report.update(state='completed',passed=True,finite=True);write(out/'report.json',report);write(out/'status.json',report)
def train(root,package,out,spec,data,draws,arm):
    adm=json.loads((package/'cuda_admission.json').read_text());gd=root/adm['remote']
    assert sha(gd/'result/report.json')==adm['report_sha256'] and sha(gd/'exit.json')==adm['exit_sha256']
    guard=json.loads((gd/'result/report.json').read_text());ex=json.loads((gd/'exit.json').read_text());assert guard['passed'] and guard['optimizer_calls']==4 and guard['plan_sha256']==sha(package/'plan.json') and ex['returncode']==0 and not ex['guard_stop'] and not ex['docker_state']['OOMKilled']
    model,opt=initialize(root,spec,arm);initial=identities(model,opt);assert initial==guard['identities'][arm]
    state=dict(state='training',arm=arm,durable_steps=0,total_steps=300,initial=initial,plan_sha256=sha(package/'plan.json'));write(out/'status.json',state)
    if arm=='P':evaluate(model,data,out/'baseline_B')
    times=[]
    for step in range(300):
        t=time.monotonic();metrics,ledger=update(model,opt,data,draws[step*4:(step+1)*4],step,spec);times.append(time.monotonic()-t)
        with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(update=step+1,seconds=times[-1],**metrics))+'\n')
        with (out/'inputs.jsonl').open('a') as f:
            for record in ledger:f.write(json.dumps(record)+'\n')
        if (step+1)%25==0:
            finite(model,opt);assert identities(model,opt)['encoder']==initial['encoder']
            ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),steps=step+1,arm=arm,seed=1729,plan_sha256=sha(package/'plan.json'),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all());torch.save(ck,out/'resume.tmp');(out/'resume.tmp').replace(out/'resume.pt')
            state.update(durable_steps=step+1,recent_seconds_per_update=float(np.median(times[-25:])),saved_host_time=time.time());write(out/'status.json',state)
    evaluate(model,data,out/'predictions');finite(model,opt);state.update(state='completed',finite_model_optimizer=True,final=identities(model,opt),resume_sha256=sha(out/'resume.pt'));write(out/'status.json',state)
    write(out/'report.json',dict(state='completed',optimizer_calls=300,arm=arm,scientific_gate_pending=True))
def main():
    ap=argparse.ArgumentParser()
    for n in ['root','package','output','mode']:ap.add_argument('--'+n,required=True)
    ap.add_argument('--arm',choices=['R','P']);a=ap.parse_args();root=Path(a.root);package=Path(a.package);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);setup()
    spec=json.loads((package/'plan.json').read_text());assert spec['updates']==300 and spec['microbatch']==4 and spec['arms']==['R','P']
    for n,h in spec['data_hashes'].items():assert sha(package/n)==h,n
    data=Data(package);draws=json.loads((package/'draws.json').read_text());assert len(draws)==1200
    try:
        if a.mode=='preflight':preflight(root,package,out,spec,data,draws)
        else:train(root,package,out,spec,data,draws,a.arm)
        write(out/'result_hashes.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='result_hashes.json'})
    except Exception as e:
        state=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {};state.update(state='failed',error=repr(e));write(out/'status.json',state);raise
if __name__=='__main__':main()
