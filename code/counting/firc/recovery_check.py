"""Supplemental nonempty-Adam recovery; three disposable engineering calls."""
import argparse,copy,hashlib,json,time,traceback
from pathlib import Path
from .preflight import configure,fixture,state_diff
from .model import FIRC,load_official_encoder,optimizer_for
from . import torch_ops as ops
import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--assets',required=True);ap.add_argument('--cuda',action='store_true');args=ap.parse_args()
    out=Path(args.output);r={'state':'running','disposable_optimizer_calls':0,'persistent_steps':0,'steps':[],'purpose':'Nonempty Adam/RNG recovery supplement; artificial pixels only'}
    def save():(out/'report.json').write_text(json.dumps(r,indent=2))
    save()
    try:
        configure();checkpoint=Path(args.assets)/'dinov2_vitb14_reg4_pretrain.pth'
        with checkpoint.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71'
        model=FIRC(load_official_encoder(checkpoint)).cuda().train();opt=optimizer_for(model)
        images,refs,boxes,truth,field,membership=fixture(224,'cuda');taps=model.encode(images);rt=model.encode(refs)
        def update():
            opt.zero_grad(set_to_none=True);rows=model(images,refs,boxes,[(224,224)]*4,taps=taps,reference_taps=rt)
            loss=torch.stack([ops.view_loss(a,b)['total'] for a,b in zip(rows,truth)]).mean()
            pred=torch.stack([ops.splat(a['points'],a['probabilities'],field) for a in rows]);gt=torch.stack([ops.splat(a,torch.ones(len(a),device='cuda'),field) for a in truth])
            t=ops.response_terms(pred,gt,membership,[1,1,0,2]);loss=loss+.25*t['first']+.1*t['mixed'];loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();r['disposable_optimizer_calls']+=1;r['steps'].append({'loss':loss.item()});save()
            return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},copy.deepcopy(opt.state_dict())
        initial_encoder={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
        primed,primed_opt=update();assert primed_opt['state']
        assert any(v['exp_avg'].abs().max().item()>0 for v in primed_opt['state'].values())
        cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
        expected,eo=update()
        model.load_state_dict(primed);opt.load_state_dict(primed_opt);torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
        assert state_diff(primed_opt,opt.state_dict())==0
        actual,ao=update()
        r['primed_adam_entries']=len(primed_opt['state']);r['model_max_abs']=state_diff(expected,actual);r['adam_max_abs']=state_diff(eo,ao)
        assert r['model_max_abs']==0 and r['adam_max_abs']==0
        r['encoder_max_abs']=state_diff(initial_encoder,{k:v.detach().cpu() for k,v in model.encoder.state_dict().items()});assert r['encoder_max_abs']==0
        r['state']='passed';save();print(json.dumps(r,indent=2))
    except Exception:r['state']='failed';r['error']=traceback.format_exc();save();raise
if __name__=='__main__':main()
