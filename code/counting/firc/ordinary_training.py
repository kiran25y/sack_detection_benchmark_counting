"""Fresh ordinary public foundation then sack adaptation; no quartet claims."""
import argparse,copy,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from .ordinary_data import Data
from .model import FIRC,load_official_encoder,optimizer_for
from .preflight import configure,state_diff
from .tiny_fit import digest,state_digest,write_json
from .torch_ops import view_loss

def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value

def head(model):return {k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')}
def phase(model,enabled):
    for p in model.parameters():p.requires_grad_(True)
    model.encoder.requires_grad_(False)
    model.detail.requires_grad_(enabled);model.alpha.requires_grad_(enabled)
    model.train();return optimizer_for(model)

def update(model,opt,loader,rows,draws,lr,detail):
    assert len(draws)==8;opt.zero_grad(set_to_none=True)
    for group in opt.param_groups:group['lr']=lr
    logs=[];start=time.monotonic()
    for draw in draws:
        row=rows[draw['row_index']];batch,ledger=loader.prepare(row,draw)
        images=torch.from_numpy(batch['image'])[None].cuda();refs=torch.from_numpy(batch['references']).cuda()
        boxes=torch.from_numpy(batch['boxes']).cuda();truth=torch.from_numpy(batch['truth']).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=model(images,refs,boxes,[batch['valid_hw']],detail_enabled=detail)[0]
            terms=view_loss(result,truth);loss=(terms['focal']+5*terms['point'])/8
        assert torch.isfinite(loss);loss.backward()
        ledger.update(focal=float(terms['focal'].detach()),point=float(terms['point'].detach()),
                      soft_count=float(result['probabilities'].detach().sum()),hard_count=int((result['probabilities'].detach()>=.5).sum()),matched=terms['matched'])
        assert terms['matched']==len(truth);logs.append(ledger)
        del result,terms,loss,images,refs,boxes,truth,batch
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.encoder.parameters())
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    assert torch.isfinite(norm);opt.step()
    assert all(torch.isfinite(v).all() for v in head(model).values())
    assert all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
    torch.cuda.synchronize()
    return {'lr':lr,'gradient_norm_before_clip':float(norm),'seconds':time.monotonic()-start,'queries':logs,'finite_model_adam':True}

def run(args):
    data,assets,out=Path(args.data),Path(args.assets),Path(args.output)
    plan=json.loads((data/'plan.json').read_text())
    for k,h in json.loads((data/'hashes.json').read_text()).items():assert digest(data/k)==h,k
    configure();assert torch.cuda.is_bf16_supported()
    model=FIRC(load_official_encoder(assets/'dinov2_vitb14_reg4_pretrain.pth')).cuda()
    assert digest(assets/'dinov2_vitb14_reg4_pretrain.pth')==plan['pretrained_sha256']
    rows=[json.loads(s) for s in (data/'rows.jsonl').read_text().splitlines()]
    loader=Data(data,args.project,plan);encoder=state_digest(model.encoder.state_dict());initial_head=state_digest(head(model))
    report={'state':'running','mode':args.mode,'optimizer_calls':0,'saved_updates':0,'stages':{},
            'plan_sha256':digest(data/'plan.json'),'encoder_initial_sha256':encoder,'fresh_head_sha256':initial_head,
            'diagnostic_weights_loaded':False,'physical_quartets':0,'finite_model_adam':True}
    write_json(out/'status.json',report);start=time.monotonic()
    if args.mode=='preflight':
        # Audited real data covers both public categories, sacks, all scales and worst FSC count.
        draws=plan['preflight_draws'];report['checks']={};opt=phase(model,False)
        with (out/'preflight_updates.jsonl').open('x') as log:
            for name,detail in [('foundation',False),('adaptation',True)]:
                opt=phase(model,detail)
                for step in [1,2,2]:
                    if step==1:
                        entry=update(model,opt,loader,rows,draws,2e-4/(80 if not detail else 150),detail)
                        saved={'head':cpu(head(model)),'optimizer':cpu(opt.state_dict()),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
                    elif name+'_expected' not in report['checks']:
                        entry=update(model,opt,loader,rows,draws,4e-4/(80 if not detail else 150),detail)
                        expected_head=cpu(head(model));expected_opt=cpu(opt.state_dict());report['checks'][name+'_expected']=True
                        missing=model.load_state_dict(saved['head'],strict=False);assert not missing.unexpected_keys
                        opt.load_state_dict(copy.deepcopy(saved['optimizer']));torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
                        assert state_diff(saved['head'],cpu(head(model)))==0 and state_diff(saved['optimizer'],cpu(opt.state_dict()))==0
                    else:
                        entry=update(model,opt,loader,rows,draws,4e-4/(80 if not detail else 150),detail)
                        diff={'model':state_diff(expected_head,cpu(head(model))),'optimizer':state_diff(expected_opt,cpu(opt.state_dict()))}
                        assert max(diff.values())==0,diff;report['checks'][name+'_exact_recovery']=diff
                    report['optimizer_calls']+=1;entry.update(phase=name,call=report['optimizer_calls'],schedule_step=step)
                    log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush();write_json(out/'status.json',report)
        report.update(state='completed',disposable_calls=6,persistent_steps=0,encoder_final_sha256=state_digest(model.encoder.state_dict()),elapsed_seconds=time.monotonic()-start,peak_gpu_allocated=torch.cuda.max_memory_allocated())
        assert encoder==report['encoder_final_sha256'];write_json(out/'report.json',report);write_json(out/'status.json',report)
        return
    assert args.mode=='train'
    with (out/'updates.jsonl').open('x') as log:
        for name in ['foundation','adaptation']:
            stage=plan['stages'][name];draws=[json.loads(s) for s in (data/stage['draw_file']).read_text().splitlines()]
            opt=phase(model,name=='adaptation');detail=name=='adaptation'
            a={'saved_updates':0,'optimizer_calls':0,'head_initial_sha256':state_digest(head(model)),'optimizer_reset':True,'completed':False}
            report['stages'][name]=a;report['active_stage']=name
            for step in range(1,stage['updates']+1):
                entry=update(model,opt,loader,rows,draws[(step-1)*8:step*8],stage['learning_rates'][step-1],detail)
                report['optimizer_calls']+=1;a['optimizer_calls']=step
                entry.update(stage=name,stage_step=step,call=report['optimizer_calls']);log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
                if step%25==0 or step==stage['updates']:
                    payload={'counter':cpu(head(model)),'optimizer':cpu(opt.state_dict()),'step':step,'stage':name,
                             'total_calls':report['optimizer_calls'],'seed':1729,'plan_sha256':report['plan_sha256'],
                             'encoder_sha256':encoder,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                             'research_initialization_allowed':True,'ordinary_supervision_only':True}
                    tmp=out/(name+'_resume.pt.tmp');torch.save(payload,tmp);tmp.replace(out/(name+'_resume.pt'))
                    a['saved_updates']=step;report['saved_updates']=sum(s['saved_updates'] for s in report['stages'].values());write_json(out/'status.json',report)
                print(json.dumps({'stage':name,'step':step,'total_calls':report['optimizer_calls'],'seconds':entry['seconds']}),flush=True)
            final={'counter':cpu(head(model)),'step':stage['updates'],'stage':name,'seed':1729,'plan_sha256':report['plan_sha256'],
                   'pretrained_sha256':plan['pretrained_sha256'],'ordinary_supervision_only':True,'research_initialization_allowed':True}
            torch.save(final,out/(name+'.pt'));a.update(completed=True,checkpoint_sha256=digest(out/(name+'.pt')),resume_sha256=digest(out/(name+'_resume.pt')),head_final_sha256=state_digest(head(model)))
            assert state_digest(model.encoder.state_dict())==encoder;write_json(out/'status.json',report)
    report.update(state='completed',elapsed_seconds=time.monotonic()-start,encoder_final_sha256=state_digest(model.encoder.state_dict()),peak_gpu_allocated=torch.cuda.max_memory_allocated(),query_presentations=report['optimizer_calls']*8)
    assert report['optimizer_calls']==report['saved_updates']==3800 and report['query_presentations']==30400
    write_json(out/'status.json',report);write_json(out/'report.json',report)
    write_json(out/'result_hashes.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name not in ('status.json','result_hashes.json')})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--project',required=True);ap.add_argument('--assets',required=True);ap.add_argument('--output',required=True);ap.add_argument('--mode',choices=['preflight','train'],required=True)
    args=ap.parse_args()
    try:run(args)
    except Exception:write_json(Path(args.output)/'failure.json',{'traceback':traceback.format_exc()});raise
