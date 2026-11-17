"""Matched learning-rate continuation with unchanged dynamic assignments; no research initializer output."""
import argparse, json, time, traceback
from pathlib import Path
import numpy as np
import torch
from .assignment_probe import Context, cpu_tree
from .preflight import state_diff
from .tiny_fit import digest, state_digest, write_json


def run(args):
    out=Path(args.output);ctx=Context(args.data,args.assets);plan=ctx.plan
    assert plan['updates']==150 and plan['max_optimizer_calls']==300 and plan['threshold']==.5
    assert plan['arms']==['reference_lr','lower_lr']
    baseline=json.loads((Path(args.data)/'control/report.json').read_text())
    report={'state':'training','saved_updates':0,'optimizer_calls':0,'arms':{},'plan_sha256':digest(Path(args.data)/'plan.json'),
            'initial_checkpoint_sha256':plan['initial_checkpoint_sha256'],'encoder_initial_sha256':ctx.initial_encoder,
            'purpose':plan['purpose'],'persistent_research_steps':0,'threshold':.5}
    write_json(out/'status.json',report);start=time.perf_counter()
    with (out/'updates.jsonl').open('x') as log:
        for arm in plan['arms']:
            ctx.restore();dest=out/arm;dest.mkdir(exist_ok=False)
            assert state_digest({k:v for k,v in ctx.model.state_dict().items() if not k.startswith('encoder.')})==ctx.initial_head
            assert state_diff(ctx.initial_payload['optimizer'],cpu_tree(ctx.optimizer.state_dict()))==0
            initial=ctx.save_predictions(dest,'initial')
            for e,b in zip(initial,baseline['final_predictions']):
                with np.load(dest/e['file'],allow_pickle=False) as x,np.load(Path(args.data)/'control'/b['file'],allow_pickle=False) as y:
                    assert x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
            with torch.no_grad():outputs=ctx.forward();signals,frozen=ctx.signals(outputs)
            fixed=[{'gt_indices':g.tolist(),'candidate_ids':p.tolist()} for g,p in frozen]
            a={'saved_updates':0,'optimizer_calls':0,'initial_head_sha256':ctx.initial_head,'initial_adam_exact':True,
               'initial_predictions':initial,'initial_assignments':fixed,'history':[{'state_step':0,'images':signals}],
               'finite_model_adam':True,'matching_calls':4}
            report['arms'][arm]=a;report['active_arm']=arm;write_json(out/'status.json',report)
            for step in range(1,151):
                tick=time.perf_counter();entry=ctx.update(step,lr_scale=plan['lr_scales'][arm])
                a['lr_scale']=plan['lr_scales'][arm];a['optimizer_calls']=step;a['matching_calls']+=8;report['optimizer_calls']+=1
                used=[s['candidate_ids'] for s in entry['signals_before_update']]
                entry.update(arm=arm,arm_step=step,step=report['optimizer_calls'],used_candidate_ids=used,seconds=time.perf_counter()-tick)
                log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
                # Before-update identity histories require no extra inference and are diagnostic only.
                if step>1:a['history'].append({'state_step':step-1,'images':entry['signals_before_update']})
                if step%25==0:
                    payload={'counter':{k:v.cpu() for k,v in ctx.model.state_dict().items() if not k.startswith('encoder.')},
                      'optimizer':ctx.optimizer.state_dict(),'step':150+step,'new_updates':step,'seed':1729,'arm':arm,
                      'plan_sha256':report['plan_sha256'],'research_initialization_allowed':False,
                      'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
                    tmp=dest/'diagnostic.pt.tmp';torch.save(payload,tmp);tmp.replace(dest/'diagnostic.pt')
                    a['saved_updates']=step;a['checkpoint_sha256']=digest(dest/'diagnostic.pt')
                    report['saved_updates']=sum(r['saved_updates'] for r in report['arms'].values());write_json(out/'status.json',report)
                print(json.dumps({'arm':arm,'step':step,'total_calls':report['optimizer_calls'],'seconds':entry['seconds']}),flush=True)
            a['final_predictions']=ctx.save_predictions(dest,'final')
            with torch.no_grad():outputs=ctx.forward();signals,_=ctx.signals(outputs)
            a['history'].append({'state_step':150,'images':signals});a['matching_calls']+=4
            assert state_digest(ctx.model.encoder.state_dict())==ctx.initial_encoder
            a['head_final_sha256']=state_digest({k:v for k,v in ctx.model.state_dict().items() if not k.startswith('encoder.')})
            a['completed']=True;write_json(dest/'report.json',a)
        report.update(state='completed',elapsed_seconds=time.perf_counter()-start,query_presentations=2400,encoder_final_sha256=state_digest(ctx.model.encoder.state_dict()),peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(out/'status.json',report);write_json(out/'report.json',report)
        write_json(out/'result_hashes.json',{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and p.name not in ['status.json','result_hashes.json']})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--data',required=True);ap.add_argument('--assets',required=True);args=ap.parse_args()
    try:run(args)
    except Exception:write_json(Path(args.output)/'failure.json',{'traceback':traceback.format_exc()});raise
