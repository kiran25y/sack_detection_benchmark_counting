"""Fixed1000-update matched arm. No resume, early stopping or test access."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .adaptation_core import initialize_from_foundation,adaptation_pair_loss,require_screen_admission
from .adaptation_data import Data,sha,warp_ids
from .adaptation_preflight import evidence,finite,FOUNDATION
from .stable_adaptation_runtime import install
from .training import optimizer_update,set_learning_rate
from .evaluation import metrics,choose_threshold
from .transforms import resize_pad
from .foundation import write_json

def evaluate(model,data,out,step):
    """The11 frozen natural dev queries, fixed1024 resize, no augmentation/TTA."""
    records=[];model.eval();device=next(model.parameters()).device
    ref,boxes=data.references('filled_sack')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        proto=model.encode_references(torch.as_tensor(ref,device=device),torch.as_tensor(boxes,device=device))
        for row in data.dev:
            path=data.package/row['image']
            if sha(path)!=row['encoded_sha256']:raise ValueError('Development image changed')
            image=data.image(path);rgb,valid,g=resize_pad(image,1024)
            roi=np.zeros((image.height,image.width),np.int32);x0,y0,x1,y1=row['roi_xyxy'];roi[y0:y1,x0:x1]=1
            valid=valid&(warp_ids(roi,g.record(),0)==1)
            v=model(torch.as_tensor(rgb,device=device)[None],torch.as_tensor(valid,device=device),proto)
            points=g.points(v['points'][v['valid']].float().cpu().numpy(),inverse=True)
            confidence=v['probabilities'][v['valid']].float().cpu().numpy()
            if not np.isfinite(points).all() or not np.isfinite(confidence).all():raise ValueError('Nonfinite development output')
            records.append(dict(query_id=row['image_id'],dependency_group=row['dependency_group'],count=row['count'],truth=row['points'],
                points=points.tolist(),probabilities=confidence.tolist(),width=row['width'],height=row['height']))
    chosen=choose_threshold(records,'development');scores={}
    for name,t in [('common_0.50',.5),('development_selected',chosen['threshold'])]:
        rows=[]
        for r in records:
            pts=np.asarray(r['points']).reshape(-1,2);p=pts[np.asarray(r['probabilities'])>=t]
            rows.append(dict(query_id=r['query_id'],**metrics(p,r['truth'],r['width'],r['height'])))
        scores[name]=dict(threshold=t,MAE=float(np.mean([r['AE'] for r in rows])),RMSE=float(np.sqrt(np.mean([r['error']**2 for r in rows]))),
            bias=float(np.mean([r['error'] for r in rows])),GAME1=float(np.mean([r['GAME1'] for r in rows])),GAME2=float(np.mean([r['GAME2'] for r in rows])),
            F1_8=float(np.mean([r['point_8']['F1'] for r in rows])),rows=rows)
    with (out/f'development_predictions_{step:04}.jsonl').open('x') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    write_json(out/f'development_{step:04}.json',dict(step=step,threshold_selection=chosen,scores=scores,
        dependency_groups=1,queries=11,independent_test=False,physical_pair_gate='unavailable',dense_extreme_claim=False))
    model.train();return scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--package',required=True);ap.add_argument('--output',required=True);ap.add_argument('--arm',choices=list('ABCDEF'),required=True);a=ap.parse_args()
    root=Path(a.root);package=Path(a.package);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);runtime=install();spec,audit,draws,identities=evidence(root,package)
    admission=json.loads((package/'screen_admission.json').read_text());guard_path=root/admission['guard_remote']
    if sha(guard_path/'result/report.json')!=admission['guard_report_sha256'] or sha(guard_path/'exit.json')!=admission['guard_exit_sha256']:raise ValueError('Guard evidence changed')
    guard=json.loads((guard_path/'result/report.json').read_text());exit_state=json.loads((guard_path/'exit.json').read_text())
    if exit_state['returncode']!=0 or exit_state['docker_state']['OOMKilled'] or exit_state['guard_stop']:raise ValueError('Guard execution failed')
    require_screen_admission(admission['readiness'],a.arm,guard,identities)
    model,opt=initialize_from_foundation(root/FOUNDATION,a.arm,'cuda');data=Data(root,package)
    plan=dict(spec,arm=a.arm,identities=identities,runtime=runtime,updates=1000,seed=1729,optimizer_reset=True,
        endpoint='fixed_final1000',automatic_recovery=False,confirmatory_auto_launch=False)
    write_json(out/'plan.json',plan);ph=sha(out/'plan.json');state=dict(state='training',arm=a.arm,durable_optimizer_steps=0,logged_updates=0,total_updates=1000,plan_sha256=ph)
    write_json(out/'status.json',state);curves=[]
    expected={r['draw']['draw_id']:r['array_hashes'] for r in map(json.loads,(audit/'prepared_ledger.jsonl').read_text().splitlines())}
    try:
        evaluate(model,data,out,0)
        for step in range(1000):
            start=time.monotonic();micro=[];records=[]
            for d in draws[step*8:(step+1)*8]:
                batch,record=data.prepare(d,'cpu')
                hashes={k:hashlib.sha256(batch[k].numpy().tobytes()).hexdigest() for k in ['a','b','valid_a','valid_b','points_a','points_b']}
                if d['draw_id'] in expected and hashes!=expected[d['draw_id']]:raise ValueError('Prepared image/point replay differs from frozen audit')
                def move(value):
                    if torch.is_tensor(value):return value.cuda()
                    if isinstance(value,list):return [move(x) for x in value]
                    return value
                micro.append({k:move(v) for k,v in batch.items()});record['array_hashes']=hashes;records.append(record)
            set_learning_rate(opt,step,1000,100)
            m=optimizer_update(model,opt,micro,a.arm,loss_function=adaptation_pair_loss);torch.cuda.synchronize()
            curve=dict(update=step+1,seconds=time.monotonic()-start,learning_rates=[g['lr'] for g in opt.param_groups],**m);curves.append(curve)
            with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(curve)+'\n')
            with (out/'augmentation_ledger.jsonl').open('a') as f:
                for record in records:f.write(json.dumps(record)+'\n')
            del micro,records,batch;state['logged_updates']=step+1
            if (step+1)%25==0:
                finite(model,opt);temp=out/'resume.tmp';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),
                    optimizer_steps=step+1,seed=1729,arm=a.arm,draw_cursor=(step+1)*8,cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),plan_sha256=ph,curves=curves,identities=identities),temp);temp.replace(out/'resume.pt')
                state.update(durable_optimizer_steps=step+1,last_saved_host_unix=time.time(),recent_seconds_per_update=float(np.median([r['seconds'] for r in curves[-25:]])))
                write_json(out/'status.json',state)
            if (step+1)%250==0:evaluate(model,data,out,step+1)
        finite(model,opt);torch.save(dict(model=model.state_dict(),optimizer_steps=1000,seed=1729,arm=a.arm,plan_sha256=ph,identities=identities),out/'counter.pt')
        state.update(state='completed',finite_model_optimizer=True,checkpoint_sha256=sha(out/'counter.pt'),resume_sha256=sha(out/'resume.pt'))
        write_json(out/'status.json',state);write_json(out/'result_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='result_hashes.json'})
    except Exception as exc:state.update(state='failed',error=repr(exc));write_json(out/'status.json',state);raise
if __name__=='__main__':main()
