"""Bounded input audit, disposable CUDA guard and fixed200-update transfer cells.

No auto-resume, no threshold selection, no test access and no full-run promotion.
"""
import argparse,copy,gc,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from .transfer_protocol import CELLS,UPDATES,WARMUP,ORIGINAL_GROUP
from .transfer_data import TransferData,evidence
from .adaptation_data import sha,warp_ids
from .adaptation_core import initialize_from_foundation,adaptation_pair_loss,require_runtime
from .adaptation_preflight import finite,FOUNDATION
from .assignment_cudnn_preflight import state_digest
from .stable_adaptation_runtime import install
from .training import optimizer_update,set_learning_rate
from .transforms import resize_pad,MEAN,STD
from .evaluation import metrics
from .foundation import write_json

ARRAY_KEYS=('a','b','valid_a','valid_b','points_a','points_b')
def array_hashes(batch):
    return {k:hashlib.sha256(batch[k].detach().cpu().numpy().tobytes()).hexdigest() for k in ARRAY_KEYS}

def move(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,list):return [move(x,device) for x in value]
    return value

def checked_receipt(root,package,filename,identities,stage):
    admission=json.loads((package/filename).read_text());remote=root/admission['remote']
    for file,key in [('result/report.json','report_sha256'),('exit.json','exit_sha256')]:
        if sha(remote/file)!=admission[key]:raise ValueError('Changed stage evidence')
    report=json.loads((remote/'result/report.json').read_text());ex=json.loads((remote/'exit.json').read_text())
    if ex['returncode']!=0 or ex['docker_state']['OOMKilled'] or ex['guard_stop']:raise ValueError('Prior stage did not exit successfully')
    if report.get('passed') is not True or report['state']!='completed' or report['stage']!=stage or report['identities']!=identities:raise ValueError('Wrong/incomplete prior stage')
    return report,remote

def input_audit(root,package,out,spec,draws,identities):
    report=dict(state='running',stage='transfer_input_audit',passed=False,identities=identities,optimizer_calls=0,model_forward_calls=0,witnesses=[],public_replay_checks=0)
    write_json(out/'report.json',report);parent=root/spec['parent_pair_audit_remote']
    if sha(parent/'draws.jsonl')!=spec['parent_draws_sha256']:raise ValueError('Parent audit changed')
    old={r['draw']['draw_id']:r['array_hashes'] for r in map(json.loads,(parent/'prepared_ledger.jsonl').read_text().splitlines())}
    for cell in CELLS:
        data=TransferData(root,package,cell);seen=set();panel=[]
        for d in draws[cell]:
            if d['dataset']=='sacks' and d['query_id'] not in seen:seen.add(d['query_id']);panel.append(d)
        expected={r['image_id'] for r in data.train if CELLS[cell]['coverage']=='broadened' or r['dependency_group']==ORIGINAL_GROUP}
        if seen!=expected:raise ValueError('A training frame has no planned exposure')
        public=[d for d in draws[cell] if d['dataset']!='sacks'];panel+=public[:8]
        panel+=[d for d in public if d['source_audit_draw_id'] in old and d not in panel]
        for d in panel:
            batch,record=data.prepare(d,'cpu');hashes=array_hashes(batch)
            if d['dataset']!='sacks' and d['source_audit_draw_id'] in old:
                if hashes!=old[d['source_audit_draw_id']]:raise ValueError('Public input differs from original audit')
                report['public_replay_checks']+=1
            witness=dict(cell=cell,draw_id=d['draw_id'],query_id=d['query_id'],array_hashes=hashes,record=record)
            if d['dataset']=='sacks':
                # Native labels travel through the exact runtime transform. No new labels.
                canvas=Image.new('RGB',(1024,600),'white');pen=ImageDraw.Draw(canvas)
                for j,side in enumerate(('a','b')):
                    arr=batch[side][0].numpy().transpose(1,2,0);im=Image.fromarray(np.uint8(np.clip(arr*STD+MEAN,0,1)*255+.5));sx=512/im.width;sy=550/im.height;s=min(sx,sy);im=im.resize((round(im.width*s),round(im.height*s)));canvas.paste(im,(j*512,30))
                    for x,y in batch['points_'+side].numpy():
                        x=j*512+x*s;y=30+y*s;pen.ellipse((x-3,y-3,x+3,y+3),fill='cyan')
                    pen.text((j*512+5,5),f'{cell} {d["query_id"]} {side} N={len(batch["points_"+side])}',fill='black')
                name=f'{cell}_{d["draw_id"]:04}_{d["query_id"]}.jpg';canvas.save(out/name,quality=94);witness['image']=name;witness['image_sha256']=sha(out/name)
            report['witnesses'].append(witness)
            with (out/'prepared_ledger.jsonl').open('a') as f:f.write(json.dumps(witness)+'\n')
            del batch
    if not report['public_replay_checks']:raise ValueError('No actual public replay check')
    report.update(state='completed',passed=True,manual_visual_review_required=True,checked_pairs=len(report['witnesses']))
    write_json(out/'report.json',report)

def require_audit(root,package,identities):
    report,remote=checked_receipt(root,package,'input_audit_admission.json',identities,'transfer_input_audit')
    manual=json.loads((package/'input_visual_review.json').read_text());expected={w['image']:w['image_sha256'] for w in report['witnesses'] if 'image' in w}
    if manual.get('passed') is not True or manual.get('reviewed_witness_hashes')!=expected or manual.get('audit_report_sha256')!=sha(remote/'result/report.json'):raise ValueError('Complete actual transformed-input visual review required')
    return report

def preflight(root,package,out,spec,draws,identities):
    require_audit(root,package,identities);runtime=install();require_runtime('A',runtime)
    report=dict(state='running',stage='transfer_cuda_preflight',passed=False,identities=identities,optimizer_calls=0,maximum_optimizer_calls=6,persistent_steps=0,cells=list(CELLS),runtime=runtime,updates=[])
    write_json(out/'report.json',report);initial=None;torch.cuda.reset_peak_memory_stats()
    for cell in CELLS:
        data=TransferData(root,package,cell);dd=draws[cell]
        sacks=[d for d in dd if d['dataset']=='sacks'];dense=max(sacks,key=lambda d:data.train[d['row_index']]['count']);small=min(sacks,key=lambda d:data.train[d['row_index']]['count'])
        public=[d for d in dd if d['dataset']!='sacks'];choices=[dense,small]+public[:6]
        micro=[{k:move(v,'cuda') for k,v in data.prepare(d,'cpu')[0].items()} for d in choices]
        model,opt=initialize_from_foundation(root/FOUNDATION,'A','cuda');digest=state_digest(model.state_dict())
        if initial is None:initial=digest
        elif digest!=initial:raise ValueError('Initialization mismatch')
        if opt.state:raise ValueError('Adam not reset')
        def update(step):
            set_learning_rate(opt,step,UPDATES,WARMUP);m=optimizer_update(model,opt,micro,'A',loss_function=adaptation_pair_loss);torch.cuda.synchronize();finite(model,opt)
            report['optimizer_calls']+=1;report['updates'].append(dict(cell=cell,schedule_step=step,**m));write_json(out/'report.json',report)
        update(0)
        if cell=='P11':
            snap=dict(model=model.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all());original={k:state_digest(v) for k,v in snap.items()};torch.save(snap,out/'recovery.pt');del snap
            update(1);expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};moments={str(i)+':'+k:v.detach().cpu().clone() for i,s in opt.state_dict()['state'].items() for k,v in s.items() if torch.is_tensor(v)}
            ck=torch.load(out/'recovery.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);del ck
            actual=dict(model=model.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            if {k:state_digest(v) for k,v in actual.items()}!=original:raise ValueError('Recovery differs before update')
            del actual;update(1);maximum=0.
            for k,v in model.state_dict().items():
                val=v.detach().cpu();torch.testing.assert_close(val,expected[k],rtol=.003,atol=1e-5);maximum=max(maximum,float((val-expected[k]).abs().max()))
            for i,s in opt.state_dict()['state'].items():
                for k,v in s.items():
                    if torch.is_tensor(v):torch.testing.assert_close(v.cpu(),moments[str(i)+':'+k],rtol=.003,atol=1e-5)
            report['recovery']=dict(preupdate_exact=True,poststep_model_max_abs=maximum,rtol=.003,atol=1e-5);del expected,moments
        del model,opt,micro,data;gc.collect();torch.cuda.empty_cache()
    if report['optimizer_calls']!=6:raise ValueError('Guard budget mismatch')
    report.update(state='completed',passed=True,finite_model_optimizer=True,fresh_Adam_all_cells=True,shared_initial_model=initial,actual_two_view_backward=True,peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved())
    write_json(out/'report.json',report)

def aggregate(rows):
    if not rows:return None
    return dict(queries=len(rows),MAE=float(np.mean([r['AE'] for r in rows])),RMSE=float(np.sqrt(np.mean([r['error']**2 for r in rows]))),bias=float(np.mean([r['error'] for r in rows])),GAME1=float(np.mean([r['GAME1'] for r in rows])),GAME2=float(np.mean([r['GAME2'] for r in rows])),F1_8=float(np.mean([r['point_8']['F1'] for r in rows])),FP_total=sum(r['point_8']['FP'] for r in rows),rows=rows)

def evaluate(model,data,out,step):
    model.eval();device=next(model.parameters()).device;ref,boxes=data.references('filled_sack');records=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        proto=model.encode_references(torch.as_tensor(ref,device=device),torch.as_tensor(boxes,device=device))
        for r in data.dev:
            p=data.package/r['image']
            if sha(p)!=r['encoded_sha256']:raise ValueError('Development image changed')
            im=data.image(p);rgb,valid,g=resize_pad(im,1024);roi=np.zeros((im.height,im.width),np.int32);x0,y0,x1,y1=r['roi_xyxy'];roi[y0:y1,x0:x1]=1;valid=valid&(warp_ids(roi,g.record(),0)==1)
            v=model(torch.as_tensor(rgb,device=device)[None],torch.as_tensor(valid,device=device),proto);points=g.points(v['points'][v['valid']].float().cpu().numpy(),inverse=True);prob=v['probabilities'][v['valid']].float().cpu().numpy()
            if not np.isfinite(points).all() or not np.isfinite(prob).all():raise ValueError('Nonfinite predictions')
            records.append(dict(query_id=r['image_id'],dependency_group=r['dependency_group'],truth=r['points'],count=r['count'],width=r['width'],height=r['height'],points=points.tolist(),probabilities=prob.tolist()))
    with (out/f'development_predictions_{step:04}.jsonl').open('x') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    scores={}
    for threshold in (.5,.1):
        rows=[]
        for r in records:
            pts=np.asarray(r['points']).reshape(-1,2)[np.asarray(r['probabilities'])>=threshold];rows.append(dict(query_id=r['query_id'],dependency_group=r['dependency_group'],**metrics(pts,r['truth'],r['width'],r['height'])))
        scores[str(threshold)]={'all':aggregate(rows),'zero':aggregate([r for r in rows if r['truth']==0]),'low_1_11':aggregate([r for r in rows if 1<=r['truth']<=11]),'higher_12_plus':aggregate([r for r in rows if r['truth']>=12]),'groups':{g:aggregate([r for r in rows if r['dependency_group']==g]) for g in sorted({r['dependency_group'] for r in rows})}}
    write_json(out/f'development_{step:04}.json',dict(step=step,scores=scores,threshold_selection=False,independent_test=False,dense_extreme_claim=False));model.train()

def train(root,package,out,spec,draws,identities,cell):
    audit=require_audit(root,package,identities);guard,_=checked_receipt(root,package,'cuda_admission.json',identities,'transfer_cuda_preflight')
    if guard['optimizer_calls']!=6 or guard['persistent_steps']!=0 or guard['cells']!=list(CELLS) or not guard['fresh_Adam_all_cells'] or not guard['actual_two_view_backward'] or not guard['recovery']['preupdate_exact']:raise ValueError('Incomplete actual CUDA guard')
    runtime=install();require_runtime('A',runtime);data=TransferData(root,package,cell);model,opt=initialize_from_foundation(root/FOUNDATION,'A','cuda')
    if opt.state or state_digest(model.state_dict())!=guard['shared_initial_model']:raise ValueError('Wrong initial model/Adam')
    state=dict(state='training',cell=cell,durable_optimizer_steps=0,logged_updates=0,total_updates=UPDATES,identities=identities);write_json(out/'status.json',state);write_json(out/'plan.json',dict(spec,cell=cell,runtime=runtime,identities=identities));ph=sha(out/'plan.json');curves=[]
    expected={(w['cell'],w['draw_id']):w['array_hashes'] for w in audit['witnesses']}
    evaluate(model,data,out,0)
    for step in range(UPDATES):
        start=time.monotonic();micro=[];records=[]
        for d in draws[cell][step*8:(step+1)*8]:
            batch,record=data.prepare(d,'cpu');hashes=array_hashes(batch)
            if (cell,d['draw_id']) in expected and hashes!=expected[(cell,d['draw_id'])]:raise ValueError('Actual input replay changed')
            micro.append({k:move(v,'cuda') for k,v in batch.items()});record['array_hashes']=hashes;records.append(record)
        set_learning_rate(opt,step,UPDATES,WARMUP);m=optimizer_update(model,opt,micro,'A',loss_function=adaptation_pair_loss);torch.cuda.synchronize();curve=dict(update=step+1,seconds=time.monotonic()-start,**m);curves.append(curve)
        with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(curve)+'\n')
        with (out/'augmentation_ledger.jsonl').open('a') as f:
            for r in records:f.write(json.dumps(r)+'\n')
        del micro,records,batch;state['logged_updates']=step+1
        if (step+1)%25==0:
            finite(model,opt);tmp=out/'resume.tmp';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),optimizer_steps=step+1,cell=cell,seed=1729,draw_cursor=(step+1)*8,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),plan_sha256=ph,identities=identities),tmp);tmp.replace(out/'resume.pt');state.update(durable_optimizer_steps=step+1,last_saved_host_unix=time.time(),recent_seconds_per_update=float(np.median([r['seconds'] for r in curves[-25:]])));write_json(out/'status.json',state)
    finite(model,opt);torch.save(dict(model=model.state_dict(),optimizer_steps=UPDATES,cell=cell,seed=1729,plan_sha256=ph,identities=identities),out/'counter.pt');evaluate(model,data,out,UPDATES)
    state.update(state='completed',finite_model_optimizer=True,checkpoint_sha256=sha(out/'counter.pt'),resume_sha256=sha(out/'resume.pt'));write_json(out/'status.json',state)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--package',required=True);ap.add_argument('--output',required=True);ap.add_argument('--mode',choices=['inputs','preflight','train'],required=True);ap.add_argument('--cell',choices=list(CELLS));a=ap.parse_args()
    if a.mode=='train' and not a.cell:raise ValueError('Explicit prescribed cell required')
    root=Path(a.root);package=Path(a.package);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    try:
        spec,draws,identities=evidence(root,package)
        if a.mode=='inputs':input_audit(root,package,out,spec,draws,identities)
        elif a.mode=='preflight':preflight(root,package,out,spec,draws,identities)
        else:train(root,package,out,spec,draws,identities,a.cell)
        write_json(out/'result_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='result_hashes.json'})
    except Exception as exc:
        old=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {};old.update(state='failed',error=repr(exc));write_json(out/'status.json',old);raise
if __name__=='__main__':main()
