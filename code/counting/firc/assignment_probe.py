"""Bounded assignment diagnosis from an authentic diagnostic checkpoint, never a paper fit."""
import argparse, copy, json, math, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .geometry import transform, image_to_model, points_to_model
from .model import FIRC, load_official_encoder, optimizer_for
from .preflight import configure, state_diff
from .tiny_fit import digest, state_digest, write_json
from .torch_ops import match, view_loss


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def assigned_loss(row, truth, assignment):
    gi, pi = assignment
    logits = row['logits'].float(); q = logits.sigmoid()
    target = torch.zeros_like(logits); target[pi] = 1
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = torch.where(target.bool(), q, 1-q); alpha = torch.where(target.bool(), .25, .75)
    focal = (alpha*(1-pt).square()*ce).sum()/max(1,len(truth))
    point = F.huber_loss((row['points'][pi]-truth[gi]).float()/16,
                        torch.zeros((len(gi),2),device=q.device),delta=1,reduction='sum')/max(1,len(gi))
    return {'total': focal+5*point, 'focal': focal, 'point': point, 'matched': len(gi)}


class Context:
    def __init__(self, data, assets):
        self.data, self.assets = Path(data), Path(assets)
        self.plan = json.loads((self.data/'plan.json').read_text())
        for name, sha in json.loads((self.data/'hashes.json').read_text()).items():
            assert digest(self.data/name)==sha, name
        configure(); assert torch.cuda.is_bf16_supported()
        self.rows, self.refs = self.plan['rows'], self.plan['references']
        assert [r['count'] for r in self.rows]==[0,1,21,34]
        images, truth, transforms = [], [], []
        for r in self.rows:
            assert r['split']=='train' and r['exhaustive_reviewed'] and len(r['points'])==r['count']
            with Image.open(self.data/r['image']) as im:
                assert im.size==(r['width'],r['height'])
                t=transform(*im.size,896,r['roi_xyxy']);t['padded_hw']=[896,896]
                images.append(torch.from_numpy(image_to_model(im,t)))
            truth.append(torch.tensor(points_to_model(r['points'],t),dtype=torch.float32,device='cuda'))
            transforms.append(t)
        self.images=torch.stack(images).cuda();self.truth=truth;self.transforms=transforms
        refs,boxes=[],[]
        for i,r in enumerate(self.refs):
            assert r['split']=='train'
            with Image.open(self.data/r['image']) as im:
                t=transform(*im.size,224);t['padded_hw']=[224,224]
                refs.append(torch.from_numpy(image_to_model(im,t)))
                h,w=t['valid_hw'];boxes.append([i,0,0,w,h])
        self.references=torch.stack(refs).cuda();self.boxes=torch.tensor(boxes,dtype=torch.float32,device='cuda')
        checkpoint=self.assets/'dinov2_vitb14_reg4_pretrain.pth'
        assert digest(checkpoint)==self.plan['pretrained_sha256']
        self.model=FIRC(load_official_encoder(checkpoint)).cuda().train()
        self.optimizer=optimizer_for(self.model)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            self.taps=tuple(t.detach() for t in self.model.encode(self.images))
            self.rt=tuple(t.detach() for t in self.model.encode(self.references))
        self.initial_payload=torch.load(self.data/'initial.pt',map_location='cpu',weights_only=True)
        assert self.initial_payload['step']==150 and self.initial_payload['research_initialization_allowed'] is False
        assert self.initial_payload['seed']==1729 and self.initial_payload['plan_sha256']==self.plan['initial_plan_sha256']
        assert digest(self.data/'initial.pt')==self.plan['initial_checkpoint_sha256']
        self.restore()
        self.initial_encoder=state_digest(self.model.encoder.state_dict())
        self.initial_head=state_digest(self.initial_payload['counter'])

    def restore(self):
        result=self.model.load_state_dict(self.initial_payload['counter'],strict=False)
        assert not result.unexpected_keys and all(k.startswith('encoder.') for k in result.missing_keys)
        self.optimizer.load_state_dict(copy.deepcopy(self.initial_payload['optimizer']))
        assert state_diff(self.initial_payload['optimizer'],cpu_tree(self.optimizer.state_dict()))==0
        torch.set_rng_state(self.initial_payload['torch_rng'])
        torch.cuda.set_rng_state_all(self.initial_payload['cuda_rng'])
        self.optimizer.zero_grad(set_to_none=True)

    def forward(self):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return self.model(self.images,self.references,self.boxes,[t['valid_hw'] for t in self.transforms],taps=self.taps,reference_taps=self.rt)

    def signals(self, outputs):
        details=[];assignments=[]
        for r,v,g in zip(self.rows,outputs,self.truth):
            gi,pi=match(v,g);assert np.array_equal(gi,np.arange(len(g)))
            assignments.append((gi,pi))
            q=v['probabilities'].detach().cpu().numpy();anchors=v['anchors'].detach().cpu().numpy()
            points=v['points'].detach().cpu().numpy();gt=g.cpu().numpy();margins=[]
            for j,k in zip(gi,pi):
                eligible=np.flatnonzero((abs(anchors-gt[j])<=16).all(1))
                competitors=eligible[eligible!=k]
                cost=abs(points[eligible]-gt[j]).sum(1)/16-.2*np.log(q[eligible]+1e-6)+1e-8
                ordered=np.sort(cost)
                margins.append({'assigned_q':float(q[k]),'best_competing_q':float(q[competitors].max()),
                                'local_cost_gap':float(ordered[1]-ordered[0])})
            details.append({'image_id':r['image_id'],'truth':len(g),'candidate_ids':pi.tolist(),
                            'margins':margins,'hard_count':int((q>=.5).sum()),'max_q':float(q.max())})
        return details,assignments

    def save_predictions(self, output, stage):
        with torch.no_grad(): predictions=self.forward()
        entries=[]
        for r,v,t in zip(self.rows,predictions,self.transforms):
            name=stage+'_'+r['image_id']+'.npz'
            np.savez_compressed(output/name,points=v['points'].cpu().numpy(),probabilities=v['probabilities'].cpu().numpy(),truth_native=np.asarray(r['points'],dtype=float).reshape(-1,2))
            entries.append({'file':name,'image_id':r['image_id'],'sha256':digest(output/name),'transform':t})
        return entries

    def update(self, step, frozen=None, lr_scale=1.):
        self.optimizer.zero_grad(set_to_none=True)
        frac=step/15 if step<=15 else .01+.99*(1+math.cos(math.pi*(step-15)/135))/2
        lr=2e-4*frac*lr_scale
        for group in self.optimizer.param_groups:group['lr']=lr
        bundle_logs=[];identities=None
        for _ in range(2):
            outputs=self.forward();signals,assignments=self.signals(outputs)
            if identities is None: identities=[x['candidate_ids'] for x in signals]
            else: assert identities==[x['candidate_ids'] for x in signals]
            losses=[assigned_loss(v,g,a) for v,g,a in zip(outputs,self.truth,assignments if frozen is None else frozen)]
            loss=torch.stack([v['total'] for v in losses]).mean()/2
            assert torch.isfinite(loss);loss.backward()
            bundle_logs.append([{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in entry.items()} for entry in losses])
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters() if p.requires_grad)
        assert all(p.grad is None for p in self.model.encoder.parameters())
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.);assert torch.isfinite(norm)
        self.optimizer.step()
        assert all(torch.isfinite(v).all() for s in self.optimizer.state.values() for v in s.values() if torch.is_tensor(v))
        assert all(torch.isfinite(p).all() for p in self.model.parameters() if p.requires_grad)
        return {'step':step,'lr':lr,'gradient_norm_before_clip':float(norm),'bundles':bundle_logs,'signals_before_update':signals,'query_presentations':8}


def run(args):
    out=Path(args.output);ctx=Context(args.data,args.assets);start=time.perf_counter()
    report={'state':'running','saved_updates':0,'optimizer_calls':0,'persistent_steps':0,
            'purpose':'Assignment diagnosis only; not research initialization or accuracy claim',
            'plan_sha256':digest(Path(args.data)/'plan.json'),'initial_head_sha256':ctx.initial_head,
            'initial_encoder_sha256':ctx.initial_encoder,'source_checkpoint_sha256':ctx.plan['initial_checkpoint_sha256']}
    write_json(out/'status.json',report)
    entries=ctx.save_predictions(out,'initial')
    old=json.loads((Path(args.data)/'control/report.json').read_text())
    for e,original in zip(entries,old['final_predictions']):
        with np.load(out/e['file'],allow_pickle=False) as a,np.load(Path(args.data)/'control'/original['file'],allow_pickle=False) as b:
            assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
    # Exact new explicit-assignment loss/gradient parity with original no-count objective.
    outputs=ctx.forward();signals,assignments=ctx.signals(outputs)
    a=[assigned_loss(v,g,m) for v,g,m in zip(outputs,ctx.truth,assignments)]
    b=[view_loss(v,g) for v,g in zip(outputs,ctx.truth)]
    va=torch.stack([v['total'] for v in a]).mean();vb=torch.stack([v['focal']+5*v['point'] for v in b]).mean()
    assert torch.equal(va,vb)
    params=[p for p in ctx.model.parameters() if p.requires_grad]
    ga=torch.autograd.grad(va,params,retain_graph=True);gb=torch.autograd.grad(vb,params)
    assert all(torch.equal(x,y) for x,y in zip(ga,gb))
    del outputs,a,b,va,vb,ga,gb
    report['exact_loss_gradient_parity']=True;report['exact_checkpoint_forward_parity']=True
    before=signals;expected=ctx.update(1);report['optimizer_calls']+=1
    expected_head={k:v.detach().cpu().clone() for k,v in ctx.model.state_dict().items() if not k.startswith('encoder.')}
    expected_opt=copy.deepcopy(ctx.optimizer.state_dict())
    ctx.restore();history=[{'state_step':0,'images':before}]
    with (out/'updates.jsonl').open('x') as log:
        expected.update(optimizer_call=1,role='recovery_reference')
        log.write(json.dumps(expected,allow_nan=False)+'\n');log.flush()
        for step in range(1,13):
            tick=time.perf_counter();entry=ctx.update(step);report['optimizer_calls']+=1
            if step==1:
                assert state_diff(expected_head,{k:v.detach().cpu() for k,v in ctx.model.state_dict().items() if not k.startswith('encoder.')})==0
                assert state_diff(expected_opt,ctx.optimizer.state_dict())==0
                report['exact_nonempty_adam_recovery']=True
            with torch.no_grad():outputs=ctx.forward();signals,_=ctx.signals(outputs)
            history.append({'state_step':step,'images':signals})
            entry.update(optimizer_call=step+1,role='diagnostic_trajectory',seconds=time.perf_counter()-tick)
            log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
            report['diagnostic_updates']=step;write_json(out/'status.json',report)
            print(json.dumps({'diagnostic_updates':step,'optimizer_calls':report['optimizer_calls']}),flush=True)
    report['final_predictions']=ctx.save_predictions(out,'final');report['initial_predictions']=entries
    write_json(out/'assignment_history.json',history)
    report['encoder_final_sha256']=state_digest(ctx.model.encoder.state_dict());assert report['encoder_final_sha256']==ctx.initial_encoder
    report.update(state='completed',elapsed_seconds=time.perf_counter()-start,finite_state=True,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(out/'status.json',report);write_json(out/'report.json',report)
    write_json(out/'result_hashes.json',{p.name:digest(p) for p in out.iterdir() if p.name.endswith('.npz') or p.name in ['report.json','updates.jsonl','assignment_history.json']})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--data',required=True);ap.add_argument('--assets',required=True);args=ap.parse_args()
    try:run(args)
    except Exception:write_json(Path(args.output)/'failure.json',{'traceback':traceback.format_exc()});raise
