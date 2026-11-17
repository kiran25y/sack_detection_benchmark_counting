"""Sack-only factorial training with separate carried-box and inventory supervision."""
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from counting.firc.model import load_official_encoder
from counting.firc.preflight import configure
from counting.firc.tiny_fit import digest, state_digest, write_json

JOB=Path('/job')
OUT=Path('/output')
SIDE=336
GRID=24
MEAN=np.array([.485,.456,.406],dtype=np.float32)
STD=np.array([.229,.224,.225],dtype=np.float32)


class SackHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk=nn.Sequential(nn.Conv2d(768,64,1),nn.GroupNorm(8,64),nn.SiLU(),
                                 nn.Conv2d(64,48,3,padding=1),nn.GroupNorm(6,48),nn.SiLU())
        self.carried=nn.Conv2d(48,5,1)
        self.inventory=nn.Conv2d(48,4,1)
        with torch.no_grad():
            self.inventory.bias[3]=-4.6

    def forward(self,x):
        h=self.trunk(x.float())
        return self.carried(h),self.inventory(h)


def rgb(row):
    raw=Path(row['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==row['sha256'],row['id']
    with Image.open(io.BytesIO(raw)) as im:
        assert im.size==tuple(row['size']) and im.getexif().get(274,1)==1
        image=im.convert('RGB').crop(row['roi']).resize((SIDE,SIDE),Image.Resampling.BILINEAR)
        pixels=np.asarray(image,dtype=np.float32)/255.
    return ((pixels-MEAN)/STD).transpose(2,0,1).copy()


@torch.no_grad()
def encode(encoder,rows):
    x=torch.from_numpy(np.stack([rgb(r) for r in rows])).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        z=encoder.get_intermediate_layers(x,n=[11],reshape=True,return_class_token=False,norm=True)[0].half()
    assert z.shape==(len(rows),768,GRID,GRID) and torch.isfinite(z).all()
    return z


def targets(rows,kind):
    """Separate calls for carried boxes and exhaustive target centers."""
    target=np.zeros((len(rows),GRID,GRID),bool)
    geometry=np.zeros((len(rows),GRID,GRID,4 if kind=='carried' else 2),np.float32)
    region=np.zeros((len(rows),4,4),np.float32)
    for i,r in enumerate(rows):
        if kind=='carried':
            box=np.asarray(r['boxes'],dtype=float).reshape(-1,4)
            xy=box[:,:2]
        else:
            roi=np.asarray(r['roi'])
            xy=(np.asarray(r['points'],dtype=float).reshape(-1,2)+.5-roi[:2])/(roi[2:]-roi[:2])
        assert ((0<=xy)&(xy<1)).all()
        grid=xy*GRID
        x,y=grid.astype(int).T
        target[i,y,x]=True
        # The audited sources contain at most two carried boxes. Preserve both for
        # the ledger even if two targets fall into one cell; fail closed on ambiguity.
        assert int(target[i].sum())==len(xy),(r['id'],'colliding centers')
        geometry[i,y,x,:2]=grid-np.floor(grid)
        if kind=='carried':geometry[i,y,x,2:]=np.log(np.maximum(box[:,2:],1e-6))
        else:
            qx,qy=(xy*4).astype(int).T
            np.add.at(region[i],(qy,qx),1)
    return tuple(torch.as_tensor(a,device='cuda') for a in (target,geometry,region))


def point_loss(output,labels,box_geometry=False):
    mask,geometry,_=labels
    logits=output[:,0]
    positive=F.softplus(-logits[mask]).mean() if mask.any() else logits.sum()*0
    # Each frame contributes hard negatives, including zero-carried-box frames.
    negatives=logits.masked_fill(mask,-1e5).flatten(1).topk(32,dim=1).values
    loss=positive+F.softplus(negatives).mean()
    if mask.any():
        g=output[:,1:].permute(0,2,3,1)[mask]
        truth=geometry[mask]
        loss=loss+2*F.smooth_l1_loss(g[:,:2].sigmoid(),truth[:,:2],beta=.1)
        if box_geometry:
            loss=loss+.5*F.smooth_l1_loss(g[:,2:4],truth[:,2:4],beta=.25)
    return loss


def count_loss(output,counts):
    pred=F.softplus(output[:,3]).sum(dim=(1,2))
    return F.smooth_l1_loss(torch.log1p(pred),torch.log1p(counts),beta=.25)


def benchmark_loss(output,labels):
    heat=point_loss(output,labels)
    truth=labels[2]
    region=F.softplus(output[:,3]).reshape(-1,4,6,4,6).sum(dim=(2,4))
    return (heat+count_loss(output,truth.sum(dim=(1,2)))
            +F.smooth_l1_loss(torch.log1p(region),torch.log1p(truth),beta=.25))


def cache(encoder,rows):
    bank={}
    for start in range(0,len(rows),8):
        chunk=rows[start:start+8]
        feat=encode(encoder,chunk)
        for r,z in zip(chunk,feat):
            bank[r['id']]=z.cpu().contiguous()
    return bank


@torch.no_grad()
def candidates(head,z,row):
    _,y=head(z[None].cuda())
    mass=float(F.softplus(y[0,3]).sum())
    score=y[0,0].flatten().sigmoid().cpu().numpy()
    offsets=y[0,1:3].sigmoid().permute(1,2,0).reshape(-1,2).cpu().numpy()
    iy,ix=np.indices((GRID,GRID))
    uv=(np.column_stack([ix.flatten(),iy.flatten()])+offsets)/GRID
    roi=np.array(row['roi']);points=uv*(roi[2:]-roi[:2])+roi[:2]-.5
    assert np.isfinite(points).all() and np.isfinite(score).all() and np.isfinite(mass)
    return dict(mass=mass,points=points.tolist(),scores=score.tolist())


def tiled_schedule(rng,n,length):
    return np.concatenate([rng.permutation(n) for _ in range(math.ceil(length/n))])[:length]


def main():
    configure();plan=json.loads((JOB/'plan.json').read_text());step_budget=plan['steps']
    boxrows=[json.loads(l) for l in (JOB/'carried.jsonl').read_text().splitlines()]
    counts=json.loads((JOB/'counts.json').read_text());bench=plan['benchmark']
    fit=[r for r in bench if r['split']=='train']
    evaluation=[r for r in bench if r['split']!='train']
    assert all('points' not in r and 'count' not in r for r in evaluation)
    assert len(boxrows)==21006 and len(counts)==300 and len(fit)==42
    assert sum(len(r['boxes']) for r in boxrows)==14403 and all('count' not in r for r in boxrows)
    assert all('boxes' not in r and r['count']>=0 for r in counts)
    ck=Path('/assets/dinov2_vitb14_reg4_pretrain.pth')
    assert digest(ck)==plan['encoder_file']
    encoder=load_official_encoder(ck).requires_grad_(False).cuda().eval()
    assert state_digest(encoder.state_dict())==plan['encoder_state']
    count_cache=cache(encoder,counts);target_cache=cache(encoder,fit)
    heads={};opts={};initial={}
    for arm in plan['arms']:
        torch.manual_seed(plan['seed'])
        heads[arm]=SackHead().cuda()
        opts[arm]=torch.optim.AdamW(heads[arm].parameters(),lr=.0004,weight_decay=.01)
        initial[arm]=state_digest(heads[arm].state_dict())
    assert len(set(initial.values()))==1
    rng=np.random.default_rng(plan['seed'])
    order=rng.permutation(len(boxrows))
    ct=tiled_schedule(rng,len(counts),step_budget*2)
    tg=tiled_schedule(rng,len(fit),step_budget*2)
    count_seen=set();box_seen=set();fit_seen=set();zero_carried=0
    with (OUT/'updates.jsonl').open('x') as log:
        for i in range(step_budget):
            batch=[boxrows[j] for j in order[32*i:32*(i+1)]]
            exact=[counts[j] for j in ct[2*i:2*i+2]]
            local=[fit[j] for j in tg[2*i:2*i+2]]
            # Only the 300 reviewed scalar targets feed the external inventory loss.
            y_count=torch.tensor([r['count'] for r in exact],device='cuda',dtype=torch.float32)
            y_box=targets(batch,'carried');y_local=targets(local,'inventory')
            feat=encode(encoder,batch)
            z=torch.cat([feat,torch.stack([count_cache[r['id']] for r in exact]).cuda(),
                         torch.stack([target_cache[r['id']] for r in local]).cuda()])
            ledger={}
            for arm,head in heads.items():
                opt=opts[arm];opt.zero_grad(set_to_none=True)
                carried,inventory=head(z);n=len(batch)
                lb=point_loss(carried[:n],y_box,True)
                lc=count_loss(inventory[n:n+2],y_count)
                lt=benchmark_loss(inventory[n+2:],y_local)
                loss=lt+(0.5 if arm in ('boxes','both') else 0.)*lb+(0.5 if arm in ('totals','both') else 0.)*lc
                loss.backward()
                norm=nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
                ledger[arm]=[float(v.detach()) for v in (lt,lb,lc,norm)]
            count_seen.update(r['id'] for r in exact);box_seen.update(r['id'] for r in batch)
            fit_seen.update(r['id'] for r in local);zero_carried+=sum(not r['boxes'] for r in batch)
            log.write(json.dumps(dict(step=i+1,box_indices=order[32*i:32*(i+1)].tolist(),count_indices=ct[2*i:2*i+2].tolist(),
                                      target_indices=tg[2*i:2*i+2].tolist(),losses=ledger),allow_nan=False)+'\n')
            if i%20==0 or i+1==step_budget:
                log.flush();write_json(OUT/'status.json',dict(stage='training',step=i+1,total=step_budget,boxes_seen=len(box_seen)))
    assert len(box_seen)==21006 and len(count_seen)==300 and len(fit_seen)==42 and zero_carried==6723
    freezes={}
    for arm,h in heads.items():
        assert all(torch.isfinite(v).all() for v in h.parameters())
        assert all(torch.isfinite(v).all() for s in opts[arm].state.values() for v in s.values() if torch.is_tensor(v))
        h.eval();p=OUT/(arm+'.pt');torch.save(h.state_dict(),p)
        zero_masses={r['id']:candidates(h,target_cache[r['id']],r)['mass'] for r in fit if not r['points']}
        assert len(zero_masses)==3
        freezes[arm]=dict(checkpoint=digest(p),zero_masses=zero_masses,empty_floor=max(.5,max(zero_masses.values())),
                          steps=step_budget,initial=initial[arm])
    write_json(OUT/'freeze.json',freezes)
    target_cache.update(cache(encoder,evaluation))
    predictions={arm:{r['id']:candidates(h,target_cache[r['id']],r) for r in bench} for arm,h in heads.items()}
    write_json(OUT/'candidates.json',predictions)
    assert state_digest(encoder.state_dict())==plan['encoder_state']
    sums={p.name:digest(p) for p in OUT.iterdir() if p.name not in ('status.json','report.json')}
    write_json(OUT/'report.json',dict(state='completed',box_train_images=len(box_seen),carried_boxes=14403,reviewed_total_frames=len(count_seen),
               count_presentations=len(ct),point_train_frames=len(fit_seen),point_presentations=len(tg),zero_carried_frames=zero_carried,
               optimizer_calls=step_budget*4,parameters=sum(v.numel() for v in heads['both'].parameters()),
               encoder=plan['encoder_state'],plan_sha256=digest(JOB/'plan.json'),hashes=sums))


if __name__=='__main__':main()
