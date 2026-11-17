"""One preregistered three-seed same-backbone experiment. The worker has no dev labels."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from architectures import ContextPointDetector, context_features
from proposer_reference import Proposer, optimization_loss, propose, metrics
from counting.firc.geometry import image_to_model, points_to_model, transform
from counting.firc.model import load_official_encoder
from counting.firc.preflight import configure
from counting.firc.tiny_fit import digest, state_digest, write_json

OUT = Path('/output')


def load_features(plan):
    weights=Path('/assets/dinov2_vitb14_reg4_pretrain.pth')
    assert digest(weights)==plan['weight_hash']
    encoder=load_official_encoder(weights).requires_grad_(False).cuda().eval()
    assert state_digest(encoder.state_dict())==plan['encoder_hash']
    items,cache=[],{}
    forwards=0
    for r in plan['rows']:
        path=Path('/data')/r['image']
        assert digest(path)==r['encoded_sha256']
        with Image.open(path) as im:
            assert im.size==(r['width'],r['height'])
            t=transform(*im.size,896,r['roi_xyxy'])
            image=torch.from_numpy(image_to_model(im,t))[None].cuda()
        h,w=[x//14 for x in t['padded_hw']]
        yy,xx=np.ogrid[:h,:w]
        valid=torch.tensor((yy*14<t['valid_hw'][0])&(xx*14<t['valid_hw'][1]),device='cuda')
        features,calls=context_features(encoder,image,valid,cache)
        forwards+=calls
        it=dict(id=r['image_id'],group=r['group'],split=r['split'],features=features,valid=valid,
                radius=8*max(r['width'],r['height'])/1024,transform=t)
        if r['split']=='train':
            gt=np.asarray(r['points'],dtype=float).reshape(-1,2)
            grid=points_to_model(gt,t)/14
            x,y=grid.astype(int).T
            target=np.zeros((h,w),bool);target[y,x]=True
            assert target.sum()==len(gt)
            offsets=np.zeros((h,w,2),np.float32);offsets[y,x]=grid-grid.astype(int)
            it.update(truth=gt,target=torch.as_tensor(target,device='cuda'),offsets=torch.as_tensor(offsets,device='cuda'))
        else:
            assert 'points' not in r
        items.append(it)
    assert state_digest(encoder.state_dict())==plan['encoder_hash']
    return items,forwards,len(cache)


def make_model(seed,mode):
    torch.manual_seed(seed)
    return ContextPointDetector(Proposer().head,mode).cuda()


def main():
    configure()
    plan_path=Path('/job/plan.json')
    plan=json.loads(plan_path.read_text())
    items,forwards,null_grids=load_features(plan)
    train=[it for it in items if it['split']=='train']
    assert len(train)==42 and len(items)==65
    audit,schedules,thresholds={}, {}, {}
    for seed in plan['seeds']:
        rng=np.random.default_rng(seed)
        schedule=np.concatenate([rng.permutation(len(train)) for _ in range(11)])[:420]
        schedules[str(seed)]=[train[i]['id'] for i in schedule]
        for mode in plan['modes']:
            name=f'{mode}_{seed}'
            if mode=='ordinary' and seed==1729:
                continue
            model=make_model(seed,mode)
            initial=state_digest(model.point_head.state_dict())
            opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
            losses=[]
            for step,ix in enumerate(schedule,1):
                loss=optimization_loss(model,train[ix])
                opt.zero_grad();loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                losses.append([float(loss.detach()),float(norm)])
                if step%140==0:write_json(OUT/'status.json',dict(arm=name,step=step))
            assert np.isfinite(losses).all() and all(torch.isfinite(v).all() for v in model.parameters())
            assert all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
            model.eval()
            candidates={it['id']:propose(model,it,None,False)[0] for it in train}
            curve=[metrics(candidates,train,t) for t in plan['thresholds']]
            best=min(curve,key=lambda r:(r['empty_fp'],-r['positive_f1'],r['mae'],-r['tau']))
            thresholds[name]=best['tau']
            ck=OUT/(name+'.pt')
            torch.save(model.state_dict(),ck)
            audit[name]=dict(initial_point_head=initial,steps=len(losses),loss_gradnorm=losses,curve=curve,
                             threshold=best['tau'],checkpoint=digest(ck),parameters=sum(p.numel() for p in model.parameters()))
    write_json(OUT/'training.json',dict(arms=audit,schedules=schedules))
    write_json(OUT/'thresholds.json',thresholds)
    # All eight new endpoints and cutoffs are saved before decoding any development frame.
    predictions=[]
    reference_check=[]
    checkpoint=Path('/job/ordinary_reference.pt')
    assert digest(checkpoint)==plan['ordinary_reference_sha256']
    old=Proposer()
    old.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    original=ContextPointDetector(old.head,'ordinary').cuda().eval()
    for it in items:
        if it['split']=='development':
            reference_check.append(dict(id=it['id'],**propose(original,it,None,False)[0]))
    write_json(OUT/'reference_replay.json',reference_check)
    for seed in plan['seeds']:
        for mode in plan['modes']:
            name=f'{mode}_{seed}'
            if name not in audit:continue
            model=make_model(seed,mode)
            model.load_state_dict(torch.load(OUT/(name+'.pt'),map_location='cuda',weights_only=True))
            model.eval()
            for it in items:
                row=propose(model,it,None,False)[0]
                predictions.append(dict(id=it['id'],arm=name,**row))
    write_json(OUT/'candidates.json',predictions)
    hashes={p.name:digest(p) for p in OUT.iterdir() if p.name!='status.json'}
    write_json(OUT/'report.json',dict(state='completed',steps=8*420,features=65,encoder=plan['encoder_hash'],
               encoder_passes=forwards,null_geometries=null_grids,plan_sha256=digest(plan_path),hashes=hashes))


if __name__=='__main__':
    main()
