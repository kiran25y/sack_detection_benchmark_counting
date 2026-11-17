"""Generate recording-excluded proposals and a matched, recording-included bank."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from counting.firc.geometry import image_to_model, points_to_model, points_to_native, transform
from counting.firc.model import load_official_encoder
from counting.firc.preflight import configure
from counting.firc.tiny_fit import digest, state_digest, write_json
from point_utils import array, assign

OUT = Path('/output')
STRIDE = 14


class Proposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(768, 48, 1), nn.GroupNorm(6, 48), nn.SiLU(),
                                  nn.Conv2d(48, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
                                  nn.Conv2d(32, 3, 1))

    def forward(self, feat):
        y = self.head(feat.float())[0]
        return y[0], y[1:].sigmoid().permute(1, 2, 0)


def optimization_loss(model, item):
    logits, predicted_offsets = model(item['features'])
    mask = item['target']
    negatives = logits[item['valid'] & ~mask]
    n = int(mask.sum())
    background = F.softplus(negatives.topk(min(max(24, 3*n), negatives.numel())).values).mean()
    if n == 0:
        return background
    return background + F.softplus(-logits[mask]).mean() + 2*F.smooth_l1_loss(
        predicted_offsets[mask], item['offsets'][mask], beta=.1)


@torch.no_grad()
def propose(model, item, projection, describe):
    logits, offsets = model(item['features'])
    prob = logits.sigmoid().cpu().numpy()
    off = offsets.cpu().numpy()
    y, x = np.nonzero(item['valid'].cpu().numpy())
    centers = (np.stack([x, y], axis=1) + off[y, x])*STRIDE
    t = item['transform']
    valid = (centers < np.asarray(t['valid_hw'][::-1])).all(axis=1)
    centers, scores = centers[valid], prob[y[valid], x[valid]]
    native = points_to_native(centers, t)
    indices = []
    for i in np.argsort(-scores, kind='stable'):
        if indices and np.min(np.linalg.norm(native[indices]-native[i], axis=1)) <= item['radius']:
            continue
        indices.append(i)
        if len(indices) == 64:
            break
    xy, score = native[indices], scores[indices]
    result = dict(points=xy.tolist(), scores=score.tolist())
    if not describe:
        return result, None
    ring = np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]], dtype=float)*STRIDE
    coords = np.clip(centers[indices,None,:]+ring, 0, np.asarray(t['valid_hw'][::-1]))
    grid = torch.as_tensor(2*coords/np.asarray(t['padded_hw'][::-1])-1, device='cuda', dtype=torch.float32)
    v = F.grid_sample(item['features'].float(), grid[None], mode='bilinear', padding_mode='border',
                      align_corners=False)[0].permute(1, 2, 0)
    v = F.normalize(v,dim=-1)
    center = v[:,0]
    contrast = F.normalize(center-v[:,1:].mean(dim=1),dim=-1)
    visual = torch.cat([F.normalize(center@projection,dim=-1),
                        F.normalize(contrast@projection,dim=-1)], dim=1)
    return result, visual.cpu().numpy().astype('float32')


def metrics(preds, items, tau):
    f1, errors, empty_fp = [], [], 0
    for it in items:
        p = preds[it['id']]
        xy = array(p['points'])[np.asarray(p['scores'])>=tau]
        gt = it['truth']
        hits = int((assign(xy, gt, it['radius'])>=0).sum())
        errors.append(abs(len(xy)-len(gt)))
        if len(gt):
            f1.append(2*hits/(len(gt)+len(xy)))
        else:
            empty_fp += len(xy)
    return dict(tau=tau, empty_fp=empty_fp, positive_f1=float(np.mean(f1)), mae=float(np.mean(errors)))


def load_items(plan):
    ck=Path('/assets/dinov2_vitb14_reg4_pretrain.pth')
    assert digest(ck)==plan['weight_hash']
    encoder=load_official_encoder(ck).requires_grad_(False).cuda().eval()
    assert state_digest(encoder.state_dict())==plan['encoder_hash']
    items=[]
    for r in plan['rows']:
        p=Path('/data')/r['image']
        assert digest(p)==r['encoded_sha256']
        with Image.open(p) as im:
            assert im.size==(r['width'],r['height'])
            t=transform(*im.size,896,r['roi_xyxy'])
            img=torch.from_numpy(image_to_model(im,t))[None].cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            feat=encoder.get_intermediate_layers(img,n=[11],reshape=True,return_class_token=False,norm=True)[0].half()
        h,w=feat.shape[-2:]; yy,xx=np.ogrid[:h,:w]
        valid=(xx*STRIDE<t['valid_hw'][1])&(yy*STRIDE<t['valid_hw'][0])
        it=dict(id=r['image_id'], split=r['split'], group=r['group'], transform=t, features=feat,
                radius=8*max(im.size)/1024,valid=torch.as_tensor(valid,device='cuda'))
        if r['split']=='train':
            it['truth']=array(r['points'])
            grid=points_to_model(it['truth'],t)/STRIDE
            x,y=grid.astype(int).T
            mask=np.zeros((h,w),bool); mask[y,x]=True
            assert mask.sum()==len(grid) and valid[mask].all()
            offsets=np.zeros((h,w,2),np.float32);offsets[y,x]=grid-grid.astype(int)
            it.update(target=torch.as_tensor(mask,device='cuda'),offsets=torch.as_tensor(offsets,device='cuda'))
        else:
            assert 'points' not in r
        items.append(it)
    assert state_digest(encoder.state_dict())==plan['encoder_hash']
    return items


def main():
    configure()
    plan=json.loads(Path('/job/plan.json').read_text())
    items=load_items(plan)
    train=[i for i in items if i['split']=='train']
    groups=sorted({i['group'] for i in train})
    assert len(groups)==3 and len(train)==42 and len(items)==65
    # Fixed random projection uses no image statistics from any recording.
    projection=torch.as_tensor(np.random.default_rng(1301).normal(size=(768,96)).astype('float32'),device='cuda')
    receipts, records, descriptors = {}, [], {}
    for fold,excluded in enumerate([*groups,None]):
        source=[i for i in train if i['group']!=excluded]
        ids=[i['id'] for i in source]
        assert not excluded or all(it['group']!=excluded for it in source)
        torch.manual_seed(1729)
        model=Proposer().cuda()
        start=state_digest(model.state_dict())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
        rng=np.random.default_rng(1729)
        schedule=np.concatenate([rng.permutation(len(source)) for _ in range(420//len(source)+1)])[:420]
        losses=[]
        for step,ix in enumerate(schedule,1):
            loss=optimization_loss(model,source[ix]); optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
            if step%42==0:
                write_json(OUT/'status.json',dict(fold=fold,step=step,excluded=excluded))
        assert np.isfinite(losses).all() and all(torch.isfinite(p).all() for p in model.parameters())
        model.eval()
        name='fold'+str(fold)
        ck=OUT/(name+'.pt');torch.save(model.state_dict(),ck)
        source_preds={i['id']:propose(model,i,projection,False)[0] for i in source}
        curve=[metrics(source_preds,source,t) for t in plan['thresholds']]
        best=min(curve,key=lambda r:(r['empty_fp'],-r['positive_f1'],r['mae'],-r['tau']))
        receipts[name]=dict(excluded=excluded,fit_ids=ids,calibration_ids=ids,threshold=best,
                            curve=curve,initial=start,checkpoint=digest(ck),steps=420,
                            order=[ids[j] for j in schedule],loss=losses)
        write_json(OUT/(name+'_freeze.json'),receipts[name])
        for it in items:
            if it['split']=='development':
                if excluded is not None:continue
                mode='development'
            else:
                if excluded is None:continue
                g=groups.index(it['group'])
                if fold==g:mode='excluded'
                elif fold==(g+1)%3:mode='included'
                else:continue
            p,visual=propose(model,it,projection,True)
            key=mode+'_'+it['id']
            descriptors[key]=visual
            records.append(dict(key=key,mode=mode,id=it['id'],fold=name,threshold=best['tau'],**p))
    assert len(records)==107 and len({v['initial'] for v in receipts.values()})==1
    write_json(OUT/'banks.json',records)
    np.savez_compressed(OUT/'descriptor.npz',**descriptors)
    hashes={p.name:digest(p) for p in OUT.iterdir() if p.name!='status.json'}
    write_json(OUT/'report.json',dict(state='completed',steps=1680,encoder=plan['encoder_hash'],
               plan_sha256=digest(Path('/job/plan.json')),output_hashes=hashes,unique_features=len(items)))


if __name__=='__main__':
    main()
