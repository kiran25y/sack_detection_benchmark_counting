"""Transfer the published P2PNet checkpoint with fixed-budget point supervision."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from models import build_model


JOB,OUT=Path('/job'),Path('/output')


def write(name,value):
    dest=OUT/name
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False))
    tmp.replace(dest)


def load(row):
    import hashlib
    p=Path(row['path'])
    with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
    roi=np.array(row['roi'])
    with Image.open(p) as im:
        assert list(im.size)==row['size']
        crop=im.convert('RGB').crop(roi)
    wh=np.array(crop.size)//128*128
    rgb=np.asarray(crop.resize(tuple(wh),Image.Resampling.LANCZOS))
    scale=wh/(roi[2:]-roi[:2])
    pts=(np.asarray(row.get('points',[]),float).reshape(-1,2)+.5-roi[:2])*scale-.5
    return rgb,pts,scale


def tensor(rgb):
    x=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()/255
    return ((x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]).cuda()


def sample(item,rng):
    rgb,pts,_=item
    h,w=rgb.shape[:2]
    corner=rng.uniform([0,0],[w-256,h-256])
    if len(pts) and rng.random()<.5:
        corner=np.clip(pts[rng.integers(len(pts))]-rng.uniform(32,224,2),[0,0],[w-256,h-256])
    x,y=corner.astype(int)
    points=pts-np.array([x,y])
    points=points[np.all((points>=0)&(points<256),axis=1)]
    patch=rgb[y:y+256,x:x+256]
    if rng.random()<.5:
        patch=patch[:,::-1];points[:,0]=255-points[:,0]
    return tensor(patch),dict(point=torch.as_tensor(points,device='cuda',dtype=torch.float32),
                              labels=torch.ones(len(points),device='cuda',dtype=torch.long))


def main():
    import hashlib
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    plan=json.loads((JOB/'plan.json').read_text())
    weights=JOB/'weights/SHTechA.pth'
    with weights.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==plan['weight_sha256']
    state=torch.load(weights,map_location='cpu',weights_only=True)['model']
    args=SimpleNamespace(backbone='vgg16_bn',row=2,line=2,set_cost_class=1.,set_cost_point=.05,point_loss_coef=.0002,eos_coef=.5)
    model,criterion=build_model(args,training=True)
    model.cuda();criterion.cuda()
    rows=plan['rows']
    fit=[r for r in rows if r['split']=='train'];ev=[r for r in rows if r['split']!='train']
    assert len(fit)==42 and len(ev)==74 and all('points' not in r for r in ev)
    cache={r['id']:load(r) for r in fit}
    methods=['pretrained']+[str(i) for i in plan['seeds']]
    checkpoints={}
    with (OUT/'updates.jsonl').open('x') as ledger:
        for method in methods:
            model.load_state_dict(state,strict=True)
            if method!='pretrained':
                seed=int(method);torch.manual_seed(seed);rng=np.random.default_rng(seed)
                model.train()
                for m in model.modules():
                    if isinstance(m,torch.nn.BatchNorm2d):m.eval()
                opt=torch.optim.AdamW([dict(params=[p for n,p in model.named_parameters() if not n.startswith('backbone.')]),
                                       dict(params=model.backbone.parameters(),lr=1e-5)],lr=1e-4,weight_decay=1e-4)
                draws=np.concatenate([rng.permutation(len(fit)) for _ in range(12)])
                for step in range(plan['steps']):
                    ids=[fit[i]['id'] for i in draws[2*step:2*step+2]]
                    samples=[sample(cache[k],rng) for k in ids]
                    x=torch.stack([im for im,_ in samples]);targets=[target for _,target in samples]
                    outputs=model(x)
                    if sum(len(t['point']) for t in targets):
                        losses=criterion(outputs,targets)
                        loss=losses['loss_ce']+.0002*losses['loss_point']
                    else:
                        logits=outputs['pred_logits']
                        loss=F.cross_entropy(logits.transpose(1,2),torch.zeros(logits.shape[:2],device='cuda',dtype=torch.long),weight=criterion.empty_weight)
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);opt.step()
                    ledger.write(json.dumps(dict(seed=seed,step=step+1,ids=ids,loss=float(loss.detach())),allow_nan=False)+'\n')
                    if (step+1)%25==0 or step==0:
                        ledger.flush();write('status.json',dict(stage='training',seed=seed,step=step+1))
                assert all(torch.isfinite(p).all() for p in model.parameters())
                dest=OUT/(method+'.pt');torch.save(model.state_dict(),dest)
                with dest.open('rb') as f:checkpoints[method]=hashlib.file_digest(f,'sha256').hexdigest()
                del opt
            model.eval();pred={}
            with torch.inference_mode():
                for r in ev:
                    rgb,_,scale=load(r)
                    out=model(tensor(rgb)[None])
                    logits=out['pred_logits'][0];xy=out['pred_points'][0]
                    assert torch.isfinite(logits).all() and torch.isfinite(xy).all()
                    keep=logits.softmax(-1)[:,1]>.5
                    native=(xy[keep].cpu().numpy()+.5)/scale+np.array(r['roi'][:2])-.5
                    pred[r['id']]=native.tolist()
            write(method+'.json',pred)
            write('status.json',dict(stage='evaluated',arm=method,frames=len(pred)))
    hashes={}
    for file in OUT.iterdir():
        if file.suffix in ('.jsonl','.json') and file.name!='status.json':
            with file.open('rb') as f:hashes[file.name]=hashlib.file_digest(f,'sha256').hexdigest()
    write('report.json',dict(state='completed',steps=plan['steps']*len(plan['seeds']),checkpoints=checkpoints,hashes=hashes,frames_per_endpoint=len(ev),methods=methods))


if __name__=='__main__':
    main()
