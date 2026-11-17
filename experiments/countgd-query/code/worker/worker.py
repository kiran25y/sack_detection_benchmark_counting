"""One frozen CountGD++ forward per ROI and matched point-supervised query adapters."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F
from clarification_capture import CountGD


JOB, OUT = Path('/job'), Path('/output')
ARMS = ('geometry', 'decoder')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,value):
    dest=OUT/name
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False))
    tmp.replace(dest)


def rgb(row):
    path=Path(row['path'])
    assert sha(path)==row['sha256']
    with Image.open(path) as im:
        assert list(im.size)==row['size']
        return im.convert('RGB').crop(row['roi'])


def match(xy,gt,radius):
    d=np.linalg.norm(xy[:,None]-gt[None],axis=2)/radius
    a,b=linear_sum_assignment(np.where(d<=1,d,max(d.shape)+1))
    valid=d[a,b]<=1
    return a[valid],b[valid]


class QueryAdapter(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim+6,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,row,semantic):
        dec=row['features'] if semantic else torch.zeros_like(row['features'])
        out=self.net(torch.cat((dec,row['geometry']),1))
        logit=row['prior']+out[:,0]
        move=out[:,1:].tanh()
        move=16*move/torch.linalg.vector_norm(move,dim=1,keepdim=True).clamp_min(1)
        return logit,move


class Detector:
    def __init__(self,ref,box):
        self.backend=CountGD()
        self.prompts={'image':ref,'points':[[*box[:2],2.,*box[2:],3.]]}
        self.values={}
        def decoder_hook(module,inputs,output):
            self.values['features']=output[0][-1][0].detach().float().cpu()
        def model_hook(module,inputs,output):
            self.values['boxes']=output['pred_boxes'][0].float().cpu()
            logits=output['pred_logits'][0].sigmoid()
            stop=int(torch.where(output['input_ids'][0]==1012)[0][0])+1
            self.values['positive']=logits[:,:stop].amax(1).float().cpu()
            self.values['negative']=logits[:,stop:].amax(1).float().cpu()
        self.backend.model.transformer.register_forward_hook(decoder_hook)
        self.backend.model.register_forward_hook(model_hook)

    @torch.no_grad()
    def capture(self,row):
        self.values.clear()
        im=rgb(row)
        boxes,scores=self.backend.author.predict(self.backend.model,self.backend.transform,im,'sack',
                                                 self.prompts,[],[],torch.device('cuda'))
        v=self.values
        positive,negative=v['positive'],v['negative']
        chosen=(positive>.23)&(positive>negative)
        np.testing.assert_allclose(v['boxes'][chosen].numpy(),boxes,rtol=1e-5,atol=1e-6)
        assert len(scores)==int(chosen.sum())
        assert v['boxes'].shape==(900,4) and v['features'].shape==(900,256)
        assert all(torch.isfinite(x).all() for x in v.values())
        xy=v['boxes'][:,:2].double().numpy()*im.size+np.array(row['roi'][:2])-.5
        prior=torch.logit(positive.clamp(1e-5,1-1e-5))
        geometry=torch.cat((v['boxes'],positive[:,None],negative[:,None]),1)
        item=dict(row=row,xy=xy,unit=max(row['size'])/1024,prior=prior,geometry=geometry,
                  features=F.layer_norm(v['features'],(256,)),native=xy[chosen.numpy()].tolist())
        if row['split']=='train':
            gt=np.asarray(row['points'],float).reshape(-1,2)
            ids,labels=match(xy,gt,16*item['unit'])
            pos=np.zeros(900,bool);pos[ids]=True
            offset=np.zeros((900,2),np.float32);offset[ids]=(gt[labels]-xy[ids])/item['unit']
            item.update(gt=gt,pos=torch.from_numpy(pos),offset=torch.from_numpy(offset))
        return item


def loss_for(model,item,semantic):
    logit,move=model(item,semantic)
    pos=item['pos']
    neg=logit[~pos]
    n=int(pos.sum())
    loss=F.softplus(neg.topk(min(len(neg),max(12,3*n))).values).mean()
    if n:
        loss=loss+F.softplus(-logit[pos]).mean()+F.smooth_l1_loss(move[pos]/8,item['offset'][pos]/8)
    count=torch.log1p(logit.sigmoid().sum())
    return loss+.25*F.smooth_l1_loss(count,count.new_tensor(np.log1p(len(item['gt']))))


@torch.no_grad()
def response(model,item,semantic):
    logits,move=model(item,semantic)
    roi=np.asarray(item['row']['roi'])
    shifted=np.clip(item['xy']+item['unit']*move.numpy(),roi[:2]-.5,roi[2:]-.5)
    assert np.isfinite(shifted).all() and torch.isfinite(logits).all()
    return logits.sigmoid().numpy(),shifted


def rank_keep(scores,coords,unit,threshold=0,limit=None):
    # Cells of side one exclusion radius avoid an O(N²) pass for low thresholds.
    order=np.argsort(-scores,kind='stable')
    keep=[];cells={};radius=8*unit
    for i in order:
        if float(scores[i])<threshold or (limit is not None and len(keep)>=limit):break
        x,y=np.floor(coords[i]/radius).astype(int)
        neighbors=[j for a in (x-1,x,x+1) for b in (y-1,y,y+1) for j in cells.get((a,b),())]
        if neighbors and np.min(np.linalg.norm(coords[neighbors]-coords[i],axis=1))<=radius:continue
        keep.append(int(i));cells.setdefault((x,y),[]).append(int(i))
    return keep


def calibrate(items,forecasts):
    floor=float(np.nextafter(max(float(forecasts[it['row']['id']][0].max()) for it in items if not len(it['gt'])),np.inf))
    cuts=sorted({floor,1.000001,*[i/20 for i in range(1,20) if i/20>=floor]})
    curves={t:{} for t in cuts}
    for it in items:
        score,xy=forecasts[it['row']['id']]
        ids=rank_keep(score,xy,it['unit'],floor)
        for t in cuts:
            selected=xy[[i for i in ids if float(score[i])>=t]]
            if not len(it['gt']):
                assert len(selected)==0
                continue
            tp=len(match(selected,it['gt'],8*it['unit'])[0])
            n=len(it['gt']);m=len(selected)
            cost=1-2*tp/(n+m)+abs(n-m)/(n+1)
            curves[t].setdefault(it['row']['group'],[]).append(cost)
    values=[dict(threshold=t,cost=float(np.mean([np.mean(v) for v in g.values()]))) for t,g in curves.items()]
    return dict(best=min(values,key=lambda x:(x['cost'],-x['threshold'])),curve=values,empty_floor=floor)


def main():
    plan=json.loads((JOB/'plan.json').read_text())
    for k,v in json.loads((JOB/'external.json').read_text()).items():
        assert sha(Path(k))==v,k
    torch.manual_seed(42);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    source=next(r for r in plan['rows'] if r['id']==plan['reference_id'])
    offset=np.array(source['roi'][:2]*2)
    refbox=(np.asarray(plan['reference_box'])-offset).tolist()
    model=Detector(rgb(source),refbox)
    versions={n:p._version for n,p in model.backend.model.named_parameters()}
    train_rows=[r for r in plan['rows'] if r['split']=='train']
    eval_rows=[r for r in plan['rows'] if r['split']!='train']
    assert len(train_rows)==42 and len(eval_rows)==74 and all('points' not in r for r in eval_rows)
    fit=[]
    for r in train_rows:
        fit.append(model.capture(r))
        write('status.json',dict(phase='training features',forwards=len(fit)))
    groups=sorted({r['group'] for r in train_rows})
    oof={k:{} for k in ARMS};fits=[]
    steps=plan['steps']
    with (OUT/'updates.jsonl').open('x') as log:
        for fold,held in enumerate([*groups,None]):
            items=[it for it in fit if it['row']['group']!=held]
            torch.manual_seed(1729)
            heads={'geometry':QueryAdapter(256)}
            heads['decoder']=copy.deepcopy(heads['geometry'])
            opts={a:torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.01) for a,h in heads.items()}
            rng=np.random.default_rng(1729)
            order=np.concatenate([rng.permutation(len(items)) for _ in range(steps//len(items)+1)])[:steps]
            for step,i in enumerate(order,1):
                it=items[i];losses={}
                for a,h in heads.items():
                    opts[a].zero_grad(set_to_none=True)
                    loss=loss_for(h,it,a=='decoder');loss.backward()
                    nn.utils.clip_grad_norm_(h.parameters(),1.,error_if_nonfinite=True)
                    opts[a].step();losses[a]=loss.item()
                log.write(json.dumps(dict(fold=fold,step=step,id=it['row']['id'],losses=losses),allow_nan=False)+'\n')
                if step%50==0 or step==1:
                    log.flush();write('status.json',dict(phase='training',fold=fold,steps_per_head=step,updates=2*(fold*steps+step)))
            checkpoint={}
            for a,h in heads.items():
                assert all(torch.isfinite(p).all() for p in h.parameters())
                file=OUT/f'fold{fold}_{a}.pt'
                torch.save(h.state_dict(),file);checkpoint[a]=sha(file)
                for it in fit:
                    if it['row']['group']==held:oof[a][it['row']['id']]=response(h,it,a=='decoder')
            fits.append(dict(excluded=held,ids=[it['row']['id'] for it in items],hashes=checkpoint))
    cuts={a:calibrate(fit,oof[a]) for a in ARMS}
    write('freeze.json',dict(calibration=cuts,fits=fits))
    records={};raw_fill={}
    for r in eval_rows:
        it=model.capture(r);pred=dict(native=it['native'])
        outputs={a:response(heads[a],it,a=='decoder') for a in ARMS}
        for a,(sc,xy) in outputs.items():
            keep=rank_keep(sc,xy,it['unit'],cuts[a]['best']['threshold'])
            pred[a]=xy[keep].tolist()
            if a=='decoder':
                pred['same_selected_unmoved']=it['xy'][keep].tolist()
                raw=rank_keep(it['geometry'][:,4].numpy(),it['xy'],it['unit'],limit=len(keep))
                # All heads draw from the same 900 queries. If raw NMS exhausts its pool,
                # fill in raw rank order to match the prescribed count and log this case.
                if len(raw)<len(keep):
                    raw_fill[r['id']]=len(keep)-len(raw)
                    seen=set(raw);raw.extend([int(j) for j in np.argsort(-it['geometry'][:,4].numpy(),kind='stable') if j not in seen][:len(keep)-len(raw)])
                pred['equal_budget_native']=it['xy'][raw].tolist()
        records[r['id']]=pred
        write('status.json',dict(phase='sealed evaluation',frames=len(records),updates=8*steps))
    assert all(p._version==versions[n] for n,p in model.backend.model.named_parameters())
    write('predictions.json',records)
    hashes={p.name:sha(p) for p in OUT.iterdir() if p.name!='status.json'}
    write('report.json',dict(state='completed',calls=len(fit)+len(records),updates=8*steps,raw_fill=raw_fill,
                             hashes=hashes,plan_sha256=sha(JOB/'plan.json')))
    write('status.json',dict(phase='completed',forwards=116,updates=8*steps))


if __name__=='__main__':
    main()
