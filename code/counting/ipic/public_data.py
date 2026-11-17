"""Audited public foundation pairs; original labels stay outside model inputs."""
import hashlib,json,pathlib
import numpy as np
import torch
from PIL import Image,ImageOps
from .transforms import MEAN,STD,appearance,expanded_affine,resize_pad,reference_crop

def view(image,points,seed):
    original_size=image.size
    rng=np.random.default_rng(seed);long_side=int(rng.choice([768,896,1024]));flip=bool(rng.random()<.5)
    prior=np.eye(3);valid=np.ones((image.height,image.width),bool);affine=None
    if rng.random()<.3:
        affine=[float(rng.uniform(-10,10)),float(rng.uniform(.9,1.1)),float(rng.uniform(-5,5))]
        image,valid,prior=expanded_affine(image,*affine)
    rgb,rect,g=resize_pad(image,long_side,flip,prior_matrix=prior,original_size=original_size)
    nw,nh=g.content_size
    vm=Image.fromarray(valid.astype(np.uint8)*255).resize((nw,nh),Image.Resampling.NEAREST)
    if flip:vm=ImageOps.mirror(vm)
    content_valid=np.zeros_like(rect);content_valid[:nh,:nw]=np.asarray(vm)>0
    # Apply appearance in final input pixels, before normalization; no source crop.
    resized=Image.fromarray(np.uint8(np.clip(rgb[:,:nh,:nw].transpose(1,2,0)*STD+MEAN,0,1)*255+.5))
    changed,appearance_log=appearance(resized,int(rng.integers(0,2**31)))
    rgb[:,:nh,:nw]=((np.asarray(changed,np.float32)/255-MEAN)/STD).transpose(2,0,1)
    rgb[:,~content_valid]=0
    truth=g.points(points).astype(np.float32)
    return rgb,content_valid,truth,{'seed':seed,'long_side':long_side,'flip':flip,'affine':affine,
                                  'appearance':appearance_log,'geometry':g.record()}

def make_draws(rows,updates,seed):
    """Exact half of draws per public source, fixed before any fitting."""
    rng=np.random.default_rng(seed);groups={}
    for i,r in enumerate(rows):
        if r['split']!='train':raise ValueError('Held-out row in public foundation')
        category=r['category'] if r['dataset']=='FSC147' else '|'.join(map(str,sorted({o['category_id'] for o in r['objects']})))
        groups.setdefault(r['dataset'],{}).setdefault(category,[]).append(i)
    count=updates*8
    if count%2:raise ValueError('Even public draw count required')
    sources=np.array(['FSC147']*(count//2)+['D2S']*(count//2));rng.shuffle(sources)
    all_modes=np.array([True]*(count//8)+[False]*(3*count//8));rng.shuffle(all_modes);di=0;draws=[]
    for source in sources:
        keys=sorted(groups[source]);category=keys[int(rng.integers(len(keys)))];indices=groups[source][category];idx=indices[int(rng.integers(len(indices)))]
        all_products=bool(all_modes[di]) if source=='D2S' else False
        if source=='D2S':di+=1
        draws.append({'row_index':idx,'query_id':rows[idx]['query_id'],'dataset':str(source),'all_products':all_products,
                      'seed':int(rng.integers(0,2**31))})
    return draws

def prepare_pair(row,draw,root,device):
    root=pathlib.Path(root);path=root/row['image'];raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('Audited image changed')
    with Image.open(path) as src:image=src.convert('RGB')
    rng=np.random.default_rng(draw['seed'])
    if row['dataset']=='FSC147':points=row['points_pixel_centers'];boxes=row['exemplars_xyxy']
    else:
        cats=sorted({o['category_id'] for o in row['objects']});category=cats[int(rng.integers(len(cats)))]
        target=row['objects'] if draw['all_products'] else [o for o in row['objects'] if o['category_id']==category]
        points=[o['point_pixel_center'] for o in target]
        boxes=[o['exemplar_xyxy'] for o in target[:3]]
    first=view(image,points,int(rng.integers(0,2**31)));second=view(image,points,int(rng.integers(0,2**31)))
    def tensor(x,dtype=None):return torch.as_tensor(np.array(x,copy=True),dtype=dtype,device=device)
    batch={'a':tensor(first[0])[None],'b':tensor(second[0])[None],
           'valid_a':tensor(first[1],torch.bool),'valid_b':tensor(second[1],torch.bool),
           'points_a':tensor(first[2],torch.float32),'points_b':tensor(second[2],torch.float32),'all_products':draw['all_products']}
    if draw['all_products']:batch.update(references=None,reference_boxes=None)
    else:
        refs=[reference_crop(image,b) for b in boxes]
        batch.update(references=tensor(np.stack([r[0] for r in refs])),reference_boxes=tensor(np.stack([r[1] for r in refs])))
    return batch,{'draw':draw,'view_a':first[3],'view_b':second[3]}
