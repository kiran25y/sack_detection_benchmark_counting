"""Frozen sack/public draws and identity-preserving paired training inputs."""
import collections, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from pycocotools import mask as coco_mask
from .public_data import view, prepare_pair as public_pair
from .transforms import reference_crop, resize_pad,expanded_affine,appearance,MEAN,STD
from .renderer import Layer, render, align_pair, AmbiguousVisibility
from .adaptation_core import validate_prepared_pair

PUBLIC='research-direction-20261012/ipic_checks_20261030/public_v1/public_train_manifest.jsonl'
PUBLIC_SHA='8a70c247df7f304b46cd47b215de23e71c83a1aaeff58def3e5b8522b35e3721'
ASSETS='research-direction-20261012/ipic_sack_d2s_assets_v1_20261103'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def warp_ids(array,geometry,fill=-1):
    """Nearest ownership transport in the same continuous pixel-center chart."""
    matrix=np.array(geometry['matrix']);shift=np.array([[1,0,.5],[0,1,.5],[0,0,1.]])
    inv=shift@np.linalg.inv(matrix)@np.linalg.inv(shift)
    return np.asarray(Image.fromarray(np.asarray(array,np.int32)).transform(tuple(geometry['padded_size']),
        Image.Transform.AFFINE,tuple(inv[:2].reshape(-1)),Image.Resampling.NEAREST,fillcolor=fill),dtype=np.int64)

def raster_anchor(ownership,identity,nominal):
    """Synthetic-only projected interior point with unit bilinear mask support.

    Nearest-neighbor label rasterization changes visible pixel boundaries. Record
    this anchor change explicitly; never use it to replace a native benchmark point.
    """
    mask=ownership==identity
    interior=mask[:-1,:-1]&mask[1:,:-1]&mask[:-1,1:]&mask[1:,1:]
    y,x=np.where(interior)
    if not len(x):raise AmbiguousVisibility('No unit-membership interior pixel')
    i=np.argmin((x-nominal[0])**2+(y-nominal[1])**2)
    return np.array([x[i],y[i]],np.float32)

def pair_view(image,points,geometry_seed,appearance_seed,appearance_function=None):
    """Separate geometry and appearance draws so each edit family is identifiable."""
    original_size=image.size;rng=np.random.default_rng(geometry_seed)
    long_side=int(rng.choice([768,896,1024]));flip=bool(rng.random()<.5);prior=np.eye(3)
    valid=np.ones((image.height,image.width),bool);affine=None
    if rng.random()<.3:
        affine=[float(rng.uniform(-10,10)),float(rng.uniform(.9,1.1)),float(rng.uniform(-5,5))]
        image,valid,prior=expanded_affine(image,*affine)
    rgb,rect,g=resize_pad(image,long_side,flip,prior_matrix=prior,original_size=original_size);nw,nh=g.content_size
    vm=Image.fromarray(valid.astype(np.uint8)*255).resize((nw,nh),Image.Resampling.NEAREST)
    if flip:vm=ImageOps.mirror(vm)
    content=np.zeros_like(rect);content[:nh,:nw]=np.asarray(vm)>0
    resized=Image.fromarray(np.uint8(np.clip(rgb[:,:nh,:nw].transpose(1,2,0)*STD+MEAN,0,1)*255+.5))
    changed,alog=(appearance if appearance_function is None else appearance_function)(resized,appearance_seed);rgb[:,:nh,:nw]=((np.asarray(changed,np.float32)/255-MEAN)/STD).transpose(2,0,1)
    rgb[:,~content]=0
    return rgb,content,g.points(points).astype(np.float32),dict(seed=geometry_seed,long_side=long_side,flip=flip,affine=affine,appearance=alog,geometry=g.record())

class Data:
    def __init__(self,root,package):
        self.root=Path(root);self.package=Path(package)
        if sha(self.root/PUBLIC)!=PUBLIC_SHA:raise ValueError('Public manifest changed')
        self.public=list(map(json.loads,(self.root/PUBLIC).read_text().splitlines()))
        self.sacks=json.loads((self.package/'sack_manifest.json').read_text())
        self.train=[r for r in self.sacks if r['split']=='train'];self.dev=[r for r in self.sacks if r['split']=='development']
        self.assets=json.loads((self.package/'asset_background_review.json').read_text())['assets']
        if len(self.assets)!=120 or not all(r['reviewed_mask'] for r in self.assets):raise ValueError('Unreviewed assets')
        self.bycat=collections.defaultdict(list)
        for r in self.assets:self.bycat[r['category']].append(r)
        self.categories=sorted(self.bycat);self.cache={}
        ann=json.loads((self.root/'datasets/D2S-native-v1/annotations/D2S_training.json').read_text())
        self.ann={r['id']:r for r in ann['annotations']}
        self.sackrefs=json.loads((self.package/'references.json').read_text())
        self.background=self.image(self.package/'inputs/backgrounds/sack_floor.png')

    def image(self,path):
        with Image.open(path) as im:return im.convert('RGB')

    def asset(self,row):
        key=row['asset_id']
        if key not in self.cache:
            base=self.root/ASSETS;im=base/row['image'];mask=base/row['mask']
            if sha(im)!=row['sha256'] or sha(mask)!=row['mask_sha256']:raise ValueError('Asset changed')
            self.cache[key]=(self.image(im),Image.open(mask).convert('L'))
        return self.cache[key]

    def references(self,category):
        if category=='filled_sack':
            refs=[]
            for r in self.sackrefs:
                path=self.package/r['image']
                if sha(path)!=r['sha256']:raise ValueError('Sack reference changed')
                im=self.image(path);refs.append(reference_crop(im,[0,0,im.width,im.height]))
        else:
            refs=[reference_crop(self.asset(r)[0],[0,0,*self.asset(r)[0].size]) for r in self.bycat[category]]
        return np.stack([r[0] for r in refs]),np.stack([r[1] for r in refs])

    def draws(self):
        rng=np.random.default_rng(1729);draws=[]
        bins=collections.defaultdict(list)
        for i,r in enumerate(self.train):bins['zero' if r['count']==0 else '1_4' if r['count']<=4 else '5_14'].append(i)
        for i in range(4000):
            band=sorted(bins)[int(rng.integers(len(bins)))];idx=int(rng.choice(bins[band]))
            draws.append(dict(dataset='sacks',row_index=idx,query_id=self.train[idx]['image_id'],all_products=False,family='ordinary',seed=int(rng.integers(2**31))))
        grouped=collections.defaultdict(list);d2s=[]
        for i,r in enumerate(self.public):
            if r['dataset']=='FSC147':grouped[r['category']].append(i)
            else:d2s.append(i)
        family_counters=collections.Counter()
        for i in range(2000):
            cat=sorted(grouped)[int(rng.integers(len(grouped)))];idx=int(rng.choice(grouped[cat]));r=self.public[idx]
            draws.append(dict(dataset='FSC147',row_index=idx,query_id=r['query_id'],all_products=False,family='ordinary',seed=int(rng.integers(2**31))))
        families=['add_remove']*250+['replacement']*150+['occlusion']*200+['distractor']*200+['appearance']*100+['geometry']*100
        rng.shuffle(families);modes=np.array([True]*500+[False]*1500);rng.shuffle(modes)
        zeros=set(map(int,rng.choice(np.where(~modes)[0],200,replace=False)))
        # Native scene/category grouping before choosing an image.
        groups=collections.defaultdict(list)
        for idx in d2s:groups[tuple(sorted({o['category_id'] for o in self.public[idx]['objects']}))].append(idx)
        keys=sorted(groups)
        for i in range(2000):
            group=keys[int(rng.integers(len(keys)))];idx=int(rng.choice(groups[group]));r=self.public[idx]
            draws.append(dict(dataset='D2S',row_index=idx,query_id=r['query_id'] if i<1000 else f'composite_{i-1000:04}',
                all_products=bool(modes[i]),verified_zero_query=i in zeros,composite=i>=1000,
                family='ordinary' if i<1000 else families[i-1000],seed=int(rng.integers(2**31))))
            if i>=1000:
                family=draws[-1]['family'];number=family_counters[family];family_counters[family]+=1
                if family=='add_remove':draws[-1]['edit_direction']='addition' if number%2==0 else 'removal'
                if family=='occlusion':draws[-1]['support_band_index']=number%3
        rng.shuffle(draws)
        for i,d in enumerate(draws):d['draw_id']=i
        return draws

    def _layer(self,row,index,cell,scale=.75):
        im,mask=self.asset(row);factor=128*scale/max(im.size);size=tuple(max(1,round(x*factor)) for x in im.size)
        im=im.resize(size,Image.Resampling.BILINEAR);mask=mask.resize(size,Image.Resampling.NEAREST)
        cx,cy=cell;x=int(cx-size[0]//2);y=int(cy-size[1]//2)
        rgb=np.zeros((768,1024,3),np.uint8);m=np.zeros((768,1024),bool)
        rgb[y:y+size[1],x:x+size[0]]=np.asarray(im);m[y:y+size[1],x:x+size[0]]=np.asarray(mask)>127
        return Layer(str(index),row['asset_id'],row['category'],rgb,m,row['parent_id'],'train',True)

    def composite(self,draw):
        rng=np.random.default_rng(draw['seed']);bg=np.asarray(self.background.resize((1024,768),Image.Resampling.BILINEAR));family=draw['family']
        # Retry only invalid rendering geometry, before any model output is read.
        failures=[]
        for attempt in range(80):
            cats=list(rng.choice(self.categories,int(rng.integers(1,5)),replace=False));category=str(rng.choice(cats))
            if draw.get('verified_zero_query'):category=str(rng.choice([c for c in self.categories if c not in cats]))
            n=int(rng.integers(4,48 if family=='occlusion' else 49));cells=[(64+128*x,64+128*y) for y in range(6) for x in range(8)];rng.shuffle(cells)
            source=[];layers=[]
            for i in range(n):
                cat=cats[i%len(cats)];row=self.bycat[cat][int(rng.integers(2))];source.append(row)
                layers.append(self._layer(row,i,cells[i],float(rng.uniform(.65,.85))))
            if not draw.get('verified_zero_query'):
                if family=='add_remove':category=source[-1]['category']
                elif family in ('replacement','occlusion'):category=source[0]['category']
            left=list(layers);right=list(layers);edit={'family':family,'attempt':attempt}
            if family=='add_remove':
                if draw['edit_direction']=='addition':left=left[:-1];edit['direction']='addition'
                else:right=right[:-1];edit['direction']='removal'
            elif family=='replacement':
                # Distinct IDs, different non-overlapping positions, equal object category.
                left=left[:-1];right=right[1:];replacement=self._layer(source[0],n,cells[-1],.75)
                right[-1]=replacement
                # Rebuild explicitly: shared middle objects; first removed / new at last.
                right=list(layers[1:-1])+[replacement]
            elif family=='occlusion':
                # A product occludes another. Select coverage geometrically, never from count prediction.
                y,x=np.where(layers[0].mask);target=draw['support_band_index'];low,high=[(.8,1.),(.6,.8),(.4,.6)][target]
                found=False
                for shift in range(10,95,3):
                    cx=int((x.min()+x.max())/2+shift);cy=int((y.min()+y.max())/2)
                    if cx>960 or cx<64 or cy>704 or cy<64:continue
                    candidate=self._layer(source[1],n,(cx,cy),.85)
                    retained=float((layers[0].mask&~candidate.mask).sum()/layers[0].mask.sum())
                    if low<=retained<high:
                        right.append(candidate);edit.update(retained_band=[low,high],retained_support=retained);found=True;break
                if not found:failures.append('occlusion_band');continue
            elif family=='distractor' and not draw['all_products']:
                cat=str(rng.choice([c for c in self.categories if c!=category]));row=self.bycat[cat][0]
                # replace a background-only reserved cell by a declared non-target object
                if n==48:left=left[:-1];right=right[:-1]
                cell=cells[n] if n<48 else cells[-1];right.append(self._layer(row,n,cell,.75))
            try:
                bg_b=bg[:,::-1].copy() if family=='distractor' and draw['all_products'] else bg
                if family=='distractor' and draw['all_products']:edit['subtype']='verified_background_plate_reflection'
                a=render(bg,left,category,'train',True);b=render(bg_b,right,category,'train',True)
                if draw['all_products']:
                    from .renderer import visible_anchor
                    for v in (a,b):
                        pts=[]
                        for j,r in enumerate(v['ledger'],1):
                            r['target_eligibility']=int(r['visible_pixels']>0)
                            if r['target_eligibility']:pts.append(visible_anchor(v['ownership']==j))
                        v['points']=np.asarray(pts,np.float32).reshape(-1,2);v['count']=len(pts)
                pair=align_pair(a,b)
                if draw.get('verified_zero_query') and any(v['count'] for v in (a,b)):raise ValueError('Synthetic absent target violated')
                pair.update(category=category,edit=edit,rejected_geometry=failures);return pair
            except AmbiguousVisibility as exc:failures.append(str(exc))
        raise ValueError('No valid deterministic rendering after80 candidates: '+str(draw))

    def prepare(self,draw,device='cpu',sack_appearance_function=None):
        rng=np.random.default_rng(draw['seed']);dataset=draw['dataset'];ownership=None;eligible=None;ids=None;ledger={}
        if dataset=='FSC147':
            row=self.public[draw['row_index']];batch,log=public_pair(row,draw,self.root,device)
            record=dict(**log,split='train',exhaustive_reviewed=True,annotation_type='complete_points',dataset=dataset,
                query_id=draw['query_id'],dependency_group='FSC:'+row['query_id'],count_a=len(batch['points_a']),count_b=len(batch['points_b']))
            validate_prepared_pair(batch,record);return batch,record
        if dataset=='sacks':
            row=self.train[draw['row_index']];path=self.package/row['image']
            if sha(path)!=row['encoded_sha256']:raise ValueError('Sack image identity changed')
            image=self.image(path);images=[image,image];points=[row['points'],row['points']];refs=self.references('filled_sack')
            roi=np.zeros((image.height,image.width),np.int32);x0,y0,x1,y1=row['roi_xyxy'];roi[y0:y1,x0:x1]=1
            group=row['dependency_group'];annotation='complete_points'
        elif draw['composite']:
            pair=self.composite(draw);images=[Image.fromarray(v['rgb']) for v in pair['views']];points=[v['points'] for v in pair['views']]
            ownership=pair['ownership'];eligible=pair['eligible'];ids=pair['region_ids'];refs=None if draw['all_products'] else self.references(pair['category'])
            group='D2S:composite_training_only';annotation='complete_masks';roi=None
            ledger={k:pair[k] for k in ('region_ids','delta_count','category','edit','rejected_geometry')}
            ledger['instance_ledger']=[v['ledger'] for v in pair['views']]
        else:
            row=self.public[draw['row_index']];path=self.root/row['image']
            if sha(path)!=row['image_sha256']:raise ValueError('D2S image changed')
            image=self.image(path);images=[image,image];cats=sorted({str(o['category_id']) for o in row['objects']})
            category=str(rng.choice([c for c in self.categories if c not in cats] if draw.get('verified_zero_query') else cats))
            target=row['objects'] if draw['all_products'] else [o for o in row['objects'] if str(o['category_id'])==category]
            points=[[o['point_pixel_center'] for o in target]]*2
            own=np.zeros((image.height,image.width),np.int64);z=[0.];ids=['__background__']
            for j,o in enumerate(row['objects'],1):
                ann=self.ann[o['annotation_id']];seg=ann['segmentation'];mask=coco_mask.decode(seg).astype(bool)
                if (own[mask]!=0).any():raise ValueError('Overlapping native visible ownership masks')
                own[mask]=j;z.append(float(draw['all_products'] or str(o['category_id'])==category));ids.append('D2S:'+str(o['annotation_id']))
            ownership=[own,own];eligible=[np.asarray(z,np.float32)]*2;roi=None;annotation='complete_masks';group='D2S:'+str(row['image_id'])
            if draw['all_products']:refs=None
            elif target:
                rr=[reference_crop(image,o['exemplar_xyxy']) for o in target[:3]];refs=(np.stack([r[0] for r in rr]),np.stack([r[1] for r in rr]))
            else:refs=self.references(category)
        batch={'all_products':draw['all_products']};record=dict(draw=draw,split='train',exhaustive_reviewed=True,
            annotation_type=annotation,dataset=dataset,query_id=draw['query_id'],dependency_group=group,**ledger)
        transform_logs=[];own_transformed=[];invalid_regions=set()
        def tensor(x,dtype=None):return torch.as_tensor(np.array(x,copy=True),device=device,dtype=dtype)
        geometry_seeds=[int(rng.integers(2**31)) for _ in range(2)];appearance_seeds=[int(rng.integers(2**31)) for _ in range(2)]
        if draw.get('composite'):
            if draw['family']!='geometry':geometry_seeds[1]=geometry_seeds[0]
            if draw['family']!='appearance':appearance_seeds[1]=appearance_seeds[0]
        for j,side in enumerate(('a','b')):
            arr,valid,pts,log=pair_view(images[j],points[j],geometry_seeds[j],appearance_seeds[j],
                appearance_function=sack_appearance_function if dataset=='sacks' else None)
            if roi is not None:valid=valid&(warp_ids(roi,log['geometry'],0)==1)
            batch[side]=tensor(arr)[None];batch['valid_'+side]=tensor(valid,torch.bool);batch['points_'+side]=tensor(pts,torch.float32)
            record['count_'+side]=len(pts);record['verified_target_absent_'+side]=len(pts)==0
            if ownership is not None:
                own=warp_ids(ownership[j],log['geometry']);own[~valid]=-1;own_transformed.append(tensor(own,torch.long))
                # All four support pixels of an annotated regional anchor must belong to its instance.
                active=np.flatnonzero(eligible[j]);assert len(active)==len(pts)
                if draw.get('composite'):
                    original_ids=[r['instance_id'] for r in pair['views'][j]['ledger'] if r['target_eligibility']]
                    corrected=np.array([raster_anchor(own,ids.index(identifier),p) for identifier,p in zip(original_ids,pts)],np.float32).reshape(-1,2)
                    log['synthetic_raster_anchor_projection']=[dict(instance_id=i,nominal=p.tolist(),final=q.tolist(),anchor_equivariance_valid=bool(np.array_equal(p,q))) for i,p,q in zip(original_ids,pts,corrected)]
                    pts=corrected;batch['points_'+side]=tensor(pts,torch.float32)
                seen=[]
                for point in pts:
                    x,y=np.floor(point).astype(int)
                    if x<0 or y<0 or x+1>=own.shape[1] or y+1>=own.shape[0]:raise ValueError('Regional anchor outside chart')
                    identity=int(own[y,x]);seen.append(identity)
                    if identity not in active or not np.all(own[y:y+2,x:x+2]==identity):
                        if draw.get('composite'):raise ValueError('Regional point lacks unit mask membership')
                        invalid_regions.update(active.tolist())
                if sorted(seen)!=sorted(active):
                    if draw.get('composite'):raise ValueError('Regional anchor/identity bijection failed')
                    invalid_regions.update(active.tolist())
            transform_logs.append(log)
        batch.update(references=None if refs is None else tensor(refs[0]),reference_boxes=None if refs is None else tensor(refs[1]))
        if ownership is not None:
            valid_regions=np.ones(len(ids),bool)
            if invalid_regions:
                # Keep all native point targets and both images; withhold only
                # ambiguous local regions, including background, in BOTH views.
                valid_regions[:]=False
            batch.update(ownership=own_transformed,eligible=[tensor(e,torch.float32) for e in eligible],region_valid=tensor(valid_regions,torch.bool))
            record.update(region_ids=ids,paired_mask_audit_passed=True,local_regions_withheld_for_native_anchor=bool(invalid_regions))
        record.update(view_a=transform_logs[0],view_b=transform_logs[1]);validate_prepared_pair(batch,record)
        return batch,record
