"""Ordinary supervised data; target-specific exemplars, no factorial labels."""
import hashlib,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance,ImageOps
from .geometry import transform,points_to_model,image_to_model,MEAN,STD

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def target(row,category_id=None):
    if row['dataset']=='FSC147':
        return np.asarray(row['points_pixel_centers'],float).reshape(-1,2),row['exemplars_xyxy']
    if row['dataset']=='D2S':
        selected=[o for o in row['objects'] if o['category_id']==category_id]
        assert selected and category_id is not None
        return np.asarray([o['point_pixel_center'] for o in selected],float),[o['exemplar_xyxy'] for o in sorted(selected,key=lambda o:o['annotation_id'])[:3]]
    assert row['dataset']=='sacks'
    return np.asarray(row['points'],float).reshape(-1,2),None

def augmented_geometry(row,draw):
    pts,_=target(row,draw.get('category_id'))
    t=transform(row['width'],row['height'],draw['long_side'],row.get('roi_xyxy'))
    xy=points_to_model(pts,t)
    if draw['flip']:xy[:,0]=t['valid_hw'][1]-xy[:,0]
    return xy,t

class Data:
    def __init__(self,data,project,plan):
        self.data,self.project,self.plan=Path(data),Path(project),plan;self.checked=set();self.ref_cache={}
    def image(self,row):
        path=(self.data if row['dataset']=='sacks' else self.project)/row['image']
        expected=row.get('image_sha256',row.get('encoded_sha256'))
        if str(path) not in self.checked:
            assert sha(path)==expected, str(path);self.checked.add(str(path))
        with Image.open(path) as im:
            assert im.size==(row['width'],row['height']) and im.getexif().get(274,1)==1
            return im.convert('RGB')
    def reference(self,row,draw,im):
        key=(row['dataset'],row.get('query_id',row.get('image_id')),draw.get('category_id')) if row['dataset']!='sacks' else ('sacks',)
        if key in self.ref_cache:return self.ref_cache[key]
        _,boxes=target(row,draw.get('category_id'));crops=[];identities=[]
        if boxes is None:
            for r in self.plan['sack_references']:
                p=self.data/r['image'];assert r['split']=='train' and sha(p)==r['sha256']
                with Image.open(p) as source:crops.append(source.convert('RGB'))
                identities.append(r['reference_id'])
        else:
            for b in boxes:
                x0,y0,x1,y1=b
                box=[max(0,math.floor(x0)),max(0,math.floor(y0)),min(im.width,math.ceil(x1)),min(im.height,math.ceil(y1))]
                assert box[2]>box[0] and box[3]>box[1]
                crops.append(im.crop(box));identities.append(box)
        refs=[];roi=[]
        for i,crop in enumerate(crops):
            t=transform(*crop.size,224);t['padded_hw']=[224,224];refs.append(image_to_model(crop,t))
            h,w=t['valid_hw'];roi.append([i,0,0,w,h])
        result=(np.stack(refs),np.asarray(roi,np.float32),identities)
        # Bounded CPU cache; no model features or labels cached into inference inputs.
        if len(self.ref_cache)>=128:self.ref_cache.pop(next(iter(self.ref_cache)))
        self.ref_cache[key]=result;return result
    def prepare(self,row,draw):
        assert row['split']=='train' or draw.get('evaluation',False)
        im=self.image(row);refs,boxes,ids=self.reference(row,draw,im)
        xy,t=augmented_geometry(row,draw);h,w=t['valid_hw'];ph,pw=t['padded_hw']
        cropped=im.crop(t['roi']).resize((w,h),Image.Resampling.BILINEAR)
        if draw['flip']:cropped=ImageOps.mirror(cropped)
        for operation,factor in draw['appearance']:
            cls={'brightness':ImageEnhance.Brightness,'contrast':ImageEnhance.Contrast,'saturation':ImageEnhance.Color}[operation]
            cropped=cls(cropped).enhance(factor)
        rgb=np.zeros((ph,pw,3),np.float32);rgb[:h,:w]=(np.asarray(cropped,np.float32)/255-MEAN)/STD
        assert np.isfinite(rgb).all() and np.isfinite(xy).all()
        # Analytic inverse, including native half-pixel and horizontal reflection.
        back=xy.copy()
        if draw['flip']:back[:,0]=w-back[:,0]
        back=back/np.asarray(t['scale_xy'])+np.asarray(t['roi'][:2])-.5
        gt,_=target(row,draw.get('category_id'));error=float(np.max(abs(back-gt))) if len(gt) else 0.
        assert error<1e-8
        return {'image':rgb.transpose(2,0,1).copy(),'truth':xy.astype(np.float32),'references':refs,'boxes':boxes,'valid_hw':t['valid_hw']}, {'draw':draw,'transform':t,'count':len(xy),'reference_ids':ids,'roundtrip_error':error}
