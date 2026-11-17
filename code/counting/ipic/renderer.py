"""Deterministic layered compositor; eligibility follows rendered visible support.

Synthetic fixture success is not a manual recognizability audit. Real assets must
carry reviewed masks, parent ancestry and a verified clean background record.
"""
from dataclasses import dataclass
import numpy as np


class AmbiguousVisibility(ValueError):
    pass


@dataclass
class Layer:
    instance_id: str
    asset_id: str
    category: str
    rgb: np.ndarray
    mask: np.ndarray
    parent_id: str
    split: str
    reviewed_mask: bool


def visible_anchor(mask):
    """Nearest interior pixel to visible centroid; unit membership at integer center."""
    y,x=np.where(mask)
    if not len(x):return None
    i=np.argmin((x-x.mean())**2+(y-y.mean())**2)
    return np.array([float(x[i]),float(y[i])],np.float32)


def render(background, layers, query, split, background_verified=False):
    if not background_verified:raise ValueError('Exact edit requires a reviewed clean background')
    h,w=background.shape[:2]
    if background.shape!=(h,w,3) or not np.isfinite(background).all():raise ValueError('Invalid RGB')
    if len({v.instance_id for v in layers})!=len(layers):raise ValueError('Rendered copies need distinct IDs')
    rgb=background.copy();own=np.zeros((h,w),np.int64)
    for i,l in enumerate(layers,1):
        if l.split!=split or not l.reviewed_mask:raise ValueError('Unreviewed mask or split leakage')
        if l.mask.shape!=(h,w) or l.mask.dtype!=bool or l.rgb.shape!=rgb.shape:raise ValueError('Layer geometry mismatch')
        rgb[l.mask]=l.rgb[l.mask];own[l.mask]=i
    ledger=[];points=[]
    for i,l in enumerate(layers,1):
        area=int(l.mask.sum());visible=own==i;va=int(visible.sum())
        if area==0:raise ValueError('Empty pre-occlusion mask')
        support=va/area;anchor=visible_anchor(visible)
        y,x=np.where(visible);scale=1024/max(h,w)
        short=0 if not va else min(x.max()-x.min()+1,y.max()-y.min()+1)*scale
        if va and (support<.4 or va*scale**2<100 or short<8):
            raise AmbiguousVisibility(f'Unreviewed visible fragment for{l.instance_id}; reject/resample')
        eligible=bool(va and l.category==query)
        if eligible:points.append(anchor)
        source_anchor=visible_anchor(l.mask)
        ledger.append({'instance_id':l.instance_id,'asset_id':l.asset_id,'parent_id':l.parent_id,
                       'split':split,'category':l.category,'pre_occlusion_pixels':area,
                       'visible_pixels':va,'retained_support':support,'target_eligibility':int(eligible),
                       'source_anchor':source_anchor.tolist(),'final_anchor':None if anchor is None else anchor.tolist(),
                       'anchor_equivariance_valid':bool(anchor is not None and np.array_equal(anchor,source_anchor)),
                       'recognizability_review_required':True})
    return {'rgb':rgb,'ownership':own,'points':np.array(points,np.float32).reshape(-1,2),
            'ledger':ledger,'count':len(points),'region_instances':['__background__']+[l.instance_id for l in layers]}


def align_pair(a,b):
    """Map both integer ownership rasters to the union of instance IDs."""
    ids=sorted(set(a['region_instances'][1:])|set(b['region_instances'][1:])); union=['__background__']+ids
    lookup={k:i for i,k in enumerate(union)};masks=[];elig=[]
    for v in (a,b):
        mapping=np.array([lookup[k] for k in v['region_instances']],np.int64)
        masks.append(mapping[v['ownership']]);z=np.zeros(len(union),np.float32)
        for row in v['ledger']:z[lookup[row['instance_id']]]=row['target_eligibility']
        elig.append(z)
    return {'ownership':masks,'eligible':elig,'region_ids':union,'region_valid':np.ones(len(union),bool),
            'delta_count':int(elig[1].sum()-elig[0].sum()),'views':[a,b]}
