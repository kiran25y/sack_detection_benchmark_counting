"""Explicit native integer pixel-center to model pixel-edge convention."""
import math
import numpy as np
from PIL import Image
MEAN=np.array([.485,.456,.406],dtype=np.float32)
STD=np.array([.229,.224,.225],dtype=np.float32)


def transform(width,height,long_side,roi=None,pad_multiple=112):
    x0,y0,x1,y1=(0,0,width,height) if roi is None else roi
    if any(int(v)!=v for v in (x0,y0,x1,y1)) or not 0<=x0<x1<=width or not 0<=y0<y1<=height:raise ValueError('Integer rectangle required')
    scale=long_side/max(x1-x0,y1-y0);w=round((x1-x0)*scale);h=round((y1-y0)*scale)
    return {'roi':[int(v) for v in (x0,y0,x1,y1)],'valid_hw':[h,w],
            'padded_hw':[math.ceil(h/pad_multiple)*pad_multiple,math.ceil(w/pad_multiple)*pad_multiple],
            'scale_xy':[w/(x1-x0),h/(y1-y0)],'native_convention':'integer pixel centers','model_convention':'pixel edges; point centers transformed with +0.5'}


def points_to_model(points,t):
    p=np.asarray(points,dtype=float).reshape(-1,2);edge=p+.5-np.array(t['roi'][:2])
    if (edge<0).any() or (edge[:,0]>t['roi'][2]-t['roi'][0]).any() or (edge[:,1]>t['roi'][3]-t['roi'][1]).any():raise ValueError('Label outside exhaustive ROI')
    return edge*np.array(t['scale_xy'])


def points_to_native(points,t):return np.asarray(points,dtype=float).reshape(-1,2)/np.array(t['scale_xy'])+np.array(t['roi'][:2])-.5


def image_to_model(image,t):
    if image.getexif().get(274,1)!=1:raise ValueError('Explicit EXIF point transform required; do not rotate RGB alone')
    rgb=image.convert('RGB').crop(t['roi']);h,w=t['valid_hw'];ph,pw=t['padded_hw']
    resized=np.asarray(rgb.resize((w,h),Image.Resampling.BILINEAR),dtype=np.float32)/255
    out=np.zeros((ph,pw,3),dtype=np.float32);out[:h,:w]=(resized-MEAN)/STD
    return out.transpose(2,0,1).copy()
