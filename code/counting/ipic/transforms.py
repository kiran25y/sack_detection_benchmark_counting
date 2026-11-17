"""Whole-image transforms with an explicit pixel-center homography and no crops."""
from dataclasses import dataclass
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


@dataclass
class Geometry:
    matrix: np.ndarray
    original_size: tuple
    content_size: tuple
    padded_size: tuple

    def points(self, xy, inverse=False):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        mat = np.linalg.inv(self.matrix) if inverse else self.matrix
        p = np.c_[xy, np.ones(len(xy))] @ mat.T
        return p[:, :2] / p[:, 2:]

    def boxes(self, boxes):
        b = np.asarray(boxes, np.float64).reshape(-1, 4)
        corners = np.stack((b[:, [0,1]], b[:, [2,1]], b[:, [2,3]], b[:, [0,3]]), 1)
        v = self.points(corners.reshape(-1, 2)).reshape(-1, 4, 2)
        return np.c_[v.min(1), v.max(1)]

    def record(self):
        return {'matrix':self.matrix.tolist(), 'original_size':self.original_size,
                'content_size':self.content_size, 'padded_size':self.padded_size,
                'coordinate_convention':'continuous pixel centers; integer raster centers'}


def decode_oriented(path):
    with Image.open(path) as src:
        w, h = src.size; o = int(src.getexif().get(274, 1))
        transforms = {1:[[1,0,0],[0,1,0]],2:[[-1,0,w-1],[0,1,0]],
                      3:[[-1,0,w-1],[0,-1,h-1]],4:[[1,0,0],[0,-1,h-1]],
                      5:[[0,1,0],[1,0,0]],6:[[0,-1,h-1],[1,0,0]],
                      7:[[0,-1,h-1],[-1,0,w-1]],8:[[0,1,0],[-1,0,w-1]]}
        if o not in transforms: raise ValueError('Unsupported EXIF orientation')
        mat = np.array(transforms[o]+[[0,0,1]], dtype=float)
        return ImageOps.exif_transpose(src).convert('RGB'), mat, (w,h)


def resize_pad(image, long_side=1024, flip=False, prior_matrix=None, original_size=None):
    w,h = image.size; scale = long_side/max(w,h)
    nw,nh = max(1,round(w*scale)),max(1,round(h*scale))
    mat = np.array([[nw/w,0,.5*nw/w-.5],[0,nh/h,.5*nh/h-.5],[0,0,1.]])
    image = image.resize((nw,nh), Image.Resampling.BILINEAR)
    if flip:
        image = ImageOps.mirror(image)
        mat = np.array([[-1,0,nw-1],[0,1,0],[0,0,1.]]) @ mat
    if prior_matrix is not None: mat = mat @ prior_matrix
    pw,ph = ((nw+31)//32)*32,((nh+31)//32)*32
    rgb = np.broadcast_to(MEAN,(ph,pw,3)).copy()
    rgb[:nh,:nw] = np.asarray(image,np.float32)/255
    valid = np.zeros((ph,pw),bool);valid[:nh,:nw] = True
    return ((rgb-MEAN)/STD).transpose(2,0,1),valid,Geometry(mat,original_size or (w,h),(nw,nh),(pw,ph))


def appearance(image, seed):
    """At most two appearance changes, with the exact document probabilities/ranges."""
    rng=np.random.default_rng(seed); candidates=[]
    for name,p in [('brightness',.25),('contrast',.25),('saturation',.2),('gamma',.2),
                   ('noise',.15),('blur',.15),('jpeg',.15)]:
        if rng.random()<p:candidates.append(name)
    rng.shuffle(candidates); log=[]
    for name in candidates[:2]:
        if name in ('brightness','contrast','saturation'):
            value=float(rng.uniform(.8,1.2)); cls={'brightness':ImageEnhance.Brightness,
                    'contrast':ImageEnhance.Contrast,'saturation':ImageEnhance.Color}[name]
            image=cls(image).enhance(value)
        elif name=='gamma':
            value=float(rng.uniform(.8,1.2));image=Image.fromarray(np.uint8(255*(np.asarray(image)/255.)**value))
        elif name=='noise':
            value=float(rng.uniform(0,.02));v=np.asarray(image)/255.+rng.normal(0,value,(*image.size[::-1],3))
            image=Image.fromarray(np.uint8(np.clip(v,0,1)*255))
        elif name=='blur':
            value=float(rng.uniform(0,1));image=image.filter(ImageFilter.GaussianBlur(value))
        else:
            value=int(rng.integers(70,101));b=io.BytesIO();image.save(b,format='JPEG',quality=value)
            b.seek(0);image=Image.open(b).convert('RGB')
        log.append({'name':name,'value':value})
    return image,{'seed':seed,'operations':log,'visibility_review_required':True}


def expanded_affine(image, angle, scale, shear):
    """Expand output to keep the whole source. Return validity and forward matrix."""
    if not (-10<=angle<=10 and .9<=scale<=1.1 and -5<=shear<=5):
        raise ValueError('Affine parameters outside frozen starting recipe')
    w,h=image.size;t=np.deg2rad(angle);s=np.tan(np.deg2rad(shear))
    linear=scale*np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]) @ np.array([[1,s],[0,1]])
    bounds=np.array([[-.5,-.5],[w-.5,-.5],[w-.5,h-.5],[-.5,h-.5]])@linear.T
    low=bounds.min(0); high=bounds.max(0);size=np.ceil(high-low).astype(int)
    mat=np.eye(3);mat[:2,:2]=linear;mat[:2,2]=-.5-low
    inv=np.linalg.inv(mat)
    # PIL affine maps pixel corners; conjugate the pixel-center transform.
    shift=np.array([[1,0,.5],[0,1,.5],[0,0,1.]])
    pil_inverse=shift@inv@np.linalg.inv(shift)
    args=tuple(pil_inverse[:2].reshape(-1));fill=tuple(round(x*255) for x in MEAN)
    output=image.transform(tuple(size),Image.Transform.AFFINE,args,Image.Resampling.BILINEAR,fillcolor=fill)
    valid=Image.new('L',(w,h),255).transform(tuple(size),Image.Transform.AFFINE,args,Image.Resampling.NEAREST,fillcolor=0)
    return output,np.asarray(valid)>0,mat


def reference_crop(image, box):
    x0,y0,x1,y1=map(float,box)
    if not (0<=x0<x1<=image.width and 0<=y0<y1<=image.height):raise ValueError('Invalid native exemplar box')
    crop=image.crop((int(np.floor(x0)),int(np.floor(y0)),int(np.ceil(x1)),int(np.ceil(y1))))
    w,h=crop.size;s=128/max(w,h);nw,nh=max(1,round(w*s)),max(1,round(h*s))
    arr=np.broadcast_to(MEAN,(128,128,3)).copy()
    arr[:nh,:nw]=np.asarray(crop.resize((nw,nh),Image.Resampling.BILINEAR))/255.
    return ((arr-MEAN)/STD).transpose(2,0,1).astype(np.float32),np.array([0,0,nw,nh],np.float32)
