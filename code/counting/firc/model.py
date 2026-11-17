"""Specified FIRC architecture. Official frozen DINO, newly initialized counter.

No checkpoint from IPIC/CounTR is compatible with this model. Input RGB is
ImageNet-normalized; valid_hw uses pixel-edge rectangles before right/bottom pad.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align as _register_roi_ops


class StableROI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, boxes):
        if boxes.requires_grad: raise ValueError('Reference boxes are fixed metadata')
        ctx.shape=tuple(x.shape);ctx.device=x.device;ctx.dtype=x.dtype
        ctx.save_for_backward(boxes)
        return torch.ops.torchvision.roi_align(x, boxes, 1/8, 3, 3, 2, True)
    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled(): raise RuntimeError('First-order runtime only')
        boxes,=ctx.saved_tensors
        dtype=torch.float64 if grad.dtype==torch.float64 else torch.float32
        out=torch.ops.torchvision._roi_align_backward(grad.detach().to('cpu',dtype).contiguous(),
            boxes.to('cpu',dtype).contiguous(),1/8,3,3,*ctx.shape,2,True)
        return out.to(ctx.device,ctx.dtype),None


class ChannelLN(nn.Module):
    def __init__(self,c): super().__init__();self.norm=nn.LayerNorm(c,eps=1e-6)
    def forward(self,x):return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)


class DetailBlock(nn.Module):
    def __init__(self):
        super().__init__();self.dw=nn.Conv2d(64,64,7,padding=3,groups=64)
        self.net=nn.Sequential(ChannelLN(64),nn.Conv2d(64,128,1),nn.GELU(),nn.Conv2d(128,64,1))
        self.scale=nn.Parameter(torch.full((1,64,1,1),.001))
    def forward(self,x):return x+self.scale*self.net(self.dw(x))


class Neck(nn.Module):
    def __init__(self):
        super().__init__();self.projections=nn.ModuleList([nn.Conv2d(768,192,1) for _ in range(4)])
        self.smooth=nn.ModuleList([nn.Sequential(nn.Conv2d(192,192,3,padding=1),nn.GroupNorm(32,192,eps=1e-5),nn.GELU()) for _ in range(4)])
    def forward(self,taps,hw):
        levels=[F.interpolate(p(t),size=(math.ceil(hw[0]/s),math.ceil(hw[1]/s)),mode='bilinear',align_corners=False)
                for p,t,s in zip(self.projections,taps,(4,8,16,32))]
        out=[None]*4
        for i in range(3,-1,-1):
            x=levels[i]
            if i<3:x=x+F.interpolate(out[i+1],size=x.shape[-2:],mode='bilinear',align_corners=False)
            out[i]=self.smooth[i](x)
        return out


class FIRC(nn.Module):
    def __init__(self, encoder):
        super().__init__();self.encoder=encoder.requires_grad_(False).eval();self.neck=Neck()
        self.detail=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.GroupNorm(8,32),nn.GELU(),
            nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.GELU(),DetailBlock(),DetailBlock(),nn.Conv2d(64,192,1))
        self.alpha=nn.Parameter(torch.zeros(()))
        self.film=nn.Sequential(nn.Linear(192,384),nn.GELU(),nn.Linear(384,384))
        self.condition=nn.Conv2d(193,192,1)
        blocks=[]
        for _ in range(2):blocks.extend([nn.Conv2d(192,192,3,padding=1,groups=192),nn.Conv2d(192,192,1),nn.GroupNorm(32,192,eps=1e-5),nn.GELU()])
        self.head=nn.Sequential(*blocks,nn.Conv2d(192,3,1))
        # Keep ordinary PyTorch weight initialization. Only biases are prescribed.
        with torch.no_grad():self.head[-1].bias.zero_();self.head[-1].bias[0]=math.log(.001/.999)
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def encode(self,x):
        if x.ndim!=4 or x.shape[-1]%14 or x.shape[-2]%14:raise ValueError('DINO input must be patch divisible')
        with torch.no_grad():
            chunks=[self.encoder.get_intermediate_layers(v,n=[2,5,8,11],reshape=True,return_class_token=False,norm=True) for v in x.split(1)]
            return tuple(torch.cat([c[i] for c in chunks],0) for i in range(4))
    def prototypes(self,references,boxes,taps=None):
        if not 1<=len(references)<=3 or references.shape[-2:]!=(224,224):raise ValueError('One to three padded 224-square reference crops')
        if boxes.shape!=(len(references),5):raise ValueError('Reference ROI shape')
        if not torch.equal(boxes[:,0],torch.arange(len(references),device=boxes.device,dtype=boxes.dtype)):raise ValueError('Reference box indices')
        if not torch.isfinite(boxes).all() or (boxes[:,1:3]<0).any() or (boxes[:,3:]>224).any() or (boxes[:,3:]<=boxes[:,1:3]).any():raise ValueError('Invalid reference rectangle')
        p8=self.neck(taps if taps is not None else self.encode(references),(224,224))[1]
        # Native ROI forward and CPU FP32 adjoint; gradients reach the common neck.
        with torch.autocast(device_type=references.device.type,enabled=False):
            return StableROI.apply(p8.float(),boxes.float()).mean((-2,-1))
    def forward(self,images,references,reference_boxes,valid_hw,detail_enabled=True,taps=None,reference_taps=None):
        if images.shape[-2]%112 or images.shape[-1]%112:raise ValueError('Pad queries to multiple 112')
        if len(valid_hw)!=len(images):raise ValueError('One valid rectangle per image')
        proto=self.prototypes(references,reference_boxes,reference_taps)
        scale,bias=self.film(proto.mean(0)).chunk(2)
        levels=self.neck(taps if taps is not None else self.encode(images),images.shape[-2:]);f4=levels[0]
        if detail_enabled:f4=f4+self.alpha*self.detail(images)
        cosine=torch.einsum('bchw,rc->brhw',F.normalize(f4.float(),dim=1,eps=1e-6),F.normalize(proto.float(),dim=1,eps=1e-6)).amax(1,keepdim=True)
        cond=(1+.1*scale.tanh())[None,:,None,None]*f4+.1*bias.tanh()[None,:,None,None]
        raw=self.head(self.condition(torch.cat((cond,cosine.to(cond.dtype)),1))).float()
        h,w=raw.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device=raw.device),torch.arange(w,device=raw.device),indexing='ij')
        anchors=torch.stack((4*xx+2,4*yy+2),-1).reshape(-1,2).float()
        rows=[]
        for i,(vh,vw) in enumerate(valid_hw):
            if vh<=0 or vw<=0 or vh>images.shape[-2] or vw>images.shape[-1]:raise ValueError('Invalid query rectangle')
            valid=(anchors[:,0]<vw)&(anchors[:,1]<vh)
            a=anchors[valid];logits=raw[i,0].flatten()[valid]
            xy=a+16*raw[i,1:].permute(1,2,0).reshape(-1,2)[valid].tanh()
            xy=torch.stack((xy[:,0].clamp(0,vw),xy[:,1].clamp(0,vh)),-1)
            rows.append({'points':xy,'logits':logits,'probabilities':logits.sigmoid(),'anchors':a})
        return rows


def load_official_encoder(checkpoint):
    from dinov2.hub.backbones import dinov2_vitb14_reg
    encoder=dinov2_vitb14_reg(pretrained=False)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    encoder.load_state_dict(state,strict=True)
    return encoder


def optimizer_for(model,lr=2e-4):
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(no_decay if p.ndim<=1 or name.endswith('.bias') else decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':.05},{'params':no_decay,'weight_decay':0}],lr=lr,betas=(.9,.999),eps=1e-8)
