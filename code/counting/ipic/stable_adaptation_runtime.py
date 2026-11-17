"""Native forwards and fixed-order adjoints for B0/B1 integer resize factors.

The retained B0 2x implementation is reused exactly. B1 additionally requires
identity, 4x and 8x projection resizes. This is an engineering implementation,
not an automatically validated runtime: CPU/CUDA guards must pass first.
"""
from . import stable_resampling as prior
import torch


class _IntegerResize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        ctx.shape=tuple(x.shape);ctx.dtype=x.dtype;ctx.factor=factor
        return torch.ops.aten.upsample_bilinear2d.default(
            x,[factor*x.shape[-2],factor*x.shape[-1]],False,None,None)

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():raise RuntimeError('First-order optimizer runtime only')
        value=grad if grad.dtype==torch.float64 else grad.float();factor=ctx.factor
        for axis,length in ((-2,ctx.shape[-2]),(-1,ctx.shape[-1])):
            inputs=torch.arange(length,device=value.device)
            total=None
            for offset in range(2*factor):
                index=factor*inputs-factor//2+offset
                valid=(index>=0)&(index<factor*length)
                clipped=index.clamp(0,factor*length-1)
                source=((clipped.to(value.dtype)+.5)/factor-.5).clamp(0,length-1)
                weights=(1-(source-inputs).abs()).clamp_min(0)*valid
                shape=[1]*value.ndim;shape[axis]=length
                term=value.index_select(axis,clipped)*weights.reshape(shape)
                total=term if total is None else total+term
            value=total
        return value.to(ctx.dtype),None


def interpolate(x,size=None,scale_factor=None,mode='nearest',align_corners=None,
                recompute_scale_factor=None,antialias=False):
    if x.ndim!=4 or mode!='bilinear' or align_corners is not False or antialias or recompute_scale_factor is not None:
        raise ValueError('Only prescribed 2D bilinear interpolation is supported')
    if size is not None and scale_factor is not None:raise ValueError('Ambiguous resize')
    if size is None:
        scales=(scale_factor,scale_factor) if isinstance(scale_factor,(int,float)) else scale_factor
        if not scales or len(scales)!=2:raise ValueError('Explicit size/factor required')
        size=tuple(int(n*s) for n,s in zip(x.shape[-2:],scales))
    size=tuple(size);factors=tuple(o//i for o,i in zip(size,x.shape[-2:]))
    if len(size)!=2 or factors[0]!=factors[1] or factors[0] not in (1,2,4,8) or tuple(n*factors[0] for n in x.shape[-2:])!=size:
        raise ValueError('Only equal integer factors1,2,4,8 are supported')
    factor=factors[0]
    if factor==1:return x
    if factor==2:return prior.bilinear2x(x,size=size,mode=mode,align_corners=False)
    return _IntegerResize.apply(x,factor)


class _Proxy:
    interpolate=staticmethod(interpolate)
    def __getattr__(self,name):return getattr(prior.native_functional,name)


def install():
    runtime=prior.install()
    from . import model
    model.F=_Proxy()
    runtime.update(supports_B0_only=False,supported_architectures=['B0','B1'],
                   bilinear_backward='retained B0 2x; fixed-order FP32 separable integer4x/8x adjoint',
                   validated_by_install=False)
    return runtime
