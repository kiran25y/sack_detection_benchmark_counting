"""Native forwards with fixed-order resampling adjoints for the B0 runtime.

No learned weights, geometry, predictions or losses change. BF16 interpolation
gradients accumulate in FP32 in a fixed order; the small reference-pool adjoint
uses the native CPU FP32 kernel. FP64 is retained for derivative verification.
Both assignment arms must use this same explicit numerical amendment.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch.nn import functional as native_functional
from torchvision.ops import roi_align as native_roi_align


class _Bilinear2x(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.shape = tuple(x.shape); ctx.dtype = x.dtype
        # F.interpolate switches to a different arithmetic decomposition when
        # strict determinism is enabled, even in a custom forward. Call the
        # retained native forward operator directly; our adjoint handles backward.
        return torch.ops.aten.upsample_bilinear2d.default(
            x, [2*x.shape[-2], 2*x.shape[-1]], False, None, None)

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled(): raise RuntimeError('First-order optimizer runtime only')
        g = grad if grad.dtype == torch.float64 else grad.float()
        # Transpose of half-pixel aligned 2x linear interpolation. Clamping the
        # four output indices implements the repeated-edge weights exactly.
        for axis, length in ((-2, ctx.shape[-2]), (-1, ctx.shape[-1])):
            index = torch.arange(length, device=g.device)*2
            a = g.index_select(axis, (index-1).clamp(0, 2*length-1))
            b = g.index_select(axis, index)
            c = g.index_select(axis, (index+1).clamp(0, 2*length-1))
            d = g.index_select(axis, (index+2).clamp(0, 2*length-1))
            g = ((a*.25 + b*.75) + c*.75) + d*.25
        return g.to(ctx.dtype)


def bilinear2x(x, size=None, scale_factor=None, mode='nearest', align_corners=None,
               recompute_scale_factor=None, antialias=False):
    if x.ndim!=4 or mode!='bilinear' or align_corners is not False or antialias or recompute_scale_factor is not None:
        raise ValueError('Stable B0 resize supports only 2D bilinear, align_corners=False, no antialias')
    expected=(2*x.shape[-2],2*x.shape[-1])
    if not ((size is not None and tuple(size)==expected and scale_factor is None) or
            (size is None and scale_factor in (2,2.0,(2,2)))):
        raise ValueError('Only the exact B0 pyramid 2x resize is supported')
    return _Bilinear2x.apply(x)


class _ROIAlignStable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, boxes, ph, pw, scale, ratio, aligned):
        if boxes.requires_grad: raise ValueError('Native exemplar coordinates are fixed inputs')
        ctx.shape=tuple(x.shape);ctx.device=x.device;ctx.dtype=x.dtype
        ctx.parameters=(scale,ph,pw,ratio,aligned)
        ctx.save_for_backward(boxes.detach())
        # Same native forward kernel/Autocast dispatch as the retained runtime;
        # do not trigger torchvision's global-determinism Python fallback.
        return torch.ops.torchvision.roi_align(x, boxes, scale, ph, pw, ratio, aligned)

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled(): raise RuntimeError('First-order optimizer runtime only')
        boxes,=ctx.saved_tensors;scale,ph,pw,ratio,aligned=ctx.parameters
        dtype=torch.float64 if grad.dtype==torch.float64 else torch.float32
        cpu_grad=grad.detach().to(device='cpu',dtype=dtype).contiguous()
        cpu_boxes=boxes.to(device='cpu',dtype=dtype).contiguous()
        result=torch.ops.torchvision._roi_align_backward(cpu_grad,cpu_boxes,scale,ph,pw,
                                                        *ctx.shape,ratio,aligned)
        return result.to(device=ctx.device,dtype=ctx.dtype),None,None,None,None,None,None


def roi_align(x, boxes, output_size, spatial_scale=1., sampling_ratio=-1, aligned=False):
    if not torch.is_tensor(boxes) or boxes.ndim!=2 or boxes.shape[1]!=5:
        raise ValueError('Require native tensor ROI coordinates with batch column')
    ph,pw=(output_size,output_size) if isinstance(output_size,int) else output_size
    if (ph,pw)!=(3,3) or spatial_scale!=1/8 or sampling_ratio!=2 or aligned is not True:
        raise ValueError('Only the unchanged B0 exemplar-pooling configuration is supported')
    return _ROIAlignStable.apply(x,boxes,ph,pw,spatial_scale,sampling_ratio,aligned)


class _FunctionalProxy:
    interpolate=staticmethod(bilinear2x)
    def __getattr__(self, name): return getattr(native_functional,name)


def install():
    from . import model
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8': raise RuntimeError('Wrong cuBLAS workspace setting')
    model.F=_FunctionalProxy(); model.roi_align=roi_align
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {'native_forward_kernels_preserved':True,'bilinear_backward':'fixed-order FP32 2x half-pixel adjoint',
            'roi_backward':'native CPU FP32 reference adjoint; only small reference feature grid crosses devices',
            'strict_deterministic_algorithms':True,'warn_only':False,'cublas_workspace_config':':4096:8',
            'same_runtime_both_arms':True,'first_order_only':True,'supports_B0_only':True,
            'numerical_amendment_not_bitwise_equal_to_old_nondeterministic_backward':True}
