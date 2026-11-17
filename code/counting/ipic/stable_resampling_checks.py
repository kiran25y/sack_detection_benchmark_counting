"""Independent derivative fixtures and actual GPU-kernel repeatability audit."""
import argparse
import json
import pathlib
import warnings

import numpy as np
import torch

from .stable_resampling import bilinear2x, roi_align, native_functional, native_roi_align


def check_cpu():
    torch.set_num_threads(2);torch.manual_seed(713)
    checks=[]
    for h,w in ((1,1),(1,4),(3,1),(3,5),(8,12)):
        x=torch.randn(2,3,h,w,dtype=torch.float64,requires_grad=True)
        upstream=torch.randn(2,3,2*h,2*w,dtype=torch.float64)
        a=bilinear2x(x,size=(2*h,2*w),mode='bilinear',align_corners=False)
        b=native_functional.interpolate(x,size=(2*h,2*w),mode='bilinear',align_corners=False)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ga=torch.autograd.grad(a,x,upstream)[0];gb=torch.autograd.grad(b,x,upstream)[0]
        torch.testing.assert_close(ga,gb,rtol=1e-12,atol=1e-12)
        # Independent dense linear interpolation matrix, including boundaries.
        def weights(n):
            t=np.zeros((2*n,n))
            for j in range(2*n):
                source=max(0,(j+.5)/2-.5);lo=int(np.floor(source));f=source-lo
                t[j,min(lo,n-1)]+=1-f;t[j,min(lo+1,n-1)]+=f
            return t
        expected=np.einsum('oi,bcoq,qj->bcij',weights(h),upstream.numpy(),weights(w))
        np.testing.assert_allclose(ga.numpy(),expected,rtol=1e-12,atol=1e-12)
        checks.append({'test':'bilinear_forward_and_independent_transpose','shape':[h,w],'passed':True})
    x=torch.randn(2,3,8,9,dtype=torch.float64,requires_grad=True)
    boxes=torch.tensor([[0,0,0,64,64],[0,3,5,68,58],[1,8,0,48,64]],dtype=torch.float64)
    a=roi_align(x,boxes,(3,3),1/8,2,True);b=native_roi_align(x,boxes,(3,3),1/8,2,True)
    upstream=torch.randn_like(a);ga=torch.autograd.grad(a,x,upstream)[0];gb=torch.autograd.grad(b,x,upstream)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(ga,gb,rtol=0,atol=0)
    # Coordinate finite differences check the adjoint independently of backward.
    for index in ((0,0,0,0),(0,2,4,5),(1,1,3,4),(1,2,7,8)):
        hi=x.detach().clone();lo=x.detach().clone();hi[index]+=1e-5;lo[index]-=1e-5
        diff=((native_roi_align(hi,boxes,(3,3),1/8,2,True)-native_roi_align(lo,boxes,(3,3),1/8,2,True))*upstream).sum()/2e-5
        torch.testing.assert_close(diff,ga[index],rtol=1e-7,atol=1e-8)
    checks.append({'test':'ROI_native_parity_and_finite_difference','passed':True})
    for wrong in ((5,6),(6,9)):
        try:bilinear2x(torch.zeros(1,1,3,4),size=wrong,mode='bilinear',align_corners=False)
        except ValueError:pass
        else:raise AssertionError('Unsupported resize accepted')
    checks.append({'test':'unsupported_geometry_rejected','passed':True})
    return checks


def check_cuda():
    torch.set_num_threads(2);torch.manual_seed(713)
    report=[]
    for kind in ('bilinear','ROIAlign'):
        shape=(1,192,32,48) if kind=='bilinear' else (3,192,16,16)
        x=torch.randn(*shape,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        boxes=torch.tensor([[0,0,0,128,128],[1,3,5,100,91],[2,12,0,91,128]],device='cuda',dtype=torch.float32)
        def forward(stable):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if kind=='bilinear':
                    if stable:return bilinear2x(x,size=(64,96),mode='bilinear',align_corners=False)
                    return torch.ops.aten.upsample_bilinear2d.default(x,[64,96],False,None,None)
                if stable:return roi_align(x,boxes,(3,3),1/8,2,True)
                return torch.ops.torchvision.roi_align(x,boxes,1/8,3,3,2,True)
        native=[];stable=[];upstream=None;first=None;warning_text=[]
        # Confirm the fixture's native reference really is the original public
        # API with determinism disabled, rather than its deterministic fallback.
        torch.use_deterministic_algorithms(False)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            if kind=='bilinear':original=native_functional.interpolate(x,size=(64,96),mode='bilinear',align_corners=False)
            else:original=native_roi_align(x,boxes,(3,3),1/8,2,True)
            torch.testing.assert_close(forward(False),original,rtol=0,atol=0)
        for use_stable,target in ((False,native),(True,stable)):
            torch.use_deterministic_algorithms(True,warn_only=not use_stable)
            for _ in range(3):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');y=forward(use_stable)
                    if upstream is None:upstream=torch.randn_like(y)
                    if first is None:first=original.detach().clone()
                    torch.testing.assert_close(y,first,rtol=0,atol=0)
                    target.append(torch.autograd.grad(y,x,upstream)[0].detach().float().cpu())
                    warning_text.extend(str(v.message) for v in caught)
        for value in stable[1:]:torch.testing.assert_close(value,stable[0],rtol=0,atol=0)
        # Higher-precision native CPU gradient is a reference for the low-precision adjoint.
        cpu_x=x.detach().float().cpu().requires_grad_(True)
        if kind=='bilinear':cpu_y=native_functional.interpolate(cpu_x,size=(64,96),mode='bilinear',align_corners=False)
        else:cpu_y=native_roi_align(cpu_x,boxes.cpu(),(3,3),1/8,2,True)
        reference=torch.autograd.grad(cpu_y,cpu_x,upstream.float().cpu())[0].to(torch.bfloat16).float()
        # One BF16 ULP around rounding boundaries is permissible against FP32 sum order;
        # exact native-forward and repeated-stable-gradient checks above remain strict.
        torch.testing.assert_close(stable[0],reference,rtol=.008,atol=1e-5)
        report.append({'operator':kind,'native_warnings':sorted(set(warning_text)),
                       'native_repeat_max_abs':max(float((v-native[0]).abs().max()) for v in native[1:]),
                       'stable_repeat_max_abs':0.,'native_forward_exact':True,
                       'stable_vs_fp32_reference_max_abs':float((stable[0]-reference).abs().max()),
                       'native_vs_stable_max_abs':float((native[0]-stable[0]).abs().max())})
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--output',required=True);p.add_argument('--cuda',action='store_true')
    args=p.parse_args();out=pathlib.Path(args.output)
    if (out/'report.json').exists():raise ValueError('Exclusive operator check')
    report={'state':'running','optimizer_calls':0,'persistent_research_steps':0}
    try:
        report['cpu_checks']=check_cpu()
        if args.cuda:report['cuda_checks']=check_cuda()
        report.update(state='completed',all_checks_passed=True)
    except Exception as exc:report.update(state='failed',all_checks_passed=False,error=repr(exc));raise
    finally:(out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
