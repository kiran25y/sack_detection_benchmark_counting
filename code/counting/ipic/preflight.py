"""Bounded engineering profile using source-training points; not an accuracy run."""
import argparse,copy,gc,hashlib,json,pathlib,time
import numpy as np
import torch
from .model import IPICNet
from .transforms import decode_oriented,resize_pad,reference_crop
from .matching import capacity_check
from .training import make_optimizer,optimizer_update

WEIGHT_SHA='14f3164e3ea6ac32ab3f574f528ce817696c9176fad4221e0a77a905a7360595'

def prepare(row,root,device,long_side=1024):
    path=root/row['image'].removeprefix('/research/')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:raise ValueError('Source hash mismatch')
    im,exif,original=decode_oriented(path)
    if not np.array_equal(exif,np.eye(3)):raise ValueError('FSC native geometry requires explicit EXIF audit')
    rgb,valid,g=resize_pad(im,long_side)
    ref=[reference_crop(im,b) for b in row['exemplars']]
    x=torch.from_numpy(rgb.copy())[None].to(device);mask=torch.from_numpy(valid).to(device)
    pts=torch.tensor(g.points(row['points']),dtype=torch.float32,device=device)
    return {'a':x,'b':x.clone(),'valid_a':mask,'valid_b':mask.clone(),
            'points_a':pts,'points_b':pts.clone(),'all_products':False,
            'references':torch.from_numpy(np.stack([r[0] for r in ref])).to(device),
            'reference_boxes':torch.from_numpy(np.stack([r[1] for r in ref])).to(device)},g

def tensor_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=pathlib.Path(a.root);out=pathlib.Path(a.output);out.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    rows=json.loads((root/'manifests/fsc147_train_groupclean.json').read_text())['queries']
    byid={r['query_id']:r for r in rows};chosen=[byid['7.jpg'],max(rows,key=lambda r:r['gt_count']),max(rows,key=lambda r:r['width']/r['height'])]
    if any(r['split']!='train' for r in chosen):raise ValueError('Training partition only')
    report={'state':'running','purpose':'Engineering only: two full-resolution disposable updates per architecture; no persistent trained model',
            'chosen':[{'query_id':r['query_id'],'count':r['gt_count'],'image_sha256':r['image_sha256']} for r in chosen],
            'optimizer_calls':0,'persistent_steps':0,'arms':[],'weight_sha256':WEIGHT_SHA}
    def save():
        p=out/'report.tmp';p.write_text(json.dumps(report,indent=2));p.replace(out/'report.json')
    save();weight=root/'research-direction-20261012/ipic_assets_v1_20261030/convnext_tiny_1k_224_ema.pth'
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        # Same small real source query: CPU/CUDA FP32 feature/head compatibility.
        for detail in (False,True):
            torch.manual_seed(1729);model=IPICNet(detail);loaded=model.context.load_official(weight,WEIGHT_SHA)
            cpu,_=prepare(chosen[0],root,'cpu',256);model.eval()
            with torch.no_grad():
                refs=model.encode_references(cpu['references'],cpu['reference_boxes']);base=model(cpu['a'],cpu['valid_a'],refs)
            model=model.cuda();cuda,_=prepare(chosen[0],root,'cuda',256)
            with torch.no_grad():
                refs=model.encode_references(cuda['references'],cuda['reference_boxes']);other=model(cuda['a'],cuda['valid_a'],refs)
            diffs={}
            for key in ('logits','points','probabilities'):
                v=other[key].cpu();torch.testing.assert_close(v,base[key],rtol=.003,atol=.01);diffs[key]=float((v-base[key]).abs().max())
            del base,other,cpu,cuda,refs;model.train();optimizer=make_optimizer(model)
            batches=[];capacities=[]
            for row in chosen:
                b,g=prepare(row,root,'cuda');batches.append(b)
                ph,pw=b['valid_a'].shape
                yy,xx=torch.meshgrid(torch.arange(ph//4,device='cuda')*4+2,torch.arange(pw//4,device='cuda')*4+2,indexing='ij')
                anchors=torch.stack((xx.flatten(),yy.flatten()),1).float()
                va=b['valid_a'][yy.flatten(),xx.flatten()]
                capacities.append(capacity_check(anchors,b['points_a'],va))
            # Populates authentic nonzero Adam moments for the serialization test.
            micro=[batches[i%len(batches)] for i in range(8)]
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
            first=optimizer_update(model,optimizer,micro,'A');torch.cuda.synchronize();elapsed=time.perf_counter()-t
            report['optimizer_calls']+=1
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite updated parameters')
            snap=out/('preupdate_'+('B1' if detail else 'B0')+'.pt')
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all()},snap)
            identity=tensor_digest(model);saved=torch.load(snap,map_location='cpu',weights_only=True)
            for k,v in saved['model'].items():torch.testing.assert_close(v,model.state_dict()[k].cpu(),rtol=0,atol=0)
            # Restore in place, then require every optimizer tensor and RNG to equal the saved state.
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            if tensor_digest(model)!=identity:raise RuntimeError('Serialization changed model')
            restored=optimizer.state_dict()
            for key,value in saved['optimizer']['state'].items():
                for k,v in value.items():
                    if torch.is_tensor(v):torch.testing.assert_close(restored['state'][key][k].cpu(),v,rtol=0,atol=0)
                    elif restored['state'][key][k]!=v:raise RuntimeError('Optimizer scalar changed')
            if not torch.equal(torch.get_rng_state(),saved['cpu_rng']):raise RuntimeError('CPU RNG mismatch')
            if not all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng'])):raise RuntimeError('CUDA RNG mismatch')
            del saved
            torch.cuda.synchronize();t=time.perf_counter();second=optimizer_update(model,optimizer,micro,'A');torch.cuda.synchronize();second_time=time.perf_counter()-t
            report['optimizer_calls']+=1
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite recovered step')
            for st in optimizer.state.values():
                if not all(torch.isfinite(v).all() for v in st.values() if torch.is_tensor(v)):raise FloatingPointError('Nonfinite moments')
            report['arms'].append({'detail':detail,'parameters':sum(p.numel() for p in model.parameters()),'pretrained':loaded,
              'fp32_cpu_cuda_max_abs':diffs,'capacity':capacities,'update_seconds':[elapsed,second_time],
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
              'preupdate_model_optimizer_rng_exact':True,'finite_recovered_update':True,
              'preupdate_sha256':hashlib.sha256(snap.read_bytes()).hexdigest(),'updates':[first,second]});save()
            del model,optimizer,batches,micro,b,restored;gc.collect();torch.cuda.empty_cache()
        report['state']='completed';report['all_checks_passed']=True;save()
    except Exception as exc:
        report.update(state='failed',error=repr(exc),all_checks_passed=False);save();raise

if __name__=='__main__':main()
