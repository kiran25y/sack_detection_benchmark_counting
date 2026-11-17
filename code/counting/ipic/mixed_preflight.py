"""Two disposable public-mixture updates through the production data adapter."""
import argparse,hashlib,json,pathlib,time
import numpy as np
import torch
from .model import IPICNet
from .preflight import WEIGHT_SHA
from .public_data import prepare_pair
from .matching import capacity_check
from .training import make_optimizer,optimizer_update,set_learning_rate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=pathlib.Path(a.root);out=pathlib.Path(a.output);out.mkdir(exist_ok=True)
    path=root/'research-direction-20261012/ipic_checks_20261030/public_v1/public_train_manifest.jsonl'
    if hashlib.sha256(path.read_bytes()).hexdigest()!='8a70c247df7f304b46cd47b215de23e71c83a1aaeff58def3e5b8522b35e3721':raise ValueError('Manifest changed')
    rows=[json.loads(s) for s in path.read_text().splitlines()];fsc={r['query_id']:r for r in rows if r['dataset']=='FSC147'}
    d2s=[r for r in rows if r['dataset']=='D2S']
    boundary=next(r for r in fsc.values() if any(x>=r['width'] or y>=r['height'] for x,y in r['native_points']))
    chosen=[fsc['7.jpg'],fsc['81.jpg'],fsc['2347.jpg'],boundary,d2s[0],d2s[len(d2s)//2],d2s[-1],max(d2s,key=lambda r:r['count_all_products'])]
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    report={'state':'running','optimizer_calls':0,'persistent_research_steps':0,'architecture':'B0',
      'purpose':'Production augmented public pairs, engineering only; no model accuracy',
      'source_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'queries':[],'updates':[]}
    def save():
        tmp=out/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(out/'report.json')
    try:
        model=IPICNet(False);model.context.load_official(root/'research-direction-20261012/ipic_assets_v1_20261030/convnext_tiny_1k_224_ema.pth',WEIGHT_SHA);model.cuda().train()
        opt=make_optimizer(model);micro=[]
        for i,row in enumerate(chosen):
            draw={'query_id':row['query_id'],'dataset':row['dataset'],'all_products':i==7,'seed':1729+i,'row_index':rows.index(row)}
            batch,record=prepare_pair(row,draw,root,'cuda');micro.append(batch);checks=[]
            for key in ('a','b'):
                h,w=batch[key].shape[-2:];yy,xx=torch.meshgrid(torch.arange(h//4,device='cuda')*4+2,torch.arange(w//4,device='cuda')*4+2,indexing='ij')
                anchors=torch.stack((xx.flatten(),yy.flatten()),1).float();valid=batch['valid_'+key][yy.flatten(),xx.flatten()]
                if not torch.isfinite(batch[key]).all():raise ValueError('Nonfinite transformed image')
                checks.append(capacity_check(anchors,batch['points_'+key],valid))
            report['queries'].append({'query_id':row['query_id'],'dataset':row['dataset'],'input_hash':row['image_sha256'],
              'all_products':draw['all_products'],'counts':[len(batch['points_a']),len(batch['points_b'])],'transforms':record,'capacities':checks})
        save();torch.cuda.reset_peak_memory_stats()
        for step in range(2):
            set_learning_rate(opt,step,2000,100);torch.cuda.synchronize();t=time.monotonic();metric=optimizer_update(model,opt,micro,'A');torch.cuda.synchronize()
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite model')
            if not all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)):raise FloatingPointError('Nonfinite Adam state')
            report['optimizer_calls']+=1;report['updates'].append({'update':step+1,'seconds':time.monotonic()-t,**metric});save()
        report.update(state='completed',engineering_guard_passed=True,peak_gpu_allocated=torch.cuda.max_memory_allocated(),
          peak_gpu_reserved=torch.cuda.max_memory_reserved(),finite_model_optimizer=True,full_training_started=False)
        save()
    except Exception as exc:report.update(state='failed',error=repr(exc),engineering_guard_passed=False);save();raise

if __name__=='__main__':main()
