"""Read-only count/point inspection of the completed engineering checkpoint.

This does not choose a deployment threshold or perform optimizer updates.
It checks the specification's distinction between soft mass and point counting.
"""
import argparse,hashlib,json,pathlib,time
import numpy as np
import torch
from .model import IPICNet
from .preflight import prepare
from .losses import supervised_loss
from .matching import sparse_match
from .evaluation import metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--arm',choices=['B0','B1'],default='B0');a=ap.parse_args()
    root=pathlib.Path(a.root);out=pathlib.Path(a.output);out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    source=root/'research-direction-20261012/ipic_checks_20261030/tiny_v1'
    status=json.loads((source/'status.json').read_text());arm=next(x for x in status['arms'] if x['name']==a.arm)
    if arm['saved_updates']!=200 or 'final_checkpoint_sha256' not in arm:raise RuntimeError('Completed arm required')
    path=source/(a.arm+'_engineering_resume.pt');digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=arm['final_checkpoint_sha256']:raise RuntimeError('Own engineering checkpoint changed')
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    if ckpt['engineering_update']!=200 or ckpt['purpose']!='disposable engineering fit; never use as research initialization':raise RuntimeError('Wrong checkpoint')
    model=IPICNet(a.arm=='B1');model.load_state_dict(ckpt['model'],strict=True);model.eval();del ckpt
    rows=json.loads((root/'manifests/fsc147_train_groupclean.json').read_text())['queries'];byid={x['query_id']:x for x in rows}
    result={'purpose':'CPU FP32 frozen engineering count/point inspection; not validation or threshold selection',
            'arm':a.arm,'checkpoint_sha256':digest,'optimizer_steps':0,'gpu_used':False,'queries':[]}
    started=time.monotonic()
    with torch.inference_mode():
        for name in ('7.jpg','9.jpg','81.jpg','2347.jpg'):
            b,_=prepare(byid[name],root,'cpu',1024)
            proto=model.encode_references(b['references'],b['reference_boxes']);v=model(b['a'],b['valid_a'],proto)
            loss=supervised_loss(v,b['points_a']);ci,gi=sparse_match(v,b['points_a'])
            p=v['probabilities'][v['valid']];matched=v['probabilities'][ci]
            item={'query_id':name,'truth':len(b['points_a']),'soft_count':float(loss['soft_count']),
                  'loss_components':{k:float(loss[k]) for k in ('total','focal','point','count')},
                  'valid_probability_quantiles':torch.quantile(p,torch.tensor([0.,.5,.9,.99,1.])).tolist(),
                  'matched_probability_quantiles':torch.quantile(matched,torch.tensor([0.,.5,.9,1.])).tolist(),
                  'thresholds':[]}
            h,w=b['a'].shape[-2:]
            for integer in range(10,91,5):
                threshold=integer/100;pred=model.predict(v,threshold)
                scores=metrics(pred['points'].numpy(),b['points_a'].numpy(),w,h)
                item['thresholds'].append({'threshold':threshold,**scores})
            result['queries'].append(item)
            (out/'inspection.json').write_text(json.dumps(result,indent=2))
    result['threshold_curve']=[{'threshold':integer/100,
       'MAE':float(np.mean([q['thresholds'][j]['AE'] for q in result['queries']])),
       'counts':[q['thresholds'][j]['prediction'] for q in result['queries']],
       'macro_F1_at_8':float(np.mean([q['thresholds'][j]['point_8']['F1'] for q in result['queries']]))}
       for j,integer in enumerate(range(10,91,5))]
    result['all_prescribed_thresholds_empty']=all(x['prediction']==0 for q in result['queries'] for x in q['thresholds'])
    result['elapsed_seconds']=time.monotonic()-started
    result['state']='completed'
    (out/'inspection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
