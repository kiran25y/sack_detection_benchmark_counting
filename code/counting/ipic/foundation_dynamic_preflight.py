"""Native-chart trainability replay and three disposable amended public updates."""
from .stable_resampling import install

import argparse,gc,json,pathlib,time
import numpy as np
import torch
from .assignment_cudnn_preflight import state_digest
from .evaluation import metrics
from .foundation import sha,write_json
from .foundation_dynamic import MANIFEST_SHA
from .model import IPICNet
from .preflight import WEIGHT_SHA,prepare
from .point_fit_diagnostic import point_only_pair
from .public_data import prepare_pair
from .matching import capacity_check
from .training import make_optimizer,optimizer_update,set_learning_rate

def native_chart_check(root,rows):
    source=root/'research-direction-20261012/ipic_checks_20261030/assignment_continuation_stable_v1_20261031/dynamic_resume.pt'
    if sha(source)!='23546b0d6d602026db6d33ce54202eb9c3bb74f9ad6e35fa6e21b1fc6ee27801':raise ValueError('Wrong validated diagnostic checkpoint')
    ck=torch.load(source,map_location='cpu',weights_only=True)
    if ck['engineering_update']!=400:raise ValueError('Wrong diagnostic duration')
    model=IPICNet(False);model.load_state_dict(ck['model'],strict=True);del ck;model.eval()
    native={r['query_id']:r for r in json.loads((root/'manifests/fsc147_train_groupclean.json').read_text())['queries']}
    public={r['query_id']:r for r in rows if r['dataset']=='FSC147'};result=[]
    with torch.inference_mode():
        for q in ('7.jpg','9.jpg','81.jpg','2347.jpg'):
            batch,g=prepare(native[q],root,'cpu',1024);proto=model.encode_references(batch['references'],batch['reference_boxes'])
            v=model(batch['a'],batch['valid_a'],proto);h,w=batch['a'].shape[-2:]
            # Explicit native continuous-edge to raster-center chart; labels stay unchanged.
            truth=g.points(public[q]['points_pixel_centers']).astype(np.float32)
            if not np.allclose(g.points(truth,inverse=True)+.5,public[q]['native_points'],atol=.001,rtol=0):raise ValueError('Public chart round trip failed')
            m=metrics(model.predict(v,.5)['points'].numpy(),truth,w,h)
            ok=(m['AE']<=2 if m['truth']<150 else m['AE']/m['truth']<=.1) and m['point_8']['F1']>=.7
            result.append({'query_id':q,'passed':ok,**m})
    del model;gc.collect()
    if not all(x['passed'] for x in result):raise ValueError('Native public-coordinate trainability check failed')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    root=pathlib.Path(args.root);out=pathlib.Path(args.output)
    if (out/'report.json').exists():raise ValueError('Exclusive guard; no automatic repeat')
    torch.set_num_threads(2);runtime=install()
    report={'state':'running','optimizer_calls':0,'maximum_optimizer_calls':3,'persistent_research_steps':0,'count_loss_coefficient':0,
            'assignment':'dynamic','runtime':runtime,'manifest_sha256':MANIFEST_SHA,'engineering_guard_passed':False,'native_chart_trainability_passed':False}
    def save():write_json(out/'report.json',report)
    save()
    try:
        manifest=root/'research-direction-20261012/ipic_checks_20261030/public_v1/public_train_manifest.jsonl'
        if sha(manifest)!=MANIFEST_SHA:raise ValueError('Public manifest changed')
        rows=[json.loads(s) for s in manifest.read_text().splitlines()]
        report['native_chart_trainability']=native_chart_check(root,rows);report['native_chart_trainability_passed']=True;save()
        fsc={r['query_id']:r for r in rows if r['dataset']=='FSC147'};d2s=[r for r in rows if r['dataset']=='D2S']
        boundary=next(r for r in fsc.values() if any(x>=r['width'] or y>=r['height'] for x,y in r['native_points']))
        chosen=[fsc['7.jpg'],fsc['81.jpg'],max(fsc.values(),key=lambda r:r['count']),boundary,d2s[0],d2s[len(d2s)//2],d2s[-1],max(d2s,key=lambda r:r['count_all_products'])]
        torch.manual_seed(1729);np.random.seed(1729);model=IPICNet(False)
        model.context.load_official(root/'research-direction-20261012/ipic_assets_v1_20261030/convnext_tiny_1k_224_ema.pth',WEIGHT_SHA)
        model.cuda().train();opt=make_optimizer(model);micro=[];report['queries']=[]
        for i,row in enumerate(chosen):
            draw={'query_id':row['query_id'],'dataset':row['dataset'],'all_products':i==7,'seed':1729+i,'row_index':rows.index(row)}
            batch,ledger=prepare_pair(row,draw,root,'cuda');checks=[]
            for key in ('a','b'):
                h,w=batch[key].shape[-2:];yy,xx=torch.meshgrid(torch.arange(h//4,device='cuda')*4+2,torch.arange(w//4,device='cuda')*4+2,indexing='ij')
                anchors=torch.stack((xx.flatten(),yy.flatten()),1).float();valid=batch['valid_'+key][yy.flatten(),xx.flatten()]
                if not torch.isfinite(batch[key]).all():raise ValueError('Nonfinite augmented image')
                checks.append(capacity_check(anchors,batch['points_'+key],valid))
            micro.append(batch);report['queries'].append({'query_id':row['query_id'],'dataset':row['dataset'],'all_products':draw['all_products'],
                'input_hash':row['image_sha256'],'counts':[len(batch['points_a']),len(batch['points_b'])],'transforms':ledger,'capacities':checks})
        scales={q['transforms'][v]['long_side'] for q in report['queries'] for v in ('view_a','view_b')}
        if scales!={768,896,1024}:raise ValueError('Guard lacks full prescribed resize coverage')
        report['resize_coverage']=sorted(scales);report['updates']=[];save();torch.cuda.reset_peak_memory_stats()
        def update(step):
            set_learning_rate(opt,step,2000,100);torch.cuda.synchronize();t=time.monotonic()
            m=optimizer_update(model,opt,micro,'A',loss_function=point_only_pair);torch.cuda.synchronize()
            if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite model')
            if not all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)):raise ValueError('Nonfinite Adam')
            report['optimizer_calls']+=1;report['updates'].append({'schedule_index':step,'seconds':time.monotonic()-t,**m});save()
        update(0)
        snap={'model':model.state_dict(),'optimizer':opt.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        initial={k:state_digest(v) for k,v in snap.items()};torch.save(snap,out/'guard_resume.pt');del snap
        update(1);expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};moments={str(i)+':'+k:v.detach().cpu().clone() for i,s in opt.state_dict()['state'].items() for k,v in s.items() if torch.is_tensor(v)}
        ck=torch.load(out/'guard_resume.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);del ck
        actual={'model':model.state_dict(),'optimizer':opt.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        if {k:state_digest(v) for k,v in actual.items()}!=initial:raise ValueError('Exact pre-update recovery failed')
        del actual;update(1);maximum={'model':0.,'moments':0.}
        for k,v in model.state_dict().items():
            x=v.detach().cpu();torch.testing.assert_close(x,expected[k],rtol=.003,atol=1e-5);maximum['model']=max(maximum['model'],float((x-expected[k]).abs().max()))
        for i,s in opt.state_dict()['state'].items():
            for k,v in s.items():
                if torch.is_tensor(v):
                    x=v.detach().cpu();y=moments[str(i)+':'+k];torch.testing.assert_close(x,y,rtol=.003,atol=1e-5);maximum['moments']=max(maximum['moments'],float((x-y).abs().max()))
        report.update(state='completed',engineering_guard_passed=True,preupdate_exact_recovery=True,poststep_max_abs=maximum,
            poststep_tolerances={'rtol':.003,'atol':1e-5},peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved(),
            finite_model_optimizer=True,research_training_started=False,pretrained_weights_sha256=WEIGHT_SHA)
        save()
    except Exception as exc:report.update(state='failed',error=repr(exc));save();raise

if __name__=='__main__':main()
