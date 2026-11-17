"""Owner-authorized single 200-update no-soft-count engineering diagnostic.

The primary IPIC loss is unchanged. This candidate cannot auto-promote a fit.
"""
import argparse,hashlib,json,pathlib,time
import numpy as np
import torch
from .model import IPICNet
from .preflight import prepare,WEIGHT_SHA,tensor_digest
from .losses import pair_loss
from .training import make_optimizer,optimizer_update
from .evaluation import metrics

def point_only_pair(a,b,truth_a,truth_b,arm,masks=None,eligible=None,region_valid=None):
    if arm!='A' or masks is not None:raise ValueError('Diagnostic is ordinary B0 point supervision only')
    result=pair_loss(a,b,truth_a,truth_b,arm,masks,eligible,region_valid)
    la,lb=result['view_a'],result['view_b']
    result['total']=.5*(la['focal']+5*la['point']+lb['focal']+5*lb['point'])
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=pathlib.Path(a.root);out=pathlib.Path(a.output);out.mkdir(exist_ok=True)
    if (out/'status.json').exists():raise ValueError('Exclusive one-shot diagnostic; no automatic resume')
    base=root/'research-direction-20261012/ipic_checks_20261030/tiny_v1'
    control=json.loads((base/'status.json').read_text());b0=next(x for x in control['arms'] if x['name']=='B0')
    if control['state']!='completed' or b0['saved_updates']!=200:raise ValueError('Finish original engineering run first')
    if b0['final_checkpoint_sha256']!='36298a418342eabc9d2828e24a93d07c1a119fedf1190c4f9922e98fcf474d3d':raise ValueError('Control identity changed')
    frozen=json.loads((base/'source_hashes.json').read_text());source_identity={}
    for filename in ('model.py','preflight.py','losses.py','matching.py','transforms.py'):
        p=pathlib.Path(__file__).parent/filename;digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest!=frozen['counting/ipic/'+filename]:raise ValueError('Matched implementation changed: '+filename)
        source_identity[filename]=digest
    manifest=root/'manifests/fsc147_train_groupclean.json';rows=json.loads(manifest.read_text())['queries'];byid={r['query_id']:r for r in rows}
    chosen=[byid[n] for n in ('7.jpg','9.jpg','81.jpg','2347.jpg')]
    if any(r['split']!='train' for r in chosen):raise RuntimeError('Training partition only')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    batches=[prepare(r,root,'cuda',1024)[0] for r in chosen];micro=[batches[i%4] for i in range(8)]
    torch.manual_seed(1729);model=IPICNet(False);model.context.load_official(root/'research-direction-20261012/ipic_assets_v1_20261030/convnext_tiny_1k_224_ema.pth',WEIGHT_SHA);model.cuda().train();opt=make_optimizer(model)
    plan={'purpose':'owner-authorized one-candidate trainability diagnostic, not scientific screen',
      'authorization':'October30 owner reply: Run the matched diagnostic (recommended)',
      'updates':200,'seed':1729,'architecture':'B0','loss':'focal + 5*point, no soft-count term',
      'only_loss_intervention':'soft-count coefficient0 rather than0.10','learning_rates':[2e-5,2e-4],
      'schedule':'constant, exactly the retained engineering control','batch_order':[r['query_id'] for r in chosen]*2,
      'same_frozen_modules':source_identity,'initial_model_tensor_digest':tensor_digest(model),
      'control_checkpoint_sha256':b0['final_checkpoint_sha256'],'original_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
      'gate':{'primary_threshold':.5,'sparse_each_abs_error_max':2,'dense_extreme_each_relative_error_max':.1,'each_point_F1_at8_min':.7},
      'full_training_auto_promotion':False,'extra_seeds':False,'heldout_queries_used':0}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));ph=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest()
    state={'state':'engineering_training','durable_optimizer_steps':0,'total_optimizer_steps':200,'research_optimizer_steps':0,
      'plan_sha256':ph,'queries':[{'id':r['query_id'],'count':r['gt_count'],'sha256':r['image_sha256']} for r in chosen],
      'curves':[],'full_training_started':False}
    def save():
        p=out/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(out/'status.json')
    save()
    try:
        for step in range(200):
            torch.cuda.synchronize();t=time.monotonic();m=optimizer_update(model,opt,micro,'A',loss_function=point_only_pair);torch.cuda.synchronize()
            row={'update':step+1,'seconds':time.monotonic()-t,'mean_candidate_loss':float(np.mean([r['total'] for r in m['microbatch_losses']])),
              'pre_clip_norm':m['pre_clip_norm'],'microbatch_losses':m['microbatch_losses']}
            state['curves'].append(row)
            if (step+1)%10==0:
                p=out/'resume.tmp';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'engineering_update':step+1,
                  'seed':1729,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'plan_sha256':ph,
                  'purpose':'disposable no-soft-count trainability diagnostic; not research initialization'},p);p.replace(out/'resume.pt')
                state['durable_optimizer_steps']=step+1;save()
        finite=all(torch.isfinite(p).all() for p in model.parameters()) and all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
        if not finite:raise FloatingPointError('Nonfinite final model or optimizer')
        state.update(state='evaluating_fitted_queries',finite_model_optimizer=True);save()
        del opt,micro,batches;model.cpu().eval();torch.cuda.empty_cache();predictions=[]
        with torch.inference_mode():
            for row in chosen:
                b,_=prepare(row,root,'cpu',1024);proto=model.encode_references(b['references'],b['reference_boxes']);v=model(b['a'],b['valid_a'],proto)
                h,w=b['a'].shape[-2:];scores=[]
                for n in range(10,91,5):
                    p=model.predict(v,n/100);scores.append({'threshold':n/100,**metrics(p['points'].numpy(),b['points_a'].numpy(),w,h)})
                predictions.append({'query_id':row['query_id'],'truth':len(b['points_a']),'soft_count':float(v['probabilities'][v['valid']].sum()),
                  'thresholds':scores,'primary':next(s for s in scores if s['threshold']==.5)})
        checks=[]
        for row in predictions:
            s=row['primary'];checks.append({'query_id':row['query_id'],'count_pass':s['AE']<=2 if row['truth']<150 else s['AE']/row['truth']<=.1,
              'point_F1_pass':s['point_8']['F1']>=.7,'count':s['prediction'],'truth':s['truth'],'AE':s['AE'],'F1_at8':s['point_8']['F1']})
        report={'state':'completed','engineering_gate_passed':all(c['count_pass'] and c['point_F1_pass'] for c in checks),
          'checks':checks,'queries':predictions,'optimizer_calls':200,'persistent_research_steps':0,'full_training_started':False,
          'checkpoint_sha256':hashlib.sha256((out/'resume.pt').read_bytes()).hexdigest(),
          'claim_limit':'Seen four-image fitting only. No validation accuracy, scientific mechanism gain, novelty or automatic promotion.'}
        (out/'report.json').write_text(json.dumps(report,indent=2));state.update(state='completed',engineering_gate_passed=report['engineering_gate_passed'],checkpoint_sha256=report['checkpoint_sha256']);save()
    except Exception as exc:state.update(state='failed',error=repr(exc));save();raise

if __name__=='__main__':main()
