"""The prescribed bounded real-set trainability check; no held-out evaluation."""
import argparse,gc,hashlib,json,pathlib,time
import numpy as np
import torch
from .preflight import prepare,WEIGHT_SHA
from .model import IPICNet
from .losses import supervised_loss
from .training import make_optimizer,optimizer_update

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=pathlib.Path(a.root);out=pathlib.Path(a.output);out.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    rows=json.loads((root/'manifests/fsc147_train_groupclean.json').read_text())['queries'];byid={r['query_id']:r for r in rows}
    chosen=[byid[n] for n in ('7.jpg','9.jpg','81.jpg','2347.jpg')]
    if any(r['split']!='train' for r in chosen):raise RuntimeError('Fitting partition only')
    state={'state':'starting','purpose':'engineering tiny-set fit, not a research arm or validation score',
           'seed':1729,'updates_per_architecture':200,'total_update_cap':400,'durable_engineering_updates':0,
           'common_learning_rates':[2e-5,2e-4],'schedule':'constant for bounded overfit check only',
           'query_long_side':1024,'microbatch_pairs':1,'accumulation_pairs':8,
           'queries':[{'id':r['query_id'],'count':r['gt_count'],'sha256':r['image_sha256']} for r in chosen],
           'prospective_engineering_gate':'finite 200-update trajectory and at least 50% reduction of the mean original supervised fitting loss for BOTH architectures',
           'held_out_queries_used':0,'research_optimizer_steps':0,'arms':[]}
    def save():
        p=out/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(out/'status.json')
    def assess(model,batches):
        model.eval();values=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for b in batches:
                proto=model.encode_references(b['references'],b['reference_boxes']);v=model(b['a'],b['valid_a'],proto)
                # Loss reductions are outside autocast in the actual training updater.
                with torch.autocast('cuda',enabled=False):loss=supervised_loss(v,b['points_a'])
                values.append({'loss':float(loss['total']),'soft_count':float(loss['soft_count']),
                               'integer_count_at_0_5':model.predict(v,.5)['visible_count'],'truth':len(b['points_a'])})
        model.train();return {'queries':values,'mean_loss':float(np.mean([x['loss'] for x in values]))}
    try:
        batches=[prepare(r,root,'cuda',1024)[0] for r in chosen];micro=[batches[i%4] for i in range(8)]
        weight=root/'research-direction-20261012/ipic_assets_v1_20261030/convnext_tiny_1k_224_ema.pth'
        for detail in (False,True):
            torch.manual_seed(1729);model=IPICNet(detail);model.context.load_official(weight,WEIGHT_SHA);model.cuda().train();optimizer=make_optimizer(model)
            arm={'name':'B1' if detail else 'B0','initial':assess(model,batches),'saved_updates':0,'curves':[]}
            state['arms'].append(arm);state.update(state='engineering_training',current_arm=arm['name']);save()
            for step in range(200):
                torch.cuda.synchronize();started=time.perf_counter();metrics=optimizer_update(model,optimizer,micro,'A');torch.cuda.synchronize()
                row={'update':step+1,'seconds':time.perf_counter()-started,'loss':float(np.mean([r['total'] for r in metrics['microbatch_losses']])),
                     'pre_clip_norm':metrics['pre_clip_norm']};arm['curves'].append(row)
                if (step+1)%25==0:row['fitting_check']=assess(model,batches)
                if (step+1)%10==0 or step==199:
                    ckpt=out/(arm['name']+'_engineering_resume.pt');tmp=ckpt.with_suffix('.tmp')
                    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':None,'engineering_update':step+1,
                                'seed':1729,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                                'purpose':'disposable engineering fit; never use as research initialization'},tmp);tmp.replace(ckpt)
                    arm['saved_updates']=step+1;state['durable_engineering_updates']=sum(x['saved_updates'] for x in state['arms']);save()
            arm['final']=assess(model,batches)
            arm['finite_model_optimizer']=bool(all(torch.isfinite(p).all() for p in model.parameters()) and
              all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)))
            arm['relative_fitting_loss_reduction']=1-arm['final']['mean_loss']/arm['initial']['mean_loss']
            arm['engineering_gate_passed']=arm['finite_model_optimizer'] and arm['relative_fitting_loss_reduction']>=.5
            arm['final_checkpoint_sha256']=hashlib.sha256(ckpt.read_bytes()).hexdigest();save()
            del model,optimizer;gc.collect();torch.cuda.empty_cache()
        state.update(state='completed',engineering_gate_passed=all(a['engineering_gate_passed'] for a in state['arms']),full_training_started=False);save()
    except Exception as exc:state.update(state='failed',error=repr(exc),full_training_started=False);save();raise

if __name__=='__main__':main()
