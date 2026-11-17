"""Zero-update per-term derivatives at the saved B0 engineering checkpoint.

Local loss/gradient evidence only; not an Adam trajectory or a repair experiment.
"""
import argparse,hashlib,json,pathlib,time
import torch
from .model import IPICNet
from .preflight import prepare,tensor_digest
from .losses import supervised_loss
from .matching import sparse_match

def dot(a,b):return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b))
def norm(a):return max(dot(a,a),0.)**.5

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=pathlib.Path(a.root);out=pathlib.Path(a.output);out.mkdir(exist_ok=True);torch.set_num_threads(2)
    source=root/'research-direction-20261012/ipic_checks_20261030/tiny_v1';status=json.loads((source/'status.json').read_text())
    arm=next(x for x in status['arms'] if x['name']=='B0');path=source/'B0_engineering_resume.pt'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if arm['saved_updates']!=200 or digest!=arm['final_checkpoint_sha256']:raise RuntimeError('Completed original B0 required')
    raw=torch.load(path,map_location='cpu',weights_only=True);model=IPICNet(False);model.load_state_dict(raw['model'],strict=True);del raw
    model.eval();before=tensor_digest(model);params=[p for p in model.parameters() if p.requires_grad]
    rows={r['query_id']:r for r in json.loads((root/'manifests/fsc147_train_groupclean.json').read_text())['queries']}
    result={'state':'running','checkpoint_sha256':digest,'purpose':'Frozen single-view CPU FP32 component derivatives, no optimizer',
      'optimizer_steps':0,'gpu_used':False,'queries':[],'limitations':['Local derivatives at a late tiny-fit checkpoint do not predict fresh training convergence.',
       'Norms and conflict fractions are not proof of Adam-update causality or evidence for an alternative coefficient.']}
    def save():
        tmp=out/'report.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(out/'report.json')
    save();started=time.monotonic()
    for name in ('7.jpg','9.jpg','81.jpg','2347.jpg'):
        b,_=prepare(rows[name],root,'cpu',1024);proto=model.encode_references(b['references'],b['reference_boxes']);v=model(b['a'],b['valid_a'],proto)
        loss=supervised_loss(v,b['points_a']);ci,gi=sparse_match(v,b['points_a']);terms={'focal':loss['focal'],'weighted_point':5*loss['point'],'weighted_count':.1*loss['count']}
        vectors={};logits={}
        for k,term in terms.items():
            # Model logits and sigmoid outputs can be sibling views of the same
            # raw tensor: include both paths when reporting dL/draw_logit.
            values=torch.autograd.grad(term,params+[v['logits'],v['probabilities']],retain_graph=True,allow_unused=True)
            vectors[k]=[torch.zeros_like(p) if x is None else x.detach() for p,x in zip(params,values[:-2])]
            direct=torch.zeros_like(v['logits']) if values[-2] is None else values[-2].detach()
            indirect=torch.zeros_like(v['probabilities']) if values[-1] is None else values[-1].detach()
            probability=v['probabilities'].detach();logits[k]=direct+indirect*probability*(1-probability)
        total=[sum(x) for x in zip(*vectors.values())];norms={k:norm(x) for k,x in vectors.items()};total_norm=norm(total)
        cosines={a+'__'+bb:dot(vectors[a],vectors[bb])/max(norms[a]*norms[bb],1e-30)
          for a,bb in [('focal','weighted_count'),('weighted_point','weighted_count'),('focal','weighted_point')]}
        item={'query_id':name,'truth':len(b['points_a']),'soft_count':float(loss['soft_count'].detach()),
          'weighted_losses':{k:float(vv.detach()) for k,vv in terms.items()},'full_parameter_gradient_norms':norms,
          'total_gradient_norm':total_norm,'single_view_clip_scale_at_1':min(1.,1/max(total_norm,1e-30)),
          'component_cosines':cosines,'focal_count_norm_ratio':norms['weighted_count']/max(norms['focal'],1e-30),
          'matched_focal_count_logit_conflict_fraction':float((logits['focal'][ci]*logits['weighted_count'][ci]<0).float().mean()),
          'matched_total_logit_descent_pushes_probability_down_fraction':float((sum(logits.values())[ci]>0).float().mean()),
          'matched_logits_focal_norm':float(logits['focal'][ci].norm()),'matched_logits_weighted_count_norm':float(logits['weighted_count'][ci].norm())}
        result['queries'].append(item);save()
        del vectors,logits,total,values,b,proto,v,loss,terms
    result.update(state='completed',model_parameters_unchanged=tensor_digest(model)==before,elapsed_seconds=time.monotonic()-started)
    if not result['model_parameters_unchanged']:raise RuntimeError('Frozen model changed')
    save();print(json.dumps(result))

if __name__=='__main__':main()
