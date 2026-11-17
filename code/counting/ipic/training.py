"""Fixed-budget optimizer mechanics. Research entry is gated by data readiness."""
import math
import torch
from .losses import pair_loss
from .data_manifest import assert_stage_ready


def make_optimizer(model):
    buckets={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        lr=2e-5 if name.startswith('context.') else 2e-4
        decay=0. if p.ndim<=1 or name.endswith('.bias') else .05
        buckets.setdefault((lr,decay),[]).append(p)
    return torch.optim.AdamW([{'params':v,'lr':lr,'initial_lr':lr,'weight_decay':wd}
                             for (lr,wd),v in buckets.items()],betas=(.9,.999),eps=1e-8)


def set_learning_rate(optimizer, update, total, warmup):
    if not 0<=update<total:raise ValueError('Update outside fixed budget')
    factor=(update+1)/warmup if update<warmup else .01+.99*.5*(1+math.cos(math.pi*(update-warmup)/max(total-warmup-1,1)))
    for group in optimizer.param_groups:group['lr']=group['initial_lr']*factor


def optimizer_update(model,optimizer,microbatches,arm,precision=torch.bfloat16,scaler=None,loss_function=None):
    """Exactly eight pairs per update. Both views and shared references keep gradients."""
    if len(microbatches)!=8:raise ValueError('Recipe requires eight accumulated image pairs')
    optimizer.zero_grad(set_to_none=True);history=[]
    device=next(model.parameters()).device
    for batch in microbatches:
        with torch.autocast(device_type=device.type,dtype=precision,enabled=device.type=='cuda'):
            proto=None if batch['all_products'] else model.encode_references(batch['references'],batch['reference_boxes'])
            a=model(batch['a'],batch['valid_a'],proto,batch['all_products'])
            b=model(batch['b'],batch['valid_b'],proto,batch['all_products'])
        # Autocast is disabled here: all count, matching and regional math is FP32.
        compute_loss=pair_loss if loss_function is None else loss_function
        losses=compute_loss(a,b,batch['points_a'],batch['points_b'],arm,batch.get('ownership'),
                         batch.get('eligible'),batch.get('region_valid'))
        loss=losses['total']/8
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; no update committed')
        if scaler is None:loss.backward()
        else:scaler.scale(loss).backward()
        history.append({k:float(losses[k].detach()) for k in ('total','global_delta','local_delta','independent_region')})
        for view in ('view_a','view_b'):
            history[-1][view]={k:float(losses[view][k].detach()) for k in ('total','focal','point','count','soft_count')}
    if scaler is not None:scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if scaler is None:optimizer.step()
    else:scaler.step(optimizer);scaler.update()
    return {'microbatch_losses':history,'pre_clip_norm':float(norm),'optimizer_updates':1}


def admit_stage(readiness,stage):
    """No command in this module launches training or changes a running job."""
    return assert_stage_ready(readiness,stage)
