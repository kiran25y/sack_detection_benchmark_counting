"""Section13/14 losses, with FP32 counts and sparse bilinear ownership accumulation."""
import torch
from torch.nn import functional as F
from .matching import sparse_match


def huber(x):
    x = x.float(); a = x.abs()
    return torch.where(a <= 1., .5 * x.square(), a - .5)


def supervised_loss(output, truth, exhaustive=True):
    if not exhaustive:
        raise ValueError('Exact absolute supervision requires an exhaustive audited region')
    truth = truth.to(output['points']).reshape(-1, 2)
    ci, gi = sparse_match(output, truth)
    target = torch.zeros_like(output['logits']); target[ci] = 1
    logits = output['logits'].float(); p = output['probabilities'].float()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = torch.where(target.bool(), p, 1 - p)
    alpha = torch.where(target.bool(), .25, .75)
    focal = (alpha * (1 - pt).square() * ce)[output['valid']].sum() / max(len(truth), 1)
    point = huber((output['points'][ci] - truth[gi]) / 16).sum(1).mean() if len(ci) else logits.sum() * 0
    soft_count = p[output['valid']].sum(dtype=torch.float32)
    count = huber(soft_count - len(truth))
    return {'total': focal + 5 * point + .1 * count, 'focal': focal,
            'point': point, 'count': count, 'soft_count': soft_count}


def region_mass(output, ownership, regions):
    """IDs0..R-1; background0, ignored-1. Sample IDs, never interpolate IDs."""
    if ownership.ndim != 2 or ownership.dtype != torch.long or regions < 1:
        raise ValueError('Integer ownership raster and explicit region count required')
    if (ownership < -1).any() or (ownership >= regions).any():
        raise ValueError('Ownership ID outside ledger')
    valid = output['valid']; xy = output['points'][valid].float()
    p = output['probabilities'][valid].float()
    h, w = ownership.shape
    x = xy[:, 0].clamp(0, w - 1); y = xy[:, 1].clamp(0, h - 1)
    x0 = x.floor().long(); y0 = y.floor().long()
    fx = x - x0; fy = y - y0
    mass = p.new_zeros(regions)
    for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                           (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        ids = ownership[(y0+dy).clamp_max(h-1), (x0+dx).clamp_max(w-1)]
        ok = ids >= 0
        mass = mass.scatter_add(0, ids[ok], p[ok] * weight[ok])
    return mass


ARM_RELATIONS = {'A':'none', 'B':'local', 'C':'none', 'D':'local',
                 'E':'global', 'F':'global', 'G':'local', 'H':'independent', 'I':'independent'}


def pair_loss(a, b, truth_a, truth_b, arm, masks=None, eligible=None, region_valid=None):
    if arm not in ARM_RELATIONS:
        raise ValueError('Unknown frozen arm')
    la = supervised_loss(a, truth_a); lb = supervised_loss(b, truth_b)
    global_delta = huber((lb['soft_count']-la['soft_count']) - (len(truth_b)-len(truth_a)))
    local = la['total'] * 0; independent = local
    if masks is not None:
        if eligible is None or region_valid is None:
            raise ValueError('Masks require aligned identity and validity ledgers')
        za, zb = (x.to(la['total']).float() for x in eligible)
        if za.shape != zb.shape or za.ndim != 1 or za[0] != 0 or zb[0] != 0:
            raise ValueError('Aligned region union with zero-valued background required')
        if not torch.all((za == 0) | (za == 1)) or not torch.all((zb == 0) | (zb == 1)):
            raise ValueError('Eligibility must be binary')
        ma = region_mass(a, masks[0], len(za)); mb = region_mass(b, masks[1], len(zb))
        ok = region_valid.to(device=ma.device, dtype=torch.bool)
        if ok.shape != za.shape:
            raise ValueError('Region validity shape mismatch')
        if ok.any():
            local = huber((mb-ma)-(zb-za))[ok].mean()
            independent = .5 * (huber(ma-za)[ok].mean() + huber(mb-zb)[ok].mean())
    relation = ARM_RELATIONS[arm]
    total = .5 * (la['total'] + lb['total'])
    if relation != 'none': total = total + .1 * global_delta
    if relation == 'local': total = total + .25 * local
    if relation == 'independent': total = total + .25 * independent
    return {'total':total,'view_a':la,'view_b':lb,'global_delta':global_delta,
            'local_delta':local,'independent_region':independent}
