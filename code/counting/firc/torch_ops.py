"""FP32 splat and loss backend, checked against the independent NumPy oracle."""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching,min_weight_full_bipartite_matching


def splat(points,masses,field):
    # index_add runs under strict deterministic-algorithm mode in our runtime.
    dtype=torch.float64 if points.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=points.device.type,enabled=False):
        xy=points.to(dtype);m=masses.to(dtype)
        if xy.shape!=(len(m),2) or not torch.isfinite(xy).all() or not torch.isfinite(m).all() or (m<0).any():raise ValueError('Invalid point measure')
        if (xy<0).any() or (xy[:,0]>field.width).any() or (xy[:,1]>field.height).any():raise ValueError('Out-of-field point')
        grid=xy/4-.5;low=grid.floor().long();t=grid-low
        ids=[];weights=[]
        for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
            x=(low[:,0]+dx).clamp(0,(field.width+3)//4-1)
            y=(low[:,1]+dy).clamp(0,(field.height+3)//4-1)
            ids.append(y*field.shape[1]+x)
            weights.append(m*(t[:,0] if dx else 1-t[:,0])*(t[:,1] if dy else 1-t[:,1]))
        out=torch.zeros(field.shape[0]*field.shape[1],device=xy.device,dtype=dtype)
        return out.index_add(0,torch.cat(ids),torch.cat(weights)).reshape(field.shape)


def pyramid(a):
    def pool(v):return v.reshape(v.shape[0]//2,2,v.shape[1]//2,2).sum((1,3))
    b=pool(a);return {4:a,8:b,16:pool(b)}


def norm(maps,memberships,counts):
    weights=maps[4].new_tensor([1,1,1,.25])/maps[4].new_tensor(counts).clamp(min=1)
    return sum(sw*(v.abs()[None]*memberships[s]*weights[:,None,None]).sum() for s,sw in ((4,.5),(8,.3),(16,.2)) for v in [maps[s]])


def response_terms(predicted,truth,memberships,counts):
    e=predicted-truth
    measure=lambda v:norm(pyramid(v),memberships,counts)
    mixed=e[3]-e[1]-e[2]+e[0]
    return {'independent':torch.stack([measure(v) for v in e]).mean(),
            'first':torch.stack([measure(e[b]-e[a]) for a,b in ((0,1),(2,3),(0,2),(1,3))]).mean(),
            'mixed':measure(mixed),'scalar_mixed':mixed.sum().abs()}


def match(row,truth):
    """All feasible edges; refuse incomplete labels rather than dropping GT."""
    a=row['anchors'].detach().cpu().double().numpy();p=row['points'].detach().cpu().double().numpy()
    q=row['probabilities'].detach().cpu().double().numpy();g=truth.detach().cpu().double().numpy()
    if len(g)==0:return np.empty(0,dtype=np.int64),np.empty(0,dtype=np.int64)
    ii=[];jj=[];cost=[]
    for i,point in enumerate(g):
        js=np.flatnonzero((np.abs(a-point)<=16).all(1))
        ii.extend([i]*len(js));jj.extend(js.tolist())
        cost.extend((np.abs(p[js]-point).sum(1)/16-.2*np.log(q[js]+1e-6)+1e-8).tolist())
    graph=csr_matrix((np.ones(len(ii)),(ii,jj)),shape=(len(g),len(a)))
    if (maximum_bipartite_matching(graph,perm_type='column')<0).any():raise ValueError('GT does not have a complete reachable-anchor matching')
    matrix=csr_matrix((cost,(ii,jj)),shape=graph.shape)
    gi,pi=min_weight_full_bipartite_matching(matrix)
    if len(gi)!=len(g) or len(set(pi))!=len(g):raise RuntimeError('Incomplete assignment')
    return gi,pi


def view_loss(row,truth):
    gi,pi=match(row,truth)
    logits=row['logits'].float();q=logits.sigmoid();target=torch.zeros_like(logits);target[pi]=1
    ce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    pt=torch.where(target.bool(),q,1-q);alpha=torch.where(target.bool(),.25,.75)
    focal=(alpha*(1-pt).square()*ce).sum()/max(1,len(truth))
    point=F.huber_loss((row['points'][pi]-truth[gi]).float()/16,torch.zeros((len(gi),2),device=q.device),delta=1,reduction='sum')/max(1,len(gi))
    count=F.huber_loss(q.sum(),q.new_tensor(float(len(truth))),delta=1,reduction='sum')
    return {'total':focal+5*point+.10*count,'focal':focal,'point':point,'count':count,'matched':len(gi)}
