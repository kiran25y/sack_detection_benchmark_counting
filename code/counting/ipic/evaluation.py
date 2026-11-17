"""Point-count metrics and development-only threshold selection."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def metrics(predicted,truth,width,height):
    pred=np.asarray(predicted,float).reshape(-1,2);gt=np.asarray(truth,float).reshape(-1,2)
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():raise ValueError('Nonfinite point')
    for v in (pred,gt):
        if len(v) and ((v<0).any() or (v[:,0]>=width).any() or (v[:,1]>=height).any()):raise ValueError('Points outside original image')
    result={'prediction':len(pred),'truth':len(gt),'error':len(pred)-len(gt),'AE':abs(len(pred)-len(gt))}
    for level in (1,2):
        bins=2**level
        def hist(p):
            xy=np.floor(p/np.array([width,height])*bins).astype(int)
            return np.bincount(xy[:,1]*bins+xy[:,0],minlength=bins*bins)
        result['GAME'+str(level)]=int(np.abs(hist(pred)-hist(gt)).sum())
    scale=1024/max(width,height)
    for tol in (4,8,16):
        neighbors=cKDTree(pred*scale).query_ball_point(gt*scale,tol) if len(pred) else [[] for _ in gt]
        ri=np.repeat(np.arange(len(gt)),[len(x) for x in neighbors]);ci=np.array([j for x in neighbors for j in x],dtype=int)
        graph=csr_matrix((np.ones(len(ri)),(ri,ci)),shape=(len(gt),len(pred)))
        tp=int((maximum_bipartite_matching(graph,perm_type='column')>=0).sum()) if len(gt) and len(pred) else 0
        precision=tp/len(pred) if len(pred) else float(not len(gt))
        recall=tp/len(gt) if len(gt) else float(not len(pred))
        result['point_'+str(tol)]={'TP':tp,'FP':len(pred)-tp,'FN':len(gt)-tp,'precision':precision,'recall':recall,
            'F1':2*tp/(len(pred)+len(gt)) if len(pred)+len(gt) else 1.}
    return result


def choose_threshold(rows,split):
    if split!='development':raise ValueError('Threshold selection only on frozen natural development data')
    thresholds=np.arange(10,91,5)/100
    scores=[]
    for t in thresholds:
        errors=[abs(int((np.asarray(r['probabilities'])>=t).sum())-r['count']) for r in rows]
        if not errors:raise ValueError('Empty development set')
        scores.append((float(np.mean(errors)),-float(t)))
    mae,negative_t=min(scores)
    return {'threshold':-negative_t,'development_MAE':mae,'common_threshold':.5}
