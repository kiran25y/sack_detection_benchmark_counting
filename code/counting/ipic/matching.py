"""Sparse restricted point assignment; never allocate GT by all-grid cost arrays."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching, min_weight_full_bipartite_matching


class AssignmentCapacityError(ValueError):
    pass


def feasible_graph(anchors, points, valid, reach=16.):
    anchors = anchors.detach().cpu().numpy() if torch.is_tensor(anchors) else anchors
    points = points.detach().cpu().numpy() if torch.is_tensor(points) else points
    valid = valid.detach().cpu().numpy() if torch.is_tensor(valid) else valid
    a = np.asarray(anchors, dtype=np.float64)
    gt = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ids = np.flatnonzero(valid)
    if not np.isfinite(a).all() or not np.isfinite(gt).all():
        raise ValueError('Nonfinite coordinates')
    neighbors = cKDTree(a[ids]).query_ball_point(gt, reach, p=np.inf)
    rows = np.repeat(np.arange(len(gt)), [len(x) for x in neighbors])
    cols = np.array([ids[j] for x in neighbors for j in x], dtype=np.int64)
    return rows, cols, (len(gt), len(a))


def capacity_check(anchors, points, valid):
    rows, cols, shape = feasible_graph(anchors, points, valid)
    if not shape[0]:
        return {'targets': 0, 'edges': 0, 'fully_assignable': True}
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)
    match = maximum_bipartite_matching(graph, perm_type='column')
    if (match < 0).any():
        raise AssignmentCapacityError(f'Only{int((match >= 0).sum())}/{shape[0]} targets reachable uniquely')
    return {'targets': shape[0], 'edges': len(rows), 'fully_assignable': True}


@torch.no_grad()
def sparse_match(output, truth):
    truth = truth.to(device=output['points'].device, dtype=torch.float32).reshape(-1, 2)
    if not len(truth):
        empty = torch.empty(0, dtype=torch.long, device=truth.device)
        return empty, empty
    a = output['anchors'].detach().cpu().numpy()
    gt = truth.cpu().numpy(); valid = output['valid'].cpu().numpy()
    rows, cols, shape = feasible_graph(a, gt, valid)
    if shape[0] > int(valid.sum()):
        raise AssignmentCapacityError('More targets than valid candidates')
    pred = output['points'].detach().cpu().numpy()
    p = output['probabilities'].detach().cpu().numpy()
    costs = np.abs(pred[cols] - gt[rows]).sum(1) / 16 - .2 * np.log(p[cols] + 1e-6)
    if not np.isfinite(costs).all():
        raise ValueError('Nonfinite matching costs')
    # A positive common shift preserves the full-assignment minimizer and sparse zero edges.
    costs = costs.astype(np.float64) + max(1., 1. - costs.min(initial=0.))
    graph = csr_matrix((costs, (rows, cols)), shape=shape)
    try:
        ri, ci = min_weight_full_bipartite_matching(graph)
    except ValueError as e:
        raise AssignmentCapacityError('Full restricted assignment unavailable; no GT may be dropped') from e
    if len(ri) != len(gt):
        raise AssignmentCapacityError('Incomplete assignment')
    return torch.as_tensor(ci, device=truth.device), torch.as_tensor(ri, device=truth.device)
