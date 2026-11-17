"""Frozen, zero-update separation of confidence, localization and assignment.

Counterfactual assignments are diagnostics, not alternative trained models.
Geometric coverage is reported alongside the unchanged dense anchor grid.
"""
import argparse
import gc
import hashlib
import json
import pathlib
import time

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from .evaluation import metrics
from .matching import capacity_check, feasible_graph, sparse_match
from .model import IPICNet
from .preflight import prepare, tensor_digest

STATES = (
    ('original_count', 'tiny_v1/B0_engineering_resume.pt',
     '36298a418342eabc9d2828e24a93d07c1a119fedf1190c4f9922e98fcf474d3d'),
    ('no_count', 'pointdiag_v1/resume.pt',
     '0a8c496bf5ca60b6cf2632177ddf54557d5bcf70ce811006f96bc3440c8b0e16'),
)
QUERIES = ('7.jpg', '9.jpg', '81.jpg', '2347.jpg')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def quantiles(a):
    a = np.asarray(a)
    return dict(zip(('min', 'p10', 'p50', 'p90', 'max'),
                    map(float, np.quantile(a, [0, .1, .5, .9, 1])))) if a.size else None


def solve(rows, cols, shape, cost):
    # Same common positive shift and sparse full assignment as production.
    cost = cost.astype(np.float64) + max(1., 1. - cost.min(initial=0.))
    ri, ci = min_weight_full_bipartite_matching(csr_matrix((cost, (rows, cols)), shape=shape))
    if not np.array_equal(ri, np.arange(shape[0])) or len(np.unique(ci)) != shape[0]:
        raise ValueError('Incomplete/nonunique assignment')
    return ci


def assignment_summary(ci, gt, points, probability):
    distance = np.linalg.norm(points[ci] - gt, axis=1)
    p = probability[ci]
    return {'probability': quantiles(p), 'distance_pixels': quantiles(distance),
            'mean_distance_pixels': float(distance.mean()),
            'within_2': int((distance <= 2).sum()), 'within_4': int((distance <= 4).sum()),
            'within_8': int((distance <= 8).sum()), 'within_16': int((distance <= 16).sum()),
            'at_least_half_confidence': int((p >= .5).sum()),
            'localized_at8_below_half_confidence': int(((distance <= 8) & (p < .5)).sum()),
            'mislocalized_at8_below_half_confidence': int(((distance > 8) & (p < .5)).sum()),
            'localized_at8_at_least_half_confidence': int(((distance <= 8) & (p >= .5)).sum())}


def inspect_arrays(v, gt):
    points, anchors, p, valid, logits = (v[k].numpy() for k in
                                         ('points', 'anchors', 'probabilities', 'valid', 'logits'))
    gt = gt.numpy()
    rows, cols, shape = feasible_graph(anchors, gt, valid)
    distance_cost = np.abs(points[cols] - gt[rows]).sum(1) / 16
    native_cost = distance_cost - .2 * np.log(p[cols] + 1e-6)
    native = solve(rows, cols, shape, native_cost)
    production_ci, production_gi = sparse_match(v, torch.from_numpy(gt))
    if not np.array_equal(native, production_ci.numpy()) or not np.array_equal(production_gi.numpy(), np.arange(len(gt))):
        raise ValueError('Independent assignment differs from production')
    assignments = {'native': native,
                   'decoded_distance_only': solve(rows, cols, shape, distance_cost),
                   'static_anchor_distance': solve(rows, cols, shape, np.abs(anchors[cols] - gt[rows]).sum(1) / 16)}
    summaries = {}
    for key, ci in assignments.items():
        summaries[key] = assignment_summary(ci, gt, points, p)
        summaries[key]['different_GT_to_proposal_fraction_vs_native'] = float((ci != native).mean())
        summaries[key]['positive_set_overlap_fraction_vs_native'] = float(len(np.intersect1d(ci, native)) / len(gt))
    # Fixed, tiny perturbations probe local numerical assignment sensitivity only.
    # They neither simulate optimizer steps nor measure historical label turnover.
    index = np.arange(len(p), dtype=np.float64)
    perturbations = []
    for sign in (-1, 1):
        changed_points = points + sign * .01 * np.stack((np.sin(index), np.cos(index)), 1)
        changed_logits = logits + sign * .001 * np.sin(index * .73)
        changed_p = 1 / (1 + np.exp(-changed_logits))
        cost = np.abs(changed_points[cols] - gt[rows]).sum(1) / 16 - .2 * np.log(changed_p[cols] + 1e-6)
        ci = solve(rows, cols, shape, cost)
        assignments['local_perturbation_' + str(sign)] = ci
        perturbations.append({'sign': sign, 'coordinate_amplitude_px': .01, 'logit_amplitude': .001,
                              'different_GT_to_proposal_fraction': float((ci != native).mean()),
                              'changed_positive_set_fraction': 1 - float(len(np.intersect1d(ci, native)) / len(gt))})
    matched = np.zeros(len(p), dtype=bool); matched[native] = True
    nearest = cKDTree(gt).query(points, k=1)[0]
    confident = valid & (p >= .5)
    near_unmatched = valid & ~matched & (nearest <= 8)
    assigned_dist = np.linalg.norm(points[native] - gt, axis=1)
    nearest_dist = cKDTree(points[valid]).query(gt, k=1)[0]
    return {'capacity': capacity_check(anchors, gt, valid), 'valid_proposals': int(valid.sum()),
            'feasible_candidates_per_GT': quantiles(np.bincount(rows, minlength=len(gt))),
            'assignment': summaries, 'local_sensitivity': perturbations,
            'native_distance_minus_nearest_candidate': quantiles(assigned_dist - nearest_dist),
            'unmatched_proposals_within8': int(near_unmatched.sum()),
            'unmatched_near_point_probability': quantiles(p[near_unmatched]),
            'confident_unmatched_within8': int((confident & near_unmatched).sum()),
            'confidence_median_native_positive_minus_near_unmatched':
                float(np.median(p[native]) - np.median(p[near_unmatched])) if near_unmatched.any() else None,
            'unmatched_semantics': 'Assignment labels, not independently verified semantic negatives.'}, assignments


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args(); root = pathlib.Path(args.root); out = pathlib.Path(args.output)
    if (out / 'status.json').exists():
        raise ValueError('Exclusive read-only diagnostic; do not duplicate')
    torch.set_num_threads(2)
    base = root / 'research-direction-20261012/ipic_checks_20261030'
    frozen = json.loads((base / 'tiny_v1/source_hashes.json').read_text())
    for filename in ('model.py', 'preflight.py', 'losses.py', 'matching.py', 'transforms.py'):
        if sha(pathlib.Path(__file__).parent / filename) != frozen['counting/ipic/' + filename]:
            raise ValueError('Frozen model/input math changed: ' + filename)
    manifest = root / 'manifests/fsc147_train_groupclean.json'
    byid = {r['query_id']: r for r in json.loads(manifest.read_text())['queries']}
    plan = {'purpose': 'Separate frozen fitted confidence/localization/assignment, not accuracy or training-time churn',
            'states': STATES, 'queries': QUERIES, 'forwards': 8, 'optimizer_calls': 0,
            'device': 'cpu', 'precision': 'float32', 'cpu_threads': 2,
            'original_manifest_sha256': sha(manifest), 'primary_threshold': .5,
            'comparison_thresholds': [n / 100 for n in range(10, 91, 5)],
            'assignments': ['native', 'decoded_distance_only', 'static_anchor_distance'],
            'perturbations': 'both signs, coordinate0.01px/logit0.001; local sensitivity only',
            'heldout_queries_used': 0, 'full_training_auto_promotion': False}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    state = {'state': 'inspecting', 'completed_forwards': 0, 'optimizer_calls': 0, 'persistent_steps': 0}
    report = {'plan_sha256': sha(out / 'plan.json'), 'queries': [], 'checkpoints': [],
              'optimizer_calls': 0, 'persistent_steps': 0, 'gpu_used': False}
    def save():
        (out / 'status.json').write_text(json.dumps(state, indent=2))
        (out / 'report.json').write_text(json.dumps(report, indent=2))
    save(); start = time.monotonic()
    try:
        for name, relative, expected in STATES:
            path = base / relative
            if sha(path) != expected: raise ValueError('Saved checkpoint identity mismatch')
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
            if ckpt['engineering_update'] != 200: raise ValueError('Wrong saved duration')
            if not all(torch.isfinite(v).all() for v in ckpt['model'].values()): raise ValueError('Nonfinite saved model')
            if not all(torch.isfinite(v).all() for s in ckpt['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v)):
                raise ValueError('Nonfinite saved optimizer')
            model = IPICNet(False); model.load_state_dict(ckpt['model'], strict=True); model.eval(); del ckpt; gc.collect()
            before = tensor_digest(model)
            with torch.inference_mode():
                for query in QUERIES:
                    row = byid[query]
                    if row['split'] != 'train': raise ValueError('Fitting queries only')
                    batch, geometry = prepare(row, root, 'cpu', 1024)
                    proto = model.encode_references(batch['references'], batch['reference_boxes'])
                    v = model(batch['a'], batch['valid_a'], proto)
                    if not all(torch.isfinite(x).all() for x in v.values()): raise ValueError('Nonfinite outputs')
                    gt = batch['points_a']; h, w = batch['a'].shape[-2:]
                    result, assignments = inspect_arrays(v, gt)
                    result.update(state=name, query_id=query, truth=len(gt), image_sha256=row['image_sha256'], width=w, height=h)
                    result['thresholds'] = [{'threshold': n / 100, **metrics(model.predict(v, n / 100)['points'].numpy(), gt.numpy(), w, h)}
                                            for n in range(10, 91, 5)]
                    result['unthresholded_geometry'] = {key: metrics(v[key][v['valid']].numpy(), gt.numpy(), w, h)
                                                       for key in ('anchors', 'points')}
                    result['geometry_caveat'] = 'Dense untrained anchors can cover all truths; coverage alone is not learned localization.'
                    archive = out / (name + '_' + query.replace('.jpg', '') + '.npz')
                    np.savez_compressed(archive, **{k: x.numpy() for k, x in v.items()}, truth=gt.numpy(),
                                        width=np.array(w), height=np.array(h),
                                        **{'assignment_' + k: ci for k, ci in assignments.items()})
                    result.update(array_file=archive.name, array_sha256=sha(archive))
                    report['queries'].append(result); state['completed_forwards'] += 1; save()
                    del batch, proto, v
            after = tensor_digest(model)
            if before != after or sha(path) != expected: raise ValueError('Frozen state changed')
            report['checkpoints'].append({'state': name, 'checkpoint_sha256': expected, 'saved_updates': 200,
                                          'finite_model_optimizer': True, 'model_tensor_digest': before, 'unchanged': True})
            del model; gc.collect(); save()
        report.update(state='completed', elapsed_seconds=time.monotonic() - start, finite_outputs=True,
                      limitations='Four seen fitting images. Local perturbations cannot establish historical assignment churn; no alternate loss trained.')
        state.update(state='completed'); save()
    except Exception as exc:
        state.update(state='failed', error=repr(exc)); save(); raise


if __name__ == '__main__':
    main()
