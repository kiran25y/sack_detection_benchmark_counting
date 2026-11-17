"""Prepared two-arm trainability test. Importing this file never launches work.

Both arms continue the same verified 200-update no-count checkpoint for 200
updates. Only the GT-to-proposal association is frozen in the candidate.
This is an engineering diagnosis, not foundation or research initialization.
"""
import argparse
import gc
import hashlib
import json
import pathlib
import time

import numpy as np
import torch
from torch.nn import functional as F

from .confidence_assignment_inspection import sha
from .evaluation import metrics
from .losses import huber
from .matching import sparse_match
from .model import IPICNet
from .preflight import prepare, tensor_digest
from .training import make_optimizer, optimizer_update

CHECKPOINT_SHA = '0a8c496bf5ca60b6cf2632177ddf54557d5bcf70ce811006f96bc3440c8b0e16'
QUERY_IDS = ('7.jpg', '9.jpg', '81.jpg', '2347.jpg')


def truth_key(truth):
    return hashlib.sha256(truth.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def assigned_loss(output, truth, ci, gi):
    truth = truth.to(output['points']).reshape(-1, 2)
    target = torch.zeros_like(output['logits']); target[ci] = 1
    logits = output['logits'].float(); p = output['probabilities'].float()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = torch.where(target.bool(), p, 1 - p)
    alpha = torch.where(target.bool(), .25, .75)
    focal = (alpha * (1 - pt).square() * ce)[output['valid']].sum() / max(len(truth), 1)
    point = huber((output['points'][ci] - truth[gi]) / 16).sum(1).mean() if len(ci) else logits.sum() * 0
    soft = p[output['valid']].sum(dtype=torch.float32); count = huber(soft - len(truth))
    return {'total': focal + 5 * point + .1 * count, 'focal': focal,
            'point': point, 'count': count, 'soft_count': soft}


class AssignmentPolicy:
    def __init__(self, mode):
        if mode not in ('dynamic', 'fixed_initial_native'): raise ValueError('Unknown prespecified arm')
        self.mode = mode; self.initial = {}; self.previous = {}; self.current = {}
        self.matching_calls = 0

    def start_update(self): self.current = {}

    def view(self, output, truth):
        key = truth_key(truth)
        ci, gi = sparse_match(output, truth); self.matching_calls += 1
        if key not in self.initial: self.initial[key] = (ci.clone(), gi.clone())
        initial_ci, initial_gi = self.initial[key]
        if key not in self.current:
            prev = self.previous.get(key, ci)
            self.current[key] = {'GT_count': len(truth),
                                 'native_GT_assignment_changed_fraction': float((ci != prev).float().mean()),
                                 'native_positive_set_changed_fraction': 1 - len(set(ci.tolist()) & set(prev.tolist())) / len(ci),
                                 'native_median_positive_probability': float(output['probabilities'][ci].median()),
                                 'fixed_median_positive_probability': float(output['probabilities'][initial_ci].median())}
            self.previous[key] = ci.clone()
        used = (ci, gi) if self.mode == 'dynamic' else (initial_ci, initial_gi)
        return assigned_loss(output, truth, *used)

    def __call__(self, a, b, truth_a, truth_b, arm, masks=None, eligible=None, region_valid=None):
        if arm != 'A' or masks is not None or not torch.equal(truth_a, truth_b):
            raise ValueError('Frozen four-query, identical-view ordinary point test only')
        la = self.view(a, truth_a); lb = self.view(b, truth_b)
        total = .5 * (la['focal'] + 5 * la['point'] + lb['focal'] + 5 * lb['point'])
        zero = total * 0
        return {'total': total, 'view_a': la, 'view_b': lb,
                'global_delta': huber(lb['soft_count'] - la['soft_count']),
                'local_delta': zero, 'independent_region': zero}

    def initial_digest(self):
        h = hashlib.sha256()
        for key, pair in sorted(self.initial.items()):
            h.update(key.encode())
            for x in pair: h.update(x.cpu().numpy().tobytes())
        return h.hexdigest()


def gate(control, candidate):
    if tuple(r['query_id'] for r in control) != QUERY_IDS or tuple(r['query_id'] for r in candidate) != QUERY_IDS:
        raise ValueError('Require all four unchanged queries in frozen order')
    checks = []
    for c, v in zip(control, candidate):
        if c['query_id'] != v['query_id'] or c['truth'] != v['truth']: raise ValueError('Different coverage')
        dense = c['truth'] >= 150
        checks.append({'query_id': c['query_id'],
                       'count': (c['AE'] - v['AE'] >= max(1., .10 * c['AE'])) if dense else (v['AE'] <= c['AE'] and v['AE'] <= 2),
                       'point_F1': v['point_8']['F1'] >= c['point_8']['F1'] + (.02 if dense else -1e-8),
                       'spatial': all(v['GAME' + str(k)] <= c['GAME' + str(k)] for k in (1, 2))})
    def trainable(rows):
        return all((r['AE'] <= 2 if r['truth'] < 150 else r['AE'] / r['truth'] <= .1)
                   and r['point_8']['F1'] >= .7 for r in rows)
    return {'mechanism_gate_passed': all(all(x[k] for k in ('count', 'point_F1', 'spatial')) for x in checks),
            'component_checks': checks, 'dynamic_trainability_passed': trainable(control),
            'fixed_trainability_passed': trainable(candidate), 'automatic_full_training': False}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--output', required=True)
    a = ap.parse_args(); root = pathlib.Path(a.root); out = pathlib.Path(a.output)
    if (out / 'status.json').exists(): raise ValueError('Exclusive test, no resume/repetition')
    path = root / 'research-direction-20261012/ipic_checks_20261030/pointdiag_v1/resume.pt'
    if sha(path) != CHECKPOINT_SHA: raise ValueError('Starting checkpoint changed')
    frozen = json.loads((path.parent / 'source_hashes.json').read_text())
    for filename in ('model.py', 'losses.py', 'matching.py', 'transforms.py', 'training.py', 'preflight.py'):
        if sha(pathlib.Path(__file__).parent / filename) != frozen['counting/ipic/' + filename]:
            raise ValueError('Matched implementation changed: ' + filename)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    byid = {r['query_id']: r for r in json.loads((root / 'manifests/fsc147_train_groupclean.json').read_text())['queries']}
    selected = [byid[k] for k in QUERY_IDS]
    if any(r['split'] != 'train' for r in selected): raise ValueError('Fitting partition only')
    plan = {'architecture': 'B0', 'checkpoint_sha256': CHECKPOINT_SHA, 'extra_updates_per_arm': 200,
            'arms': ['dynamic', 'fixed_initial_native'], 'maximum_new_optimizer_calls': 400,
            'loss_both_arms': 'focal + 5*point; no soft-count term', 'optimizer': 'restore authentic AdamW state',
            'schedule': 'same constant2e-5/2e-4', 'only_intervention': 'hold initial native CUDA BF16 GT-to-proposal assignment fixed',
            'equal_matching_compute': 'both arms recompute native assignments on all16 views per update; candidate logs them as shadow assignments',
            'query_ids': QUERY_IDS, 'native_labels_exemplars': True, 'augmentation': False,
            'threshold': .5, 'checkpoint_selection': 'fixed final extra-update200', 'gpu': True,
            'full_training_auto_promotion': False, 'additional_seeds': False}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    state = {'state': 'bounded_continuation', 'optimizer_calls': 0, 'persistent_research_steps': 0, 'arms': []}
    def save(): (out / 'status.json').write_text(json.dumps(state, indent=2))
    save(); initial_identity = None; initial_assignments = None; initial_opt_steps = None
    try:
        for mode in plan['arms']:
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
            if ckpt['engineering_update'] != 200: raise ValueError('Wrong original duration')
            model = IPICNet(False).cuda().train(); model.load_state_dict(ckpt['model'], strict=True)
            opt = make_optimizer(model); opt.load_state_dict(ckpt['optimizer'])
            identity = tensor_digest(model)
            steps = [float(s['step']) for s in opt.state.values()]
            if not steps or any(s != 200 for s in steps): raise ValueError('Authentic200-update Adam state required')
            if initial_identity is None: initial_identity = identity; initial_opt_steps = steps
            elif identity != initial_identity or steps != initial_opt_steps: raise ValueError('Unmatched initialization')
            torch.set_rng_state(ckpt['cpu_rng']); torch.cuda.set_rng_state_all(ckpt['cuda_rng']); del ckpt
            batches = [prepare(r, root, 'cuda', 1024)[0] for r in selected]
            micro = [batches[j % 4] for j in range(8)]; policy = AssignmentPolicy(mode)
            record = {'name': mode, 'saved_extra_updates': 0, 'initial_model_digest': identity, 'curves': []}
            state['arms'].append(record); save()
            for step in range(200):
                policy.start_update(); torch.cuda.synchronize(); t = time.monotonic()
                measurement = optimizer_update(model, opt, micro, 'A', loss_function=policy)
                torch.cuda.synchronize(); state['optimizer_calls'] += 1
                if step == 0:
                    digest = policy.initial_digest()
                    if initial_assignments is None: initial_assignments = digest
                    elif initial_assignments != digest: raise ValueError('First BF16 assignments differ between arms')
                    record['initial_assignment_digest'] = digest
                record['curves'].append({'extra_update': step + 1, 'seconds': time.monotonic() - t,
                                         'pre_clip_norm': measurement['pre_clip_norm'],
                                         'assignment': policy.current, 'microbatch_losses': measurement['microbatch_losses']})
                if (step + 1) % 50 == 0:
                    snap = out / (mode + '_resume.tmp')
                    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                                'engineering_update': 400 if step == 199 else 200 + step + 1,
                                'extra_updates': step + 1, 'initial_assignment': policy.initial,
                                'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                                'purpose': 'disposable assignment-continuation diagnostic; not research initialization'}, snap)
                    snap.replace(out / (mode + '_resume.pt')); record['saved_extra_updates'] = step + 1; save()
            if policy.matching_calls != 3200: raise ValueError('Unequal matching compute')
            if any(float(s['step']) != 400 for s in opt.state.values()): raise ValueError('Optimizer duration mismatch')
            if not all(torch.isfinite(x).all() for x in model.parameters()): raise ValueError('Nonfinite model')
            if not all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)):
                raise ValueError('Nonfinite optimizer')
            record['matching_calls'] = policy.matching_calls; record['finite_model_optimizer'] = True
            del opt, batches, micro, policy; model.cpu().eval(); torch.cuda.empty_cache(); predictions = []
            with torch.inference_mode():
                for row in selected:
                    b, _ = prepare(row, root, 'cpu', 1024)
                    proto = model.encode_references(b['references'], b['reference_boxes']); v = model(b['a'], b['valid_a'], proto)
                    h, w = b['a'].shape[-2:]; p = model.predict(v, .5)
                    predictions.append({'query_id': row['query_id'], **metrics(p['points'].numpy(), b['points_a'].numpy(), w, h)})
                    np.savez_compressed(out / (mode + '_' + row['query_id'].replace('.jpg', '') + '.npz'),
                                        **{k: x.numpy() for k, x in v.items()}, truth=b['points_a'].numpy(), width=w, height=h)
            record['primary_predictions'] = predictions; record['checkpoint_sha256'] = sha(out / (mode + '_resume.pt'))
            del model; gc.collect(); save()
        assert state['optimizer_calls'] == 400 and sha(path) == CHECKPOINT_SHA
        decision = gate(*(r['primary_predictions'] for r in state['arms']))
        report = {'state': 'completed', 'optimizer_calls': 400, 'persistent_research_steps': 0,
                  'decision': decision, 'arms': state['arms'], 'full_training_started': False}
        (out / 'report.json').write_text(json.dumps(report, indent=2)); state.update(state='completed', decision=decision); save()
    except Exception as exc:
        state.update(state='failed', error=repr(exc)); save(); raise


if __name__ == '__main__': main()
