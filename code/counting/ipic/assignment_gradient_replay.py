"""Zero optimizer calls: compare repeat-backward noise with callback differences.

This diagnoses the preserved failed CUDA guard; it cannot admit training.
"""
import argparse
import gc
import json
import pathlib
import time

import torch

from .assignment_continuation import AssignmentPolicy, CHECKPOINT_SHA, QUERY_IDS
from .assignment_cuda_preflight import state_digest
from .confidence_assignment_inspection import sha
from .model import IPICNet
from .point_fit_diagnostic import point_only_pair
from .preflight import prepare, tensor_digest
from .training import make_optimizer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args(); root = pathlib.Path(args.root); out = pathlib.Path(args.output)
    if (out / 'report.json').exists(): raise ValueError('Exclusive diagnostic')
    source = root / 'research-direction-20261012/ipic_checks_20261030/pointdiag_v1/resume.pt'
    if sha(source) != CHECKPOINT_SHA: raise ValueError('Source identity changed')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    rows = {r['query_id']: r for r in json.loads((root / 'manifests/fsc147_train_groupclean.json').read_text())['queries']}
    batches = [prepare(rows[k], root, 'cuda', 1024)[0] for k in QUERY_IDS]
    report = {'state': 'running', 'optimizer_calls': 0, 'persistent_research_steps': 0,
              'maximum_backward_passes': 24, 'maximum_query_forwards': 48, 'maximum_reference_forwards': 24,
              'guard_unchanged': True, 'automatic_training_admission': False, 'runs': []}
    def save():
        p = out / 'report.tmp'; p.write_text(json.dumps(report, indent=2)); p.replace(out / 'report.json')
    save(); first = None; baseline_grad = None
    try:
        for name in ('retained_first', 'retained_repeat', 'dynamic_callback'):
            ck = torch.load(source, map_location='cpu', weights_only=True)
            model = IPICNet(False).cuda().train(); model.load_state_dict(ck['model'], strict=True)
            opt = make_optimizer(model); opt.load_state_dict(ck['optimizer'])
            torch.set_rng_state(ck['cpu_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng']); del ck
            identity = {'model': tensor_digest(model), 'optimizer': state_digest(opt.state_dict()),
                        'cpu_rng': state_digest(torch.get_rng_state()), 'cuda_rng': state_digest(torch.cuda.get_rng_state_all())}
            if first is None: first = identity
            elif first != identity: raise ValueError('Different initial states')
            callback = AssignmentPolicy('dynamic') if name == 'dynamic_callback' else point_only_pair
            model.zero_grad(set_to_none=True); values = []; fields = []; started = time.monotonic()
            for i in range(8):
                b = batches[i % 4]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    proto = model.encode_references(b['references'], b['reference_boxes'])
                    a = model(b['a'], b['valid_a'], proto); z = model(b['b'], b['valid_b'], proto)
                fields.append(state_digest({'a': a, 'b': z}))
                loss = callback(a, z, b['points_a'], b['points_b'], 'A')['total'] / 8
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward(); values.append(float(loss.detach()))
                del proto, a, z, loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            grad = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}
            if not all(torch.isfinite(g).all() for g in grad.values()): raise ValueError('Nonfinite gradient')
            if tensor_digest(model) != identity['model'] or state_digest(opt.state_dict()) != identity['optimizer']:
                raise ValueError('Zero-update diagnostic changed model or optimizer')
            record = {'name': name, 'seconds': time.monotonic()-started, 'initial_identity': identity,
                      'pre_clip_norm': float(norm), 'losses': values, 'forward_field_hashes': fields,
                      'model_optimizer_unchanged': True, 'finite_gradients': True}
            torch.save(grad, out / (name + '_clipped_gradients.pt'))
            record['gradient_file_sha256'] = sha(out / (name + '_clipped_gradients.pt'))
            if baseline_grad is None: baseline_grad = grad; reference_fields = fields
            else:
                diffs = []
                for key, x in grad.items():
                    y = baseline_grad[key]; delta = (x-y).double()
                    diffs.append({'parameter': key, 'max_abs': float(delta.abs().max()),
                                  'squared_difference': float(delta.square().sum()),
                                  'baseline_squared_norm': float(y.double().square().sum()),
                                  'different_elements': int((x!=y).sum()), 'elements': x.numel()})
                record['same_forward_fields'] = fields == reference_fields
                record['gradient_comparison_to_first'] = diffs
            report['runs'].append(record); save()
            del model, opt, callback, grad; gc.collect(); torch.cuda.empty_cache()
        if sha(source) != CHECKPOINT_SHA: raise ValueError('Original checkpoint changed')
        report.update(state='completed', original_checkpoint_unchanged=True, optimizer_calls=0,
                      claim_limit='Repeatability diagnosis only; no guard override, no matched trajectory, no accuracy result.')
        save()
    except Exception as exc:
        report.update(state='failed', error=repr(exc)); save(); raise


if __name__ == '__main__': main()
