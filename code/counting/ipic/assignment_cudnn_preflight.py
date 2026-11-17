"""Four disposable updates under matched deterministic cuDNN convolution selection.

The original failed guard is preserved. All numerical tolerances are unchanged.
This flag does not assert full CUDA bitwise determinism.
"""
import argparse
import gc
import hashlib
import json
import pathlib
import time

import torch

from .assignment_continuation import AssignmentPolicy, CHECKPOINT_SHA, QUERY_IDS
from .confidence_assignment_inspection import sha
from .model import IPICNet
from .point_fit_diagnostic import point_only_pair
from .preflight import prepare, tensor_digest
from .training import make_optimizer, optimizer_update


def state_digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y = x.detach().cpu().contiguous()
            h.update(str((y.dtype, tuple(y.shape))).encode()); h.update(y.numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str): h.update(repr(k).encode()); visit(x[k])
        elif isinstance(x, (list, tuple)):
            for y in x: visit(y)
        else: h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args(); root = pathlib.Path(args.root); out = pathlib.Path(args.output)
    if (out / 'report.json').exists(): raise ValueError('Exclusive preflight; no retry')
    source = root / 'research-direction-20261012/ipic_checks_20261030/pointdiag_v1/resume.pt'
    if sha(source) != CHECKPOINT_SHA: raise ValueError('Starting checkpoint changed')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    report = {'state': 'running', 'maximum_optimizer_calls': 4, 'optimizer_calls': 0,
              'persistent_research_steps': 0, 'engineering_guard_passed': False,
              'checkpoint_sha256': CHECKPOINT_SHA, 'query_ids': QUERY_IDS,
              'initial_callback_value_gradient_tolerance': 'exact on the same actual CUDA BF16 outputs',
              'poststep_parameter_moment_tolerance': {'rtol': .003, 'atol': 1e-5},
              'poststep_is_numerical_not_bitwise': True, 'cudnn_deterministic': True, 'retained_repeat_required': True, 'callback_checks': [], 'updates': []}
    def save():
        tmp = out / 'report.tmp'; tmp.write_text(json.dumps(report, indent=2)); tmp.replace(out / 'report.json')
    save()
    try:
        frozen = json.loads((source.parent / 'source_hashes.json').read_text())
        for name in ('model.py', 'losses.py', 'matching.py', 'transforms.py', 'training.py', 'preflight.py', 'point_fit_diagnostic.py'):
            if sha(pathlib.Path(__file__).parent / name) != frozen['counting/ipic/' + name]:
                raise ValueError('Matched source changed: ' + name)
        rows = {r['query_id']: r for r in json.loads((root / 'manifests/fsc147_train_groupclean.json').read_text())['queries']}
        if any(rows[k]['split'] != 'train' for k in QUERY_IDS): raise ValueError('Not fitting data')
        batches = [prepare(rows[k], root, 'cuda', 1024)[0] for k in QUERY_IDS]
        micro = [batches[i % 4] for i in range(8)]
        initial = None; assignment = None; reference = None
        torch.cuda.reset_peak_memory_stats()
        for mode in ('retained_point_only', 'retained_repeat', 'dynamic', 'fixed_initial_native'):
            ck = torch.load(source, map_location='cpu', weights_only=True)
            if ck['engineering_update'] != 200: raise ValueError('Wrong source duration')
            model = IPICNet(False).cuda().train(); model.load_state_dict(ck['model'], strict=True)
            opt = make_optimizer(model); opt.load_state_dict(ck['optimizer'])
            torch.set_rng_state(ck['cpu_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
            if state_digest(model.state_dict()) != state_digest(ck['model']): raise ValueError('Model recovery mismatch')
            if state_digest(opt.state_dict()) != state_digest(ck['optimizer']): raise ValueError('Adam recovery mismatch')
            if not torch.equal(torch.get_rng_state(), ck['cpu_rng']): raise ValueError('CPU RNG recovery mismatch')
            if state_digest(torch.cuda.get_rng_state_all()) != state_digest(ck['cuda_rng']): raise ValueError('CUDA RNG recovery mismatch')
            identity = {'model': tensor_digest(model), 'optimizer': state_digest(opt.state_dict()),
                        'cpu_rng': state_digest(torch.get_rng_state()), 'cuda_rng': state_digest(torch.cuda.get_rng_state_all())}
            if initial is None: initial = identity
            elif initial != identity: raise ValueError('Arm initial identities differ')
            if any(float(s['step']) != 200 for s in opt.state.values()): raise ValueError('Not authentic step200')
            if mode == 'retained_point_only':
                # Compare loss values and all consumed output gradients on each real BF16 field.
                for q, b in zip(QUERY_IDS, batches):
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        proto = model.encode_references(b['references'], b['reference_boxes'])
                        field = model(b['a'], b['valid_a'], proto)
                    answers = []
                    for callback in (point_only_pair, AssignmentPolicy('dynamic'), AssignmentPolicy('fixed_initial_native')):
                        v = {k: x.detach().clone() for k, x in field.items()}
                        keys = ('logits', 'probabilities', 'points')
                        for k in keys: v[k].requires_grad_(True)
                        loss = callback(v, v, b['points_a'], b['points_b'], 'A')['total']
                        grads = torch.autograd.grad(loss, [v[k] for k in keys])
                        answers.append((loss.detach(), tuple(g.detach() for g in grads)))
                    for value, grads in answers[1:]:
                        torch.testing.assert_close(value, answers[0][0], rtol=0, atol=0)
                        for x, y in zip(grads, answers[0][1]): torch.testing.assert_close(x, y, rtol=0, atol=0)
                    report['callback_checks'].append({'query_id': q, 'exact_value_gradient_parity': True})
                    del field, proto, answers, v, grads, loss
                # The checks above may consume RNG: restore it before the matched update.
                torch.set_rng_state(ck['cpu_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
            del ck
            callback = point_only_pair if mode in ('retained_point_only', 'retained_repeat') else AssignmentPolicy(mode)
            if mode in ('dynamic', 'fixed_initial_native'): callback.start_update()
            torch.cuda.synchronize(); start = time.monotonic()
            measured = optimizer_update(model, opt, micro, 'A', loss_function=callback)
            torch.cuda.synchronize(); report['optimizer_calls'] += 1
            if any(float(s['step']) != 201 for s in opt.state.values()): raise ValueError('Wrong disposable duration')
            if not all(torch.isfinite(p).all() for p in model.parameters()): raise ValueError('Nonfinite model')
            if not all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)):
                raise ValueError('Nonfinite optimizer')
            record = {'mode': mode, 'seconds': time.monotonic() - start, 'initial_identity': identity,
                      'preupdate_exact_recovery': True, 'finite_model_optimizer': True, 'measurement': measured}
            if mode in ('dynamic', 'fixed_initial_native'):
                record['initial_assignment_digest'] = callback.initial_digest()
                record['matching_calls'] = callback.matching_calls
                if len(callback.initial) != 4 or callback.matching_calls != 16: raise ValueError('Assignment coverage mismatch')
                if assignment is None: assignment = callback.initial_digest()
                elif assignment != callback.initial_digest(): raise ValueError('Initial assignments differ')
            current = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       'moments': {str(i) + ':' + k: v.detach().cpu().clone() for i, s in opt.state_dict()['state'].items()
                                   for k, v in s.items() if torch.is_tensor(v)}}
            if reference is None: reference = current
            else:
                diffs = {}
                for group in current:
                    maximum = 0.
                    for key, x in current[group].items():
                        y = reference[group][key]
                        try: torch.testing.assert_close(x, y, rtol=.003, atol=1e-5)
                        except AssertionError as exc:
                            report['failure_tensor'] = {'mode': mode, 'group': group, 'key': key, 'max_abs': float((x-y).abs().max())}
                            raise AssertionError(str(report['failure_tensor']) + ': ' + str(exc)) from exc
                        maximum = max(maximum, float((x - y).abs().max()))
                    diffs[group] = maximum
                record['poststep_max_abs_difference_from_retained'] = diffs
            report['updates'].append(record); save()
            del current, model, opt, callback; gc.collect(); torch.cuda.empty_cache()
        if sha(source) != CHECKPOINT_SHA or report['optimizer_calls'] != 4: raise ValueError('Source/budget mismatch')
        report.update(state='completed', engineering_guard_passed=True,
                      peak_gpu_allocated=torch.cuda.max_memory_allocated(), peak_gpu_reserved=torch.cuda.max_memory_reserved(),
                      original_checkpoint_unchanged=True, full_training_started=False)
        save()
    except Exception as exc:
        report.update(state='failed', error=repr(exc)); save(); raise


if __name__ == '__main__': main()
