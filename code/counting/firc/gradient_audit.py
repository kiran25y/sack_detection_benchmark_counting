"""Zero-update descriptive gradient audit; never changes the optimizer or model."""
import torch
from .torch_ops import match, view_loss
from .tiny_fit import state_digest, write_json


def audit(model, optimizer, forward, truth, rows, output):
    params = [p for p in model.parameters() if p.requires_grad]
    before = state_digest({k: v for k, v in model.state_dict().items() if not k.startswith('encoder.')})
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]
    outputs = forward()
    losses = [view_loss(r, g) for r, g in zip(outputs, truth)]
    components = {'focal': torch.stack([v['focal'] for v in losses]).mean(),
                  'point_x5': torch.stack([5*v['point'] for v in losses]).mean(),
                  'count_x_point1': torch.stack([.1*v['count'] for v in losses]).mean()}
    vectors, norms = {}, {}
    for name, value in components.items():
        gradients = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        flat = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().float().flatten()
                          for p, g in zip(params, gradients)])
        assert torch.isfinite(flat).all()
        vectors[name] = flat
        norms[name] = float(flat.norm())
    cosines = {}
    pairs = [('focal', 'count_x_point1'), ('point_x5', 'count_x_point1'), ('focal', 'point_x5')]
    for a, b in pairs:
        den = norms[a]*norms[b]
        cosines[a+'__'+b] = float(torch.dot(vectors[a], vectors[b])/den) if den else None
    per_image = []
    for row, prediction, target, loss in zip(rows, outputs, truth, losses):
        _, pi = match(prediction, target)
        positive = torch.zeros_like(prediction['logits'], dtype=torch.bool)
        positive[pi] = True
        gradients = {}
        for name, value in [('focal', loss['focal']), ('count_x_point1', .1*loss['count'])]:
            g = torch.autograd.grad(value, prediction['logits'], retain_graph=True)[0]
            gradients[name] = {'matched_sum': float(g[positive].sum()),
                               'unmatched_sum': float(g[~positive].sum()),
                               'matched_mean': float(g[positive].mean()) if len(pi) else None,
                               'unmatched_mean': float(g[~positive].mean()), 'norm': float(g.norm())}
        per_image.append({'image_id': row['image_id'], 'truth': len(target),
                          'soft_count': float(prediction['probabilities'].sum().detach()),
                          'matched': len(pi), 'logit_gradients': gradients})
    assert before == state_digest({k: v for k, v in model.state_dict().items() if not k.startswith('encoder.')})
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    record = {'optimizer_calls': 0, 'head_sha256_before_after': before,
              'rng_unchanged': True, 'model_unchanged': True,
              'parameter_gradient_norms': norms, 'parameter_gradient_cosines': cosines,
              'loss_values': {k: float(v.detach()) for k, v in components.items()},
              'per_image': per_image,
              'interpretation': 'Signed gradients before clipping/Adam. Positive gradient descent lowers logits. Descriptive interference, not a convergence or novelty claim.'}
    write_json(output, record)
    return record
