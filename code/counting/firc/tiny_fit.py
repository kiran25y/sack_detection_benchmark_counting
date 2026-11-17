"""One ordinary-supervision trainability diagnostic, never a quartet experiment."""
import argparse, hashlib, json, math, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .geometry import transform, image_to_model, points_to_model
from .model import FIRC, load_official_encoder, optimizer_for
from .preflight import configure
from .torch_ops import view_loss


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)


def run(args):
    out, data, assets = Path(args.output), Path(args.data), Path(args.assets)
    plan = json.loads((data / 'plan.json').read_text())
    assert plan['updates'] == 150 and plan['threshold'] == .5
    assert plan['eligible_quartets'] == 0 and plan['seed'] == 1729
    assert plan['loss'] == {'focal_alpha': .25, 'focal_gamma': 2., 'point_weight': 5., 'count_weight': .10}
    for name, sha in json.loads((data / 'hashes.json').read_text()).items():
        assert digest(data / name) == sha, name
    configure(); assert torch.cuda.is_bf16_supported()
    rows, references = plan['rows'], plan['references']
    assert len(rows) == 4 and all(r['split'] == 'train' and r['exhaustive_reviewed'] for r in rows)
    assert [r['count'] for r in rows] == [0, 1, 21, 34]
    images, truth, transforms = [], [], []
    for r in rows:
        assert len(r['points']) == r['count']
        with Image.open(data / r['image']) as im:
            assert im.size == (r['width'], r['height'])
            t = transform(*im.size, 896, r['roi_xyxy'])
            t['padded_hw'] = [896, 896]  # fixed shared canvas, padding is normalized zero
            images.append(torch.from_numpy(image_to_model(im, t)))
        truth.append(torch.tensor(points_to_model(r['points'], t), dtype=torch.float32, device='cuda'))
        transforms.append(t)
    ref_images, boxes = [], []
    for i, r in enumerate(references):
        assert r['split'] == 'train'
        with Image.open(data / r['image']) as im:
            t = transform(*im.size, 224); t['padded_hw'] = [224, 224]
            ref_images.append(torch.from_numpy(image_to_model(im, t)))
            h, w = t['valid_hw']; boxes.append([i, 0, 0, w, h])
    images = torch.stack(images).cuda(); refs = torch.stack(ref_images).cuda()
    boxes = torch.tensor(boxes, dtype=torch.float32, device='cuda')
    valid_hw = [t['valid_hw'] for t in transforms]
    checkpoint = assets / 'dinov2_vitb14_reg4_pretrain.pth'
    assert digest(checkpoint) == plan['pretrained_sha256']
    model = FIRC(load_official_encoder(checkpoint)).cuda().train()
    encoder_before = state_digest(model.encoder.state_dict())
    initial_head = state_digest({k: v for k, v in model.state_dict().items() if not k.startswith('encoder.')})
    optimizer = optimizer_for(model)
    # Frozen features only: exact same pixels, resolution, padding and references at every update.
    with torch.autocast('cuda', dtype=torch.bfloat16):
        taps = tuple(t.detach() for t in model.encode(images))
        rt = tuple(t.detach() for t in model.encode(refs))
    report = {'state': 'training', 'saved_updates': 0, 'optimizer_calls': 0,
              'plan_sha256': digest(data / 'plan.json'), 'encoder_initial_sha256': encoder_before,
              'head_initial_sha256': initial_head, 'eligible_quartets': 0,
              'purpose': plan['purpose'], 'transforms': transforms,
              'parameters': {'total': sum(p.numel() for p in model.parameters()),
                             'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)},
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'feature_cache': 'frozen encoder only; common neck/reference conditioning recomputed each bundle',
              'threshold': .5, 'seed': 1729, 'finite_gradients': True, 'finite_model_adam': True}

    def forward():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(images, refs, boxes, valid_hw, taps=taps, reference_taps=rt)

    def predictions(stage):
        model.eval()
        with torch.no_grad():
            outputs = forward()
        files = []
        for r, result, t in zip(rows, outputs, transforms):
            name = stage + '_' + r['image_id'] + '.npz'
            points = result['points'].cpu().numpy(); probs = result['probabilities'].cpu().numpy()
            assert np.isfinite(points).all() and np.isfinite(probs).all()
            np.savez_compressed(out / name, points=points, probabilities=probs,
                                truth_native=np.asarray(r['points'], dtype=float).reshape(-1, 2))
            files.append({'file': name, 'sha256': digest(out / name), 'image_id': r['image_id'],
                          'hard_count': int((probs >= .5).sum()), 'soft_count': float(probs.sum()),
                          'maximum_probability': float(probs.max()), 'transform': t})
        model.train(); return files

    def save(step):
        # Diagnostic checkpoint only. Never admitted as research initialization.
        payload = {'counter': {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith('encoder.')},
                   'optimizer': optimizer.state_dict(), 'step': step, 'seed': 1729,
                   'plan_sha256': report['plan_sha256'], 'research_initialization_allowed': False,
                   'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        tmp = out / 'diagnostic.pt.tmp'; torch.save(payload, tmp); tmp.replace(out / 'diagnostic.pt')
        report['saved_updates'] = step; report['checkpoint_sha256'] = digest(out / 'diagnostic.pt')
        write_json(out / 'status.json', report)

    report['initial_predictions'] = predictions('initial')
    write_json(out / 'status.json', report)
    start = time.perf_counter()
    with (out / 'updates.jsonl').open('x') as log:
        for step in range(1, 151):
            tick = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            frac = step / 15 if step <= 15 else .01 + .99 * (1 + math.cos(math.pi * (step - 15) / 135)) / 2
            lr = 2e-4 * frac
            for group in optimizer.param_groups: group['lr'] = lr
            bundle_values = []
            for bundle in range(2):
                outputs = forward()
                losses = [view_loss(r, g) for r, g in zip(outputs, truth)]
                objective = torch.stack([x['total'] for x in losses]).mean() / 2
                assert torch.isfinite(objective)
                objective.backward()
                bundle_values.append([{k: (float(v.detach()) if torch.is_tensor(v) else v) for k, v in x.items()} for x in losses])
                del outputs, losses, objective
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            assert all(p.grad is None for p in model.encoder.parameters())
            gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(gradnorm)
            optimizer.step(); report['optimizer_calls'] = step
            assert all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
            assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
            torch.cuda.synchronize()
            entry = {'step': step, 'lr': lr, 'gradient_norm_before_clip': float(gradnorm),
                     'bundles': bundle_values, 'query_presentations': 8, 'eligible_quartets': 0,
                     'seconds': time.perf_counter() - tick, 'elapsed_seconds': time.perf_counter() - start}
            log.write(json.dumps(entry, allow_nan=False) + '\n'); log.flush()
            print(json.dumps({'step': step, 'loss': sum(x['total'] for b in bundle_values for x in b) / 8,
                              'seconds': entry['seconds']}), flush=True)
            if step % 25 == 0: save(step)
    report['final_predictions'] = predictions('final')
    report['encoder_final_sha256'] = state_digest(model.encoder.state_dict())
    assert report['encoder_initial_sha256'] == report['encoder_final_sha256']
    report['head_final_sha256'] = state_digest({k: v for k, v in model.state_dict().items() if not k.startswith('encoder.')})
    assert report['head_initial_sha256'] != report['head_final_sha256']
    report.update(state='completed', query_presentations=1200, elapsed_seconds=time.perf_counter()-start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(out / 'status.json', report); write_json(out / 'report.json', report)
    hashes = {p.name: digest(p) for p in out.iterdir() if p.name.endswith(('.npz', '.pt')) or p.name in ('updates.jsonl', 'report.json')}
    write_json(out / 'result_hashes.json', hashes)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); ap.add_argument('--data', required=True); ap.add_argument('--assets', required=True)
    args = ap.parse_args()
    try: run(args)
    except Exception:
        write_json(Path(args.output) / 'failure.json', {'state': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__': main()
