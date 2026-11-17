"""Bounded paired-history point detection on frozen dense DINOv2 features."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from counting.firc.geometry import image_to_model, points_to_model, points_to_native, transform
from counting.firc.model import load_official_encoder
from counting.firc.preflight import configure
from counting.firc.tiny_fit import digest, state_digest, write_json


OUT = Path('/output')
STRIDE = 14


def array(points):
    return np.asarray(points, dtype=float).reshape(-1, 2)


def assign(seeds, truth, radius):
    """Maximum-cardinality, minimum-distance valid matching; unmatched IDs are -1."""
    owners = np.full(len(seeds), -1, dtype=int)
    if len(seeds) and len(truth):
        d = np.linalg.norm(seeds[:, None] - truth[None, :], axis=2)
        cost = np.where(d <= radius, d / radius, max(d.shape) + 1.)
        i, j = linear_sum_assignment(cost)
        valid = d[i, j] <= radius
        owners[i[valid]] = j[valid]
    return owners


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(768, 48, 1), nn.GroupNorm(6, 48), nn.SiLU())
        self.fuse = nn.Sequential(nn.Conv2d(51, 48, 3, padding=1), nn.GroupNorm(6, 48), nn.SiLU(),
                                  nn.Conv2d(48, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU())
        self.birth = nn.Conv2d(32, 3, 1)
        self.keep = nn.Sequential(nn.Linear(34, 24), nn.SiLU(), nn.Linear(24, 1))

    def forward(self, feature, views):
        device = feature.device
        image = self.image(feature.float())
        history = torch.as_tensor(np.stack([v['map'] for v in views]), device=device)
        hidden = self.fuse(torch.cat([image.expand(len(views), -1, -1, -1), history], 1))
        outputs = self.birth(hidden)
        keeps = []
        for field, v in zip(hidden, views):
            coords = torch.as_tensor(v['grid'], device=device, dtype=field.dtype)
            x, y = coords.long().T
            descriptors = torch.cat([field[:, y, x].T, coords - coords.floor()], dim=1)
            keeps.append(self.keep(descriptors).flatten())
        return outputs[:, 0], outputs[:, 1:].sigmoid(), keeps


def make_view(item, seeds, training):
    t, (h, w) = item['transform'], item['shape']
    grid = points_to_model(seeds, t) / STRIDE
    xy = grid.astype(int)
    history = np.zeros((3, h, w), dtype=np.float32)
    if len(grid):
        x, y = xy.T
        for channel, values in enumerate([np.ones(len(x)), grid[:, 0]-x, grid[:, 1]-y]):
            np.add.at(history[channel], (y, x), values)
        history[1:] /= np.maximum(history[0], 1)
    v = {'map': history, 'grid': grid, 'seeds': seeds}
    if training:
        truth = item['truth']
        owners = assign(seeds, truth, item['radius'])
        missing = np.ones(len(truth), dtype=bool)
        missing[owners[owners >= 0]] = False
        target = np.zeros((h, w), np.float32)
        offsets = np.zeros((2, h, w), np.float32)
        residual = points_to_model(truth[missing], t) / STRIDE
        if len(residual):
            x, y = residual.astype(int).T
            target[y, x] = 1
            offsets[:, y, x] = (residual - residual.astype(int)).T
        assert target.sum() == missing.sum()
        v.update(target=target, offsets=offsets, keep=(owners >= 0).astype(np.float32))
    return v


def histories(item, rng, step):
    truth, r = item['truth'], item['row']['roi_xyxy']
    real = item['seeds'] if step % 2 else array([])
    if not len(truth):
        # Both arms see false marks even on fitting frames with no sacks.
        decoy = rng.uniform(np.array(r[:2]), np.array(r[2:])-1, size=(1, 2))
        return [make_view(item, h, True) for h in (array([]), decoy, real)]
    target = rng.integers(len(truth))
    remaining = np.delete(truth, target, axis=0)
    base = remaining[rng.random(len(remaining)) < rng.uniform(0, .85)]
    # Prefer a plausible nearby miss. Reject every candidate within matching range of any sack.
    for attempt in range(1000):
        if attempt < 50:
            proposal = truth[target] + rng.normal(0, 3*item['radius'], 2)
        else:
            proposal = rng.uniform(r[:2], np.array(r[2:])-1)
        if (proposal >= r[:2]).all() and (proposal < np.array(r[2:])-.5).all() and np.linalg.norm(truth-proposal, axis=1).min() > item['radius']:
            break
    else:
        raise ValueError('No valid decoy found')
    pair = [np.vstack([base, truth[target]]), np.vstack([base, proposal])]
    return [make_view(item, h, True) for h in (*pair, real)]


def losses(model, item, views):
    logits, offsets, keeps = model(item['feature'], views)
    target = torch.as_tensor(np.stack([v['target'] for v in views]), device='cuda')
    gt_offsets = torch.as_tensor(np.stack([v['offsets'] for v in views]), device='cuda')
    positive = target.bool()
    valid = item['valid'].expand_as(positive)
    negative = valid & ~positive
    n = int(positive.sum())
    zero = logits.sum()*0
    pos_loss = F.softplus(-logits[positive]).mean() if n else zero
    hard_logits = logits[negative].topk(min(max(48, 3*n), int(negative.sum()))).values
    loc_loss = F.smooth_l1_loss(offsets.permute(0, 2, 3, 1)[positive],
                                gt_offsets.permute(0, 2, 3, 1)[positive], beta=.1) if n else zero
    keep_logits = torch.cat(keeps)
    keep_labels = torch.as_tensor(np.concatenate([v['keep'] for v in views]), device='cuda')
    keep_loss = F.binary_cross_entropy_with_logits(keep_logits, keep_labels) if len(keep_labels) else zero
    ordinary = pos_loss + F.softplus(hard_logits).mean() + 2*loc_loss + keep_loss
    delta = logits[1].sigmoid() - logits[0].sigmoid()
    expected = target[1] - target[0]
    changed = expected.ne(0)
    pair = F.smooth_l1_loss(delta[changed], expected[changed], beta=.25) if changed.any() else zero
    same = item['valid'][0] & ~changed
    # Penalize changes elsewhere, including the cell containing a moved background mark.
    background = delta[same].abs()
    pair = pair + background.topk(min(16, len(background))).values.mean()
    return ordinary, pair


@torch.no_grad()
def proposals(model, item):
    view = make_view(item, item['seeds'], False)
    logits, offsets, keeps = model(item['feature'], [view])
    keep = item['seeds'][keeps[0].sigmoid().cpu().numpy() >= .5]
    scores = logits[0].sigmoid().cpu().numpy()
    off = offsets[0].permute(1, 2, 0).cpu().numpy()
    y, x = np.nonzero((scores >= .05) & item['valid'][0].cpu().numpy())
    centers = (np.column_stack([x, y]) + off[y, x])*STRIDE
    good = (centers < item['transform']['valid_hw'][::-1]).all(axis=1)
    order = np.argsort(-scores[y, x][good], kind='stable')
    native = points_to_native(centers[good][order], item['transform'])
    return keep, native, scores[y, x][good][order]


def suppress(raw, radius):
    keep, births, scores = raw
    chosen = list(keep)
    accepted = []
    for pt, score in zip(births, scores):
        if not chosen or np.linalg.norm(array(chosen)-pt, axis=1).min() > radius:
            chosen.append(pt)
            accepted.append(float(score))
    return array(chosen), np.array([2.]*len(keep) + accepted)


def calibrate(model, train, thresholds):
    raw = [suppress(proposals(model, it), it['radius']) for it in train]
    rows = []
    for tau in thresholds:
        empty, f1, ae = 0, [], []
        for (points, scores), it in zip(raw, train):
            pred, gt = points[scores >= tau], it['truth']
            tp = int((assign(pred, gt, it['radius']) >= 0).sum())
            ae.append(abs(len(pred)-len(gt)))
            if len(gt):
                f1.append(2*tp/(len(pred)+len(gt)))
            else:
                empty += len(pred)
        rows.append(dict(threshold=tau, empty_fp=empty, positive_f1=float(np.mean(f1)), mae=float(np.mean(ae))))
    best = min(rows, key=lambda r: (r['empty_fp'], -r['positive_f1'], r['mae'], -r['threshold']))
    return best, rows


def main():
    configure()
    plan = json.loads(Path('/job/plan.json').read_text())
    ck = Path('/assets/dinov2_vitb14_reg4_pretrain.pth')
    assert digest(ck) == plan['weight_hash']
    encoder = load_official_encoder(ck).requires_grad_(False).cuda().eval()
    encoder_hash = state_digest(encoder.state_dict())
    assert encoder_hash == plan['encoder_hash']
    items = []
    for row in plan['rows']:
        path = Path('/data')/row['image']
        assert digest(path) == row['encoded_sha256']
        with Image.open(path) as im:
            assert im.size == (row['width'], row['height'])
            t = transform(*im.size, 896, row['roi_xyxy'])
            x = torch.from_numpy(image_to_model(im, t))[None].cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            feat = encoder.get_intermediate_layers(x, n=[11], reshape=True, return_class_token=False, norm=True)[0].half()
        h, w = feat.shape[-2:]
        yy, xx = np.ogrid[:h, :w]
        valid = (yy*STRIDE < t['valid_hw'][0]) & (xx*STRIDE < t['valid_hw'][1])
        it = dict(row=row, feature=feat, shape=(h, w), valid=torch.tensor(valid,device='cuda')[None],
                  transform=t, seeds=array(row['seeds']), radius=8*max(im.size)/1024)
        if row['split'] == 'train':
            it['truth'] = array(row['points'])
        else:
            assert 'points' not in row
        items.append(it)
    assert state_digest(encoder.state_dict()) == encoder_hash
    del encoder
    train = [it for it in items if 'truth' in it]
    assert len(train) == 42 and len(items) == 65
    rng = np.random.default_rng(1729)
    schedule = np.concatenate([rng.permutation(len(train)) for _ in range(10)])
    start_hashes, arms = [], {}
    for arm in ('ordinary', 'paired'):
        torch.manual_seed(1729)
        rng = np.random.default_rng(1947)
        model = Detector().cuda()
        start_hashes.append(state_digest(model.state_dict()))
        opt = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
        with (OUT/(arm+'_updates.jsonl')).open('x') as log:
            for step, ix in enumerate(schedule, 1):
                it = train[ix]
                views = histories(it, rng, step)
                base, pair = losses(model, it, views)
                loss = base + (0.5 if arm == 'paired' else 0.)*pair
                opt.zero_grad()
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                log.write(json.dumps(dict(step=step, image=it['row']['image_id'], ordinary=base.item(), pair=pair.item(), grad=norm.item()), allow_nan=False)+'\n')
                if step % 42 == 0:
                    log.flush()
                    write_json(OUT/'status.json', dict(arm=arm, step=step, total=len(schedule)))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert all(torch.isfinite(v).all() for state in opt.state.values() for v in state.values() if torch.is_tensor(v))
        model.eval()
        best, curve = calibrate(model, train, plan['thresholds'])
        path = OUT/(arm+'_head.pt')
        torch.save(model.state_dict(), path)
        arms[arm] = dict(best=best, calibration=curve, steps=len(schedule), head_sha256=digest(path),
                         parameters=sum(p.numel() for p in model.parameters()))
        write_json(OUT/(arm+'_freeze.json'), arms[arm])
    assert start_hashes[0] == start_hashes[1]
    # Both thresholds and heads have been persisted before any development decoding.
    predictions = []
    for arm, info in arms.items():
        model.load_state_dict(torch.load(OUT/(arm+'_head.pt'), map_location='cuda', weights_only=True))
        for it in items:
            points, scores = suppress(proposals(model, it), it['radius'])
            passed = scores >= info['best']['threshold']
            predictions.append(dict(arm=arm, image_id=it['row']['image_id'], points=points[passed].tolist(),
                                    retained=int((scores[passed] > 1).sum())))
    write_json(OUT/'predictions.json', predictions)
    report = dict(state='completed', encoder_sha256=encoder_hash, feature_forwards=len(items),
                  plan_sha256=digest(Path('/job/plan.json')), initial_head=start_hashes, arms=arms)
    report['output_hashes'] = {p.name:digest(p) for p in OUT.iterdir() if p.name != 'status.json'}
    write_json(OUT/'report.json', report)


if __name__ == '__main__':
    main()
