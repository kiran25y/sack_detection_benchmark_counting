"""Matched sack-only offset supervision on local frozen DINOv2 features."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from counting.firc.geometry import image_to_model, transform
from counting.firc.model import load_official_encoder
from counting.firc.preflight import configure
from counting.firc.tiny_fit import digest, state_digest, write_json


OUT = Path('/output')
JOB = Path('/job')
GRID = np.stack(np.meshgrid(np.arange(-16, 17, 8), np.arange(-16, 17, 8)), axis=-1)


def array(x):
    return np.asarray(x, dtype=float).reshape(-1, 2)


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(768, 32, 1), nn.GroupNorm(4, 32), nn.SiLU(),
                                      nn.Conv2d(32, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU())
        self.score = nn.Linear(48, 1)
        self.heat = nn.Conv2d(24, 1, 1)
        self.register_buffer('grid', torch.tensor(GRID.reshape(-1, 2), dtype=torch.float32))

    def forward(self, x):
        z = self.features(x)
        logit = self.score(torch.cat((z.mean((2, 3)), z.amax((2, 3))), dim=1))[:, 0]
        heat = self.heat(z).flatten(1).masked_fill(self.grid.square().sum(1) > 16**2, -1e4)
        return logit, heat.softmax(1) @ self.grid


@torch.no_grad()
def encode(encoder, row):
    path = Path(row['path'])
    assert digest(path) == row['sha256']
    with Image.open(path) as image:
        assert list(image.size) == row['size']
        t = transform(*image.size, 896, row['roi'])
        x = torch.from_numpy(image_to_model(image, t))[None].cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        feature = encoder.get_intermediate_layers(x, n=[11], reshape=True, norm=True)[0].half()
    assert torch.isfinite(feature).all()
    return dict(**row, t=t, feature=feature, unit=max(row['size'])/1024)


def patches(item, xy):
    t = item['t']
    local = xy[:, None, None, :] + GRID[None] * item['unit']
    model = (local + .5 - np.array(t['roi'][:2])) * t['scale_xy']
    model = np.clip(model, .5, np.array(t['valid_hw'][::-1]) - .5)
    grid = torch.tensor(2*model/np.array(t['padded_hw'][::-1])-1, dtype=torch.float32, device='cuda')
    z = F.grid_sample(item['feature'].float(), grid.reshape(1, -1, 5, 2),
                      mode='bilinear', padding_mode='border', align_corners=False)
    return z[0].reshape(768, len(xy), 5, 5).permute(1, 0, 2, 3).contiguous()


def sample(item, rng):
    truth = array(item['points'])
    roi = np.array(item['roi'])
    xy = rng.uniform(roi[:2], roi[2:]-1, size=(64, 2))
    offset = np.zeros_like(xy)
    positive = np.zeros(len(xy), bool)
    if len(truth):
        # Thirty-two perturbations within reach, sixteen on the surrounding ring.
        anchors = truth[rng.integers(len(truth), size=48)]
        angle = rng.uniform(0, 2*np.pi, 48)
        distance = np.r_[16*np.sqrt(rng.random(32)), rng.uniform(16, 40, 16)]
        xy[:48] = np.clip(anchors + np.c_[np.cos(angle), np.sin(angle)] *
                          distance[:, None]*item['unit'], roi[:2], roi[2:]-1)
        delta = (truth[None] - xy[:, None])/item['unit']
        d = np.linalg.norm(delta, axis=2)
        nearest = d.argmin(axis=1)
        offset = delta[np.arange(len(xy)), nearest]
        positive = np.linalg.norm(offset, axis=1) <= 16
    return patches(item, xy), torch.as_tensor(positive, device='cuda'), torch.tensor(offset, dtype=torch.float32, device='cuda')


@torch.no_grad()
def infer(model, item, points):
    xy = array(points)
    if not len(xy):
        return dict(scores=[], shifted=[])
    logit, offset = model(patches(item, xy))
    corrected = np.clip(xy + offset.cpu().numpy()*item['unit'], np.array(item['roi'][:2])-.5,
                        np.array(item['roi'][2:])-.5)
    assert np.isfinite(corrected).all() and torch.isfinite(logit).all()
    assert np.max(np.linalg.norm(corrected-xy, axis=1))/item['unit'] <= 16.001
    return dict(scores=logit.sigmoid().tolist(), shifted=corrected.tolist())


def select(raw, prediction, use_offsets, threshold, unit, base):
    chosen = list(base)
    indices = []
    for i, score in enumerate(prediction['scores']):
        if score < threshold:
            continue
        point = prediction['shifted'][i] if use_offsets else raw[i]
        if chosen and np.min(np.linalg.norm(np.asarray(chosen)-point, axis=1)) <= 8*unit:
            continue
        indices.append(i)
        chosen.append(point)
    return chosen, indices


def calibrate(train, bank, predictions, use_offsets):
    empty_scores = [s for it in train if not it['points'] for s in predictions[it['id']]['scores']]
    floor = np.nextafter(max(empty_scores), np.inf)
    curve = []
    for threshold in sorted(set([float(floor), 1.000001] + [i/20 for i in range(1, 20) if i/20 >= floor])):
        groups = {}
        for it in train:
            k = it['id']
            xy, _ = select(bank[k], predictions[k], use_offsets, threshold, it['unit'], [])
            truth = array(it['points'])
            if not len(truth):
                assert not xy
                continue
            tp = 0
            if xy:
                dist = np.linalg.norm(array(xy)[:, None]-truth[None], axis=2)/(8*it['unit'])
                a, b = linear_sum_assignment(np.where(dist <= 1, dist, max(dist.shape)+1))
                tp = int((dist[a, b] <= 1).sum())
            groups.setdefault(it['group'], []).append(1.25*tp/(len(xy)+.25*len(truth)))
        curve.append(dict(threshold=threshold, macro_f05=float(np.mean([np.mean(v) for v in groups.values()]))))
    return dict(best=max(curve, key=lambda v:(v['macro_f05'], v['threshold'])), curve=curve, empty_floor=floor)


def main():
    configure()
    plan = json.loads((JOB/'plan.json').read_text())
    bank = json.loads((JOB/'banks.json').read_text())
    additions = json.loads((JOB/'candidates.json').read_text())
    weight = Path('/assets/dinov2_vitb14_reg4_pretrain.pth')
    assert digest(weight) == plan['encoder_file']
    encoder = load_official_encoder(weight).requires_grad_(False).cuda().eval()
    assert state_digest(encoder.state_dict()) == plan['encoder_state']
    train = [encode(encoder, r) for r in plan['rows'] if r['split']=='train']
    evaluations = [r for r in plan['rows'] if r['split']!='train']
    assert set(bank)=={r['id'] for r in train} and set(additions)=={r['id'] for r in evaluations}
    assert all('points' not in r for r in evaluations)
    groups = sorted({r['group'] for r in train})
    arms = plan['arms']
    oof = {arm:{} for arm in arms}
    receipts = []
    steps = plan['steps']
    with (OUT/'updates.jsonl').open('x') as log:
        for fold, excluded in enumerate([*groups, None]):
            source = [r for r in train if r['group']!=excluded]
            torch.manual_seed(plan['seed'])
            heads = {arms[0]:Refiner().cuda()}
            heads[arms[1]] = copy.deepcopy(heads[arms[0]])
            opt = {k:torch.optim.AdamW(h.parameters(), lr=.0005, weight_decay=.01) for k,h in heads.items()}
            rng = np.random.default_rng(plan['seed'])
            draws = np.concatenate([rng.permutation(len(source)) for _ in range(steps//len(source)+1)])[:steps]
            record = dict(fold=fold, excluded=excluded, fit_ids=[r['id'] for r in source],
                          initial=state_digest(heads[arms[0]].state_dict()), steps_per_arm=steps, checkpoints={})
            for step, j in enumerate(draws, 1):
                item = source[j]
                inputs, pos, offset = sample(item, rng)
                losses = {}
                for arm, head in heads.items():
                    logits, predicted = head(inputs)
                    components = []
                    if pos.any(): components.append(F.softplus(-logits[pos]).mean())
                    if (~pos).any(): components.append(F.softplus(logits[~pos]).mean())
                    score_loss = torch.stack(components).mean()
                    position_loss = F.smooth_l1_loss(predicted[pos]/8, offset[pos]/8) if pos.any() else predicted.sum()*0
                    loss = score_loss + (arm=='refined')*position_loss
                    opt[arm].zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                    opt[arm].step()
                    losses[arm] = loss.item()
                log.write(json.dumps(dict(fold=fold, step=step, frame=item['id'], losses=losses), allow_nan=False)+'\n')
                if step==1 or step%50==0:
                    log.flush()
                    write_json(OUT/'status.json',dict(state='training', fold=fold, excluded=excluded,
                               steps_per_arm=step, optimizer_calls=2*(fold*steps+step)))
            for arm, head in heads.items():
                head.eval()
                assert all(torch.isfinite(p).all() for p in head.parameters())
                name = f'fold{fold}_{arm}.pt'
                torch.save(head.state_dict(), OUT/name)
                record['checkpoints'][arm] = digest(OUT/name)
                for it in train:
                    if excluded==it['group']:
                        oof[arm][it['id']] = infer(head, it, bank[it['id']])
            receipts.append(record)
    write_json(OUT/'oof.json', oof)
    calibration = {a:calibrate(train, bank, oof[a], a=='refined') for a in arms}
    write_json(OUT/'freeze.json', dict(calibration=calibration, fits=receipts))
    sealed = {}
    for row in evaluations:
        it = encode(encoder, row)
        k = row['id']
        base, raw = additions[k]['base'], additions[k]['extra']
        sets = dict(verifier=base, joint=base+raw)
        scores = {}
        for a in arms:
            scores[a] = infer(heads[a], it, raw)
            points, ix = select(raw, scores[a], a=='refined', calibration[a]['best']['threshold'], it['unit'], base)
            sets[a] = points
            if a=='refined':
                sets['same_selected_unmoved'] = base + [raw[i] for i in ix]
                sets['equal_budget_raw'] = base + raw[:len(ix)]
        sealed[k] = dict(predictions=sets, estimates=scores)
    write_json(OUT/'predictions.json', sealed)
    assert state_digest(encoder.state_dict())==plan['encoder_state']
    hashes = {p.name:digest(p) for p in OUT.iterdir() if p.name!='status.json'}
    write_json(OUT/'report.json', dict(state='completed', optimizer_calls=2*len(receipts)*steps, images=len(train)+len(sealed),
               plan_sha256=digest(JOB/'plan.json'), encoder=plan['encoder_state'], hashes=hashes))
    write_json(OUT/'status.json',dict(state='completed', optimizer_calls=2*len(receipts)*steps))


if __name__ == '__main__':
    main()
