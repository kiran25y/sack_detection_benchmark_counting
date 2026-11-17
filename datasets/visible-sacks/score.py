import argparse, json, math
from pathlib import Path
import numpy as np


def maximum_matches(pred, gt, tolerance=8.):
    adjacency = [np.flatnonzero(((pred - g) ** 2).sum(1) <= tolerance ** 2).tolist() for g in gt]
    owner = {}
    def visit(g, seen):
        for p in adjacency[g]:
            if p in seen: continue
            seen.add(p)
            if p not in owner or visit(owner[p], seen): owner[p] = g; return True
        return False
    return sum(visit(g, set()) for g in range(len(gt)))

def evaluate(pred, gt, row):
    n, target = len(pred), len(gt)
    tp = maximum_matches(pred * (1024 / max(row['width'], row['height'])), gt * (1024 / max(row['width'], row['height'])))
    roi = np.asarray(row['roi_xyxy']); wh = roi[2:] - roi[:2]
    games = {}
    for level in (1, 2):
        side = 2 ** level
        def bins(p):
            xy = np.clip(np.floor((p + .5 - roi[:2]) / wh * side), 0, side-1).astype(int)
            return np.bincount(xy[:, 1]*side + xy[:, 0], minlength=side*side)
        games['game' + str(level)] = int(abs(bins(pred) - bins(gt)).sum())
    return {'image_id': row['image_id'], 'group': row['dependency_group'], 'truth': target, 'count': n,
            'error': n-target, 'ae': abs(n-target), 'tp': tp, 'fp': n-tp, 'fn': target-tp,
            'f1': 2*tp/(n+target) if n+target else None, **games}

def summary(records):
    if not records:return {'frames':0,'mae':None,'f1_at8':None,'reason':'No qualifying reviewed frames'}
    n=len(records);tp=sum(r['tp'] for r in records);fp=sum(r['fp'] for r in records);fn=sum(r['fn'] for r in records)
    return dict(frames=n,truth=tp+fn,predictions=tp+fp,tp=tp,fp=fp,fn=fn,
                mae=sum(r['ae'] for r in records)/n,rmse=math.sqrt(sum(r['error']**2 for r in records)/n),bias=sum(r['error'] for r in records)/n,
                f1_at8=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,game1=sum(r['game1'] for r in records)/n,game2=sum(r['game2'] for r in records)/n,
                positive_frames_without_match=sum(r['truth']>0 and r['tp']==0 for r in records),empty_frames_with_fp=sum(r['truth']==0 and r['fp']>0 for r in records))


def main():
    parser = argparse.ArgumentParser(description="Score native-coordinate sack centers")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("train","development","test"), default="test")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    rows = json.loads((base/"frames.json").read_text(encoding="utf-8"))
    labels = json.loads((base/"annotations"/(args.split+".json")).read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if set(predictions) != set(labels):
        raise ValueError("Supply exactly one entry for every image ID in the requested split.")
    records = []
    for r in rows:
        if r["split"] != args.split:
            continue
        key = r["image_id"]
        pts = predictions[key]
        if not isinstance(pts, list):
            raise ValueError(f"{key}: expected a list of [x,y] centers")
        p = np.asarray(pts, dtype=float) if pts else np.empty((0,2))
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError(f"{key}: invalid points")
        gt = np.asarray(labels[key]["points"], dtype=float).reshape(-1,2)
        records.append(evaluate(p, gt, r))
    result = {"overall": summary(records),
              "low_1_to_11": summary([r for r in records if 1 <= r["truth"] <= 11]),
              "empty": summary([r for r in records if r["truth"] == 0]),
              "groups": {g: summary([r for r in records if r["group"] == g])
                         for g in sorted({r["group"] for r in records})}}
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
