"""Fixed-count localization ablation on the saved 71-match control."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

from analyze_firc_tiny import evaluate, maximum_matches, sha, summarize
from audit_firc_countr_misses import candidates

ROOT = Path(__file__).resolve().parents[1]
FIRC = ROOT / 'artifacts/firc_ordinary_evaluation_v1_20261106'
PRIOR = ROOT / 'artifacts/firc_countr_local_support_v1_20261108'
OUT = ROOT / 'artifacts/firc_center_voting_v1_20261109'
PROTOCOL = ROOT / 'study_protocols/firc_center_voting_20261109.json'
CONTROL = 'equal_number_point_only'
PRIMARY = 'confidence_vote'
ARMS = ('unchanged', 'uniform', PRIMARY, 'opposite')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def training_spacing(training):
    frame_medians = {}
    for row in training:
        assert row['split'] == 'train'
        points = np.asarray(row['points'], dtype=float).reshape(-1, 2)
        if len(points) < 2:
            continue
        points *= 1024 / max(row['width'], row['height'])
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        frame_medians[row['image_id']] = float(np.median(distances.min(axis=1)))
    radius = .5 * float(np.median(list(frame_medians.values())))
    assert np.isfinite(radius) and radius > 0
    return radius, frame_medians


def bounded_vectors(seeds, vectors, lower, upper):
    norms = np.linalg.norm(vectors, axis=1)
    factor = np.minimum(1., np.divide(8., norms, out=np.full_like(norms, np.inf), where=norms > 0))
    available = np.maximum(0., np.minimum(seeds-lower, upper-seeds))
    magnitude = np.abs(vectors)
    bounds = np.divide(available, magnitude, out=np.full_like(vectors, np.inf), where=magnitude > 0)
    factor = np.minimum(factor, bounds.min(axis=1))
    return vectors * factor[:, None]


def refinements(seeds_native, xy_native, q, scale, roi, radius):
    if not len(seeds_native):
        return {method: seeds_native.copy() for method in ARMS}, []
    seeds = seeds_native * scale
    xy = xy_native * scale
    owner = np.full(len(q), -1, dtype=int)
    best_squared = np.full(len(q), np.inf)
    for index, center in enumerate(seeds):
        squared = np.square(xy-center).sum(axis=1)
        closer = (squared <= radius**2) & (squared < best_squared)
        owner[closer] = index
        best_squared[closer] = squared[closer]
    assigned = owner >= 0
    index = owner[assigned]
    offsets = xy[assigned] - seeds[index]
    votes = np.bincount(index, minlength=len(seeds))
    bounds = (np.asarray(roi, dtype=float) - .5) * scale

    def centroid_shift(weights):
        total = np.bincount(index, weights=weights, minlength=len(seeds))
        sums = np.column_stack([np.bincount(index, weights=weights*offsets[:,axis],
                                            minlength=len(seeds)) for axis in (0,1)])
        vectors = np.divide(sums, total[:,None], out=np.zeros_like(sums), where=total[:,None] > 0)
        return bounded_vectors(seeds, vectors, bounds[:2], bounds[2:])

    uniform = centroid_shift(np.ones(len(index)))
    weighted = centroid_shift(q[assigned].astype(float))
    points = {'unchanged':seeds_native, 'uniform':seeds_native+uniform/scale,
              PRIMARY:seeds_native+weighted/scale, 'opposite':seeds_native-weighted/scale}
    for value in points.values():
        assert value.shape == seeds_native.shape and np.isfinite(value).all()
        assert (value*scale >= bounds[:2]-1e-9).all() and (value*scale <= bounds[2:]+1e-9).all()
        assert (np.linalg.norm((value-seeds_native)*scale,axis=1) <= 8+1e-9).all()
    return points, votes.tolist()


def score_split(records):
    result = summarize(records)
    result.update(frames=len(records), positive_frames_with_tp=sum(r['tp']>0 for r in records),
                  targets=sum(r['truth'] for r in records), predictions=sum(r['count'] for r in records))
    for radius in (4,16):
        result[f'tp_at{radius}'] = sum(r[f'tp_at{radius}'] for r in records)
    return result


def main():
    old = read(PRIOR/'result.json')
    assert sha(PRIOR/'result.json') == read(PRIOR/'verification.json')['result_sha256']
    assert all(sha(PRIOR/name) == expected for name,expected in old['hashes'].items())
    plan = read(FIRC/'data/plan.json')
    label_path = ROOT/plan['local_truth_manifest']
    assert sha(label_path) == plan['truth_manifest_sha256']
    labels = {r['image_id']:r for r in read(label_path)}
    rows = plan['rows']
    assert len(rows)==len(labels)==65
    assert all('points' not in r and 'count' not in r for r in rows)
    entries = {e['image_id']:e for e in read(FIRC/'collected/result/report.json')['predictions']}
    with (PRIOR/'points.jsonl').open() as f:
        seeds = {v['image_id']:v for v in map(json.loads,f) if v['method']==CONTROL}
    assert set(seeds)==set(labels)==set(entries)=={r['image_id'] for r in rows}
    train = [labels[r['image_id']] for r in rows if r['split']=='train']
    assert len(train)==42
    radius, medians = training_spacing(train)
    OUT.mkdir(exist_ok=False)
    sources = [PROTOCOL, Path(__file__), PRIOR/'points.jsonl', PRIOR/'result.json', FIRC/'data/plan.json',
               FIRC/'collected/result/report.json', label_path,
               ROOT/'scripts/analyze_firc_tiny.py', ROOT/'scripts/audit_firc_countr_misses.py']
    write(OUT/'freeze.json', {'time_kst':now(),'primary':PRIMARY,'training_radius':radius,
          'training_spacing_by_frame':medians,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}})
    with (OUT/'predictions.jsonl').open('x') as f:
        for r in rows:
            key = r['image_id']
            xy,q = candidates(entries[key])
            seed = seeds[key]
            ids = seed['original_ids']+seed['added_ids']
            coords = np.asarray(seed['native_points'],float).reshape(-1,2)
            assert np.array_equal(xy[np.asarray(ids,int)], coords)
            scale = 1024/max(r['width'],r['height'])
            positions,votes = refinements(coords,xy,q,scale,r['roi_xyxy'],radius)
            f.write(json.dumps({'image_id':key,'split':r['split'],'seed_ids':ids,'votes':votes,
                                'native_points':{m:v.tolist() for m,v in positions.items()}}, allow_nan=False)+'\n')
    prediction_hash = sha(OUT/'predictions.jsonl')
    write(OUT/'prediction_receipt.json',{'time_kst':now(),'predictions_sha256':prediction_hash,
                                        'frames':65,'modes':list(ARMS)})
    records = {m:[] for m in ARMS}
    moves = {m:[] for m in ARMS}
    with (OUT/'predictions.jsonl').open() as f:
        for pred in map(json.loads,f):
            r = labels[pred['image_id']]
            gt = np.asarray(r['points'],float).reshape(-1,2)
            scale = 1024/max(r['width'],r['height'])
            base = np.asarray(pred['native_points']['unchanged'],float).reshape(-1,2)
            for m,points in pred['native_points'].items():
                xy = np.asarray(points,float).reshape(-1,2)
                score = evaluate(xy,gt,r)
                score.update(split=r['split'],tp_at4=maximum_matches(xy*scale,gt*scale,4),
                             tp_at16=maximum_matches(xy*scale,gt*scale,16))
                records[m].append(score)
                if r['split']=='development':
                    moves[m].extend(np.linalg.norm((xy-base)*scale,axis=1))
    previous = {r['image_id']:r for r in old['per_image'][CONTROL]}
    assert all(all(r[k]==v for k,v in previous[r['image_id']].items()) for r in records['unchanged'])
    summaries={}
    for m,items in records.items():
        train_items=[r for r in items if r['split']=='train']
        dev=[r for r in items if r['split']=='development']
        distance=np.asarray(moves[m])
        summaries[m]={'training':score_split(train_items),'development':score_split(dev),
                      'low':score_split([r for r in dev if 1<=r['truth']<=11]),
                      'higher':score_split([r for r in dev if r['truth']>=12]),
                      'groups':{g:score_split([r for r in dev if r['group']==g]) for g in sorted({r['group'] for r in dev})},
                      'movement_chart_px':{'median':float(np.median(distance)),'maximum':float(distance.max())}}
    primary=summaries[PRIMARY];control=summaries['unchanged']
    def f1(method):
        return summaries[method]['development']['pooled_f1_at8']
    def grid_protection(section,metric):
        start=control[section][metric]
        return primary[section][metric]<=start+max(.1,.01*start)+1e-12
    protection={'all_frame_counts':all([r['count'] for r in rr]==[r['count'] for r in records['unchanged']] for rr in records.values()),
                'empty_fp':primary['development']['empty_false_positives']<=control['development']['empty_false_positives'],
                'low_mae':primary['low']['mae']<=control['low']['mae'],
                'low_f1':primary['low']['positive_macro_f1_at8']>=control['low']['positive_macro_f1_at8']}
    for section in ('low','development'):
        for metric in ('game1','game2'):
            protection[section+'_'+metric]=grid_protection(section,metric)
    placement={'gain_at8':f1(PRIMARY)-f1('unchanged')>=.02-1e-12,
               'both_groups':all(primary['groups'][g]['tp']>=control['groups'][g]['tp'] for g in control['groups'])}
    attribution={m:f1(PRIMARY)-f1(m)>=.02-1e-12 for m in ('uniform','opposite')}
    write(OUT/'result.json',{'completed_kst':now(),'predictions_sha256':prediction_hash,'summary':summaries,
          'per_image':records,'placement':placement,'attribution':attribution,'protection':protection,
          'placement_passed':all(placement.values()) and all(protection.values()),
          'specific_confidence_mechanism_passed':all(placement.values()) and all(protection.values()) and all(attribution.values()),
          'reserved_scored':0,'new_model_forwards':0,'optimizer_updates':0})
    print('R',radius)
    for m,s in summaries.items():
        d=s['development']
        print(m,{k:d[k] for k in ['tp','fp','fn','tp_at4','tp_at16','mae','pooled_f1_at8','positive_macro_f1_at8','game1','game2']},
              [g['tp'] for g in s['groups'].values()],s['movement_chart_px'])
    print('placement',placement,'specific weighting',attribution,'protection',protection)

if __name__=='__main__':
    main()
