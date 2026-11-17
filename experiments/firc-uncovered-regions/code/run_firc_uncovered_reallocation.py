"""One count-preserving coverage experiment on frozen FIRC/CounTR endpoints."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_firc_tiny import evaluate, maximum_matches, sha
from audit_firc_countr_misses import FIRC, COUNTR, DATA, ROOT, candidates, mass, read
from firc_countr_support import support_image
from run_firc_center_voting import OUT as CENTER, now, score_split, write

OUT = ROOT / 'artifacts/firc_uncovered_reallocation_v1_20261109'
PLAN = ROOT / 'study_protocols/firc_uncovered_reallocation_20261109.json'
PRIMARY = 'joint_coverage'
BASELINES = ('confidence_71','centering_75')
MATCHED = ('same_donors_rank_destinations','same_destinations_rank_donors','rank_both')
RADIUS = 8.
SEPARATION = 2 * RADIUS


def destination_bank(sites, scores, ids, seeds):
    """Shared, nonoverlapping destinations in descending FIRC confidence order."""
    allowed = np.ones(len(sites), dtype=bool)
    for seed in seeds:
        allowed &= np.square(sites-seed).sum(axis=1) > SEPARATION**2
    order = np.flatnonzero(allowed)
    order = order[np.lexsort((ids[order], -scores[order]))]
    retained = []
    cells = defaultdict(list)
    for i in order:
        x,y = np.floor(sites[i]/SEPARATION).astype(int)
        neighbors = [j for a in (x-1,x,x+1) for b in (y-1,y,y+1) for j in cells.get((a,b),())]
        if neighbors and (np.square(sites[neighbors]-sites[i]).sum(axis=1)<=SEPARATION**2).any():
            continue
        cells[x,y].append(i)
        retained.append(i)
    return np.asarray(retained,dtype=int)


def kernel(sites, center):
    return np.maximum(0.,1.-np.square(sites-center).sum(axis=1)/RADIUS**2)


def best_exchanges(sites, seeds, dest, weights):
    if not len(dest) or not len(seeds) or weights.sum()==0:
        return []
    weights=weights/weights.sum()
    cover=np.stack([kernel(sites,center) for center in seeds])
    available=np.ones(len(seeds),dtype=bool)
    gains=np.array([np.sum(weights*kernel(sites,sites[i])) for i in dest])
    moves=[]
    for _ in range(min(len(seeds),len(dest))):
        first=cover.argmax(axis=0)
        peak=cover.max(axis=0)
        second=np.partition(cover,-2,axis=0)[-2] if len(seeds)>1 else np.zeros_like(peak)
        losses=np.bincount(first,weights=weights*(peak-second),minlength=len(seeds))
        losses[~available]=np.inf
        donor=int(losses.argmin())
        receiver=int(gains.argmax())
        if gains[receiver]-losses[donor]<=1e-12: break
        moves.append({'slot':donor,'bank_index':receiver,'gain':float(gains[receiver]),'loss':float(losses[donor])})
        cover[donor]=0.
        gains[receiver]=-np.inf
        available[donor]=False
    return moves


def decode(row, entry, start):
    xy,q=candidates(entry)
    seed_ids=np.array(start['seed_ids'],dtype=int)
    old=np.asarray(start['native_points']['unchanged'],float).reshape(-1,2)
    centered=np.asarray(start['native_points']['confidence_vote'],float).reshape(-1,2)
    assert np.array_equal(xy[seed_ids],old) and np.isfinite(q).all()
    assert np.isfinite(centered).all() and ((q>=0)&(q<=1)).all()
    scale=1024/max(row['width'],row['height'])
    ids=np.flatnonzero(q>=.025)
    sites=xy[ids]*scale
    seed=centered*scale
    dest=destination_bank(sites,q[ids],ids,seed) if len(seed) else np.empty(0,int)
    density=mass(row)
    with np.load(DATA/row['array'],allow_pickle=False) as arr:
        mask=arr['mask'].astype(bool)
    local=support_image(density,mask,xy[ids],row['width'],row['height'])
    joint_weights=np.sqrt(q[ids].astype(float)*local)
    if not np.isfinite(joint_weights).all(): raise ValueError('Invalid coverage evidence')
    joint=best_exchanges(sites,seed,dest,joint_weights)
    firc_only=best_exchanges(sites,seed,dest,q[ids].astype(float))
    donor_slots=[m['slot'] for m in joint]
    primary_receivers=[m['bank_index'] for m in joint]
    n=len(joint)
    lower_confidence=np.lexsort((np.arange(len(seed)),q[seed_ids]))[:n].tolist()
    higher_confidence=list(range(n))
    rules={PRIMARY:(donor_slots,primary_receivers),
           'firc_coverage_only':([m['slot'] for m in firc_only],[m['bank_index'] for m in firc_only]),
           MATCHED[0]:(donor_slots,higher_confidence), MATCHED[1]:(lower_confidence,primary_receivers),
           MATCHED[2]:(lower_confidence,higher_confidence)}
    arms={BASELINES[0]:{'native_points':old.tolist(),'moves':[]},BASELINES[1]:{'native_points':centered.tolist(),'moves':[]}}
    for method,(donors,receivers) in rules.items():
        points=centered.copy()
        ledger=[]
        for number,(slot,bank_id) in enumerate(zip(donors,receivers)):
            candidate=int(ids[dest[bank_id]])
            position=xy[candidate]
            clearance=float(np.min(np.linalg.norm((centered-position)*scale,axis=1)))
            displacement=float(np.linalg.norm((centered[slot]-position)*scale))
            assert clearance>SEPARATION-1e-9
            other=np.delete(points,slot,axis=0)
            near=float(np.linalg.norm((other-centered[slot])*scale,axis=1).min()) if len(other) else None
            change={'slot':int(slot),'receiver_candidate':candidate,'distance_from_start_set':clearance,
                    'move_length':displacement,'nearest_other_point_to_donor':near}
            if method in (PRIMARY,'firc_coverage_only'):
                choice=(joint if method==PRIMARY else firc_only)[number]
                change.update(coverage_gain=choice['gain'],coverage_loss=choice['loss'])
            points[slot]=position
            ledger.append(change)
        assert len(donors)==len(receivers)==len(set(donors))==len(set(receivers))
        arms[method]={'native_points':points.tolist(),'moves':ledger}
    return {'image_id':row['image_id'],'split':row['split'],'seed_ids':seed_ids.tolist(),'sites':len(sites),'destination_bank_size':len(dest),'arms':arms}


def summaries(records):
    dev=[r for r in records if r['split']=='development']
    groups=sorted({r['group'] for r in dev})
    return {'training':score_split([r for r in records if r['split']=='train']),
            'development':score_split(dev),'low':score_split([r for r in dev if 1<=r['truth']<=11]),
            'higher':score_split([r for r in dev if r['truth']>=12]),
            'groups':{g:score_split([r for r in dev if r['group']==g]) for g in groups}}


def score_predictions(records_file,labels):
    records=defaultdict(list)
    audit=defaultdict(list)
    for pred in map(json.loads,records_file):
        key=pred['image_id'];row=labels[key]
        scale=1024/max(row['width'],row['height'])
        gt=np.asarray(row['points'],float).reshape(-1,2)
        centered=np.array(pred['arms'][BASELINES[1]]['native_points']).reshape(-1,2)
        for method,arm in pred['arms'].items():
            pts=np.asarray(arm['native_points'],float).reshape(-1,2)
            result=evaluate(pts,gt,row)
            result.update(split=row['split'],tp_at4=maximum_matches(pts*scale,gt*scale,4),tp_at16=maximum_matches(pts*scale,gt*scale,16))
            records[method].append(result)
            if method!=PRIMARY:continue
            current=centered.copy()
            previous=maximum_matches(current*scale,gt*scale,8)
            for move in arm['moves']:
                slot=move['slot']
                old_distance=float(np.linalg.norm((gt-current[slot])*scale,axis=1).min()) if len(gt) else None
                retained=maximum_matches(np.delete(current,slot,axis=0)*scale,gt*scale,8)
                current[slot]=pts[slot]
                after=maximum_matches(current*scale,gt*scale,8)
                new_distance=float(np.linalg.norm((gt-current[slot])*scale,axis=1).min()) if len(gt) else None
                audit[row['split']].append({'image_id':key,'group':row['dependency_group'],**move,
                    'lost_match_on_removal':previous-retained,'new_match_on_insertion':after-retained,
                    'net_matches':after-previous,'old_distance_to_nearest_label':old_distance,'new_distance_to_nearest_label':new_distance})
                previous=after
            assert previous==result['tp']
    return records,audit


def main():
    plan=read(FIRC/'data/plan.json')
    truth_path=ROOT/plan['local_truth_manifest']
    assert sha(truth_path)==plan['truth_manifest_sha256']
    previous=read(CENTER/'result.json')
    assert sha(CENTER/'result.json')==read(CENTER/'verification.json')['result_sha256']
    assert sha(CENTER/'predictions.jsonl')==previous['predictions_sha256']
    sources={r['image_id']:r for r in map(json.loads,(CENTER/'predictions.jsonl').read_text().splitlines())}
    inputs={r['image_id']:r for r in read(DATA/'manifest.json')}
    entries={e['image_id']:e for e in read(FIRC/'collected/result/report.json')['predictions']}
    counter_hashes=read(COUNTR/'collected/collection_verification.json')['collected_hashes']
    input_hashes=read(COUNTR/'collected/data/plan.json')['data_hashes']
    assert len(sources)==len(plan['rows'])==65 and set(sources)==set(entries)==set(inputs)
    OUT.mkdir(exist_ok=False)
    files=[PLAN,Path(__file__),CENTER/'result.json',CENTER/'predictions.jsonl',FIRC/'data/plan.json',
           FIRC/'collected/result/report.json',COUNTR/'collected/collection_verification.json',truth_path,
           ROOT/'scripts/analyze_firc_tiny.py',ROOT/'scripts/audit_firc_countr_misses.py',ROOT/'scripts/firc_countr_support.py',
           ROOT/'scripts/run_firc_center_voting.py']
    write(OUT/'freeze.json',{'time_kst':now(),'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in files},'primary':PRIMARY})
    with (OUT/'predictions.jsonl').open('x') as output:
        for row in plan['rows']:
            key=row['image_id'];matched=inputs[key]
            assert 'count' not in row and 'points' not in row
            assert all(row[k]==matched[k] for k in ('encoded_sha256','width','height','roi_xyxy','split','dependency_group'))
            file='result/P/predictions/'+key+'.npy'
            assert sha(COUNTR/'collected'/file)==counter_hashes[file]
            assert sha(DATA/matched['array'])==input_hashes[matched['array']]
            output.write(json.dumps(decode(dict(row,array=matched['array']),entries[key],sources[key]),allow_nan=False)+'\n')
    prediction_hash=sha(OUT/'predictions.jsonl')
    write(OUT/'predictions_frozen.json',{'time_kst':now(),'sha256':prediction_hash,'frames':65})
    labels={r['image_id']:r for r in read(truth_path)}
    with (OUT/'predictions.jsonl').open() as output:records,audit=score_predictions(output,labels)
    for method,old_method in zip(BASELINES,('unchanged','confidence_vote')):
        assert records[method]==previous['per_image'][old_method]
    summary={m:summaries(rr) for m,rr in records.items()}
    primary=summary[PRIMARY];dev=primary['development'];protection={}
    for base in BASELINES:
        comp=summary[base]
        protection[base]={'identical_counts':all(a['count']==b['count'] and a['ae']==b['ae'] for a,b in zip(records[PRIMARY],records[base])),
          'low_mae':primary['low']['mae']<=comp['low']['mae'],'low_f1':primary['low']['positive_macro_f1_at8']>=comp['low']['positive_macro_f1_at8'],
          'empty_fp':dev['empty_false_positives']<=comp['development']['empty_false_positives']}
        for metric in ('game1','game2'):
            bound=comp['development'][metric]
            protection[base][metric]=dev[metric]<=bound+max(.1,.01*bound)+1e-12
            protection[base]['low_'+metric]=primary['low'][metric]<=comp['low'][metric]+1e-12
    group_gains={g:{b:primary['groups'][g]['tp']-summary[b]['groups'][g]['tp'] for b in BASELINES} for g in primary['groups']}
    both=all(diff>0 for ds in group_gains.values() for diff in ds.values())
    pooled=all(dev['pooled_f1_at8']-summary[b]['development']['pooled_f1_at8']>=.02-1e-12 for b in BASELINES)
    specific={m:dev['pooled_f1_at8']-summary[m]['development']['pooled_f1_at8']>=.02-1e-12 for m in MATCHED}
    movements=audit['development']
    assert sum(v['net_matches'] for v in movements)==dev['tp']-summary[BASELINES[1]]['development']['tp']
    protected=all(all(v.values()) for v in protection.values())
    outcome={'completed_kst':now(),'predictions_sha256':prediction_hash,'summary':summary,'per_image':records,'primary_move_audit':audit,
        'recording_gains':group_gains,'strict_gain_in_both':both,'pooled_margin_over_both':pooled,'protection':protection,
        'practical_localization_gain':both and pooled and protected,'specificity_components':specific,'specificity_passed':all(specific.values()),
        'new_forwards':0,'optimizer_updates':0,'reserved_scored':0}
    write(OUT/'result.json',outcome)
    for m,s in summary.items():
        print(m,{k:s['development'][k] for k in ('tp','fp','fn','pooled_f1_at8','mae','game1','game2')},[v['tp'] for v in s['groups'].values()])
    print('development moves',len(movements),'both',both,'pooled',pooled,'protected',protected,'specific',specific)

if __name__=='__main__':main()
