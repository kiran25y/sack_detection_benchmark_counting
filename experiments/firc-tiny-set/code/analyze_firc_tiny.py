"""Independent NumPy/stdlib metrics on saved arrays; no model or threshold search."""
import argparse, hashlib, json, math, pathlib
import numpy as np


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


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


def summarize(rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn'))
    return {'mae': float(np.mean([r['ae'] for r in rows])), 'rmse': float(np.sqrt(np.mean([r['error']**2 for r in rows]))),
            'bias': float(np.mean([r['error'] for r in rows])), 'pooled_f1_at8': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'positive_macro_f1_at8': float(np.mean([r['f1'] for r in rows if r['truth']>0])),
            'game1': float(np.mean([r['game1'] for r in rows])), 'game2': float(np.mean([r['game2'] for r in rows])),
            'empty_false_positives': sum(r['count'] for r in rows if r['truth']==0), 'tp': tp, 'fp': fp, 'fn': fn}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--attempt', required=True); args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]; attempt = root/'artifacts'/args.attempt
    data = attempt/'data'; collected = attempt/'collected'; result = collected/'result'
    plan = json.loads((data/'plan.json').read_text()); report = json.loads((result/'report.json').read_text())
    receipt = json.loads((collected/'verification.json').read_text())
    launch = json.loads((attempt/'launch_receipt.json').read_text())
    checks = {}
    checks['frozen_plan'] = sha(data/'plan.json') == launch['plan_sha256'] == report['plan_sha256']
    checks['local_release'] = all(sha(attempt/k)==h for k,h in launch['release_hashes'].items())
    checks['remote_release'] = receipt['release_hashes_match']
    checks['controller_exited'] = not receipt['controller_alive']
    c = receipt['controller_status']; d = receipt['docker_state']
    checks['controller_completed'] = c['state']=='completed' and c['exit_code']==0 and c['guard_stop'] is None
    checks['docker_completed'] = d['Status']=='exited' and d['ExitCode']==0 and not d['OOMKilled']
    hashes = json.loads((result/'result_hashes.json').read_text())
    checks['result_hashes'] = all(sha(result/k)==h for k,h in hashes.items())
    checks['checkpoint_hash'] = sha(result/'diagnostic.pt') == report['checkpoint_sha256']
    checks['budget'] = report['saved_updates']==report['optimizer_calls']==plan['updates']==150 and report['query_presentations']==1200
    logs = [json.loads(s) for s in (result/'updates.jsonl').read_text().splitlines()]
    checks['sequential_logs'] = [r['step'] for r in logs] == list(range(1,151))
    checks['presentations_and_no_quartets'] = all(r['query_presentations']==8 and r['eligible_quartets']==0 and len(r['bundles'])==2 and all(len(b)==4 for b in r['bundles']) for r in logs)
    def finite(v):
        if isinstance(v,dict): return all(finite(x) for x in v.values())
        if isinstance(v,list): return all(finite(x) for x in v)
        return not isinstance(v,(int,float)) or math.isfinite(v)
    checks['finite_logs'] = finite(logs)
    checks['worker_finite_states'] = report['finite_gradients'] and report['finite_model_adam']
    checks['worker_frozen_encoder'] = report['encoder_initial_sha256']==report['encoder_final_sha256']
    checks['new_head_updated'] = report['head_initial_sha256']!=report['head_final_sha256']
    checks['threshold_fixed'] = report['threshold']==plan['threshold']==.5
    checks['unchanged_original_loss'] = plan['loss']=={'focal_alpha':.25,'focal_gamma':2.,'point_weight':5.,'count_weight':.1}
    checks['train_only'] = all(r['split']=='train' and r['exhaustive_reviewed'] for r in plan['rows'])
    metrics = {}
    for stage in ('initial','final'):
        entries = report[stage+'_predictions']; details=[]
        checks[stage+'_coverage'] = [e['image_id'] for e in entries]==[r['image_id'] for r in plan['rows']]
        for e, row in zip(entries,plan['rows']):
            assert e['sha256']==sha(result/e['file'])
            with np.load(result/e['file'],allow_pickle=False) as arrays:
                points, q, gt = arrays['points'],arrays['probabilities'],arrays['truth_native']
                assert np.isfinite(points).all() and np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
                assert np.array_equal(gt,np.asarray(row['points']).reshape(-1,2))
                t=e['transform']; pred=points[q>=.5].astype(float)/np.asarray(t['scale_xy'])+np.asarray(t['roi'][:2])-.5
                assert len(pred)==e['hard_count']
                assert np.isclose(q.sum(),e['soft_count'],rtol=1e-7,atol=1e-7)
                assert (points>=0).all() and (points[:,0]<=t['valid_hw'][1]).all() and (points[:,1]<=t['valid_hw'][0]).all()
                m=evaluate(pred,gt,row);m.update(soft_count=e['soft_count'],maximum_probability=e['maximum_probability']);details.append(m)
        metrics[stage]={'per_image':details,'summary':summarize(details)}
    f=metrics['final']['summary']; gate={'engineering_complete':all(checks.values()),'mae_at_most1':f['mae']<=1,
        'pooled_f1_at_least_point8':f['pooled_f1_at8']>=.8,'positive_macro_f1_at_least_point8':f['positive_macro_f1_at8']>=.8,
        'empty_fp_zero':f['empty_false_positives']==0}
    output={'checks':checks,'metrics':metrics,'trainability_gate_components':gate,'trainability_gate_passed':all(gate.values()),
            'physical_quartet_gate':'not_passed_no_admitted_quartet','full_training_authorized_by_result':False,
            'limits':'Four fitting images, one seed, no held-out accuracy. Finite model/Adam and frozen encoder are worker checks; local verifier does not deserialize checkpoint tensors.',
            'plan_sha256':sha(data/'plan.json'),'checkpoint_sha256':report['checkpoint_sha256']}
    with (attempt/'analysis.json').open('x') as file:json.dump(output,file,indent=2,allow_nan=False)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
