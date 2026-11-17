"""Independent train-only selection and native-coordinate development verification."""
import hashlib,json,pathlib
import numpy as np
from analyze_firc_tiny import evaluate,summarize
ROOT=pathlib.Path(__file__).resolve().parents[1]
ATTEMPT='firc_calibration_v1_20261106'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    out=ROOT/'artifacts'/ATTEMPT; result=out/'collected/result'; plan=read(out/'data/plan.json')
    report=read(result/'report.json'); v=read(out/'collected/verification.json'); launch=read(out/'launch_receipt.json')
    truth_path=ROOT/plan['local_truth_manifest']; truth=read(truth_path); by_id={r['image_id']:r for r in truth}
    freeze=read(result/'thresholds_frozen.json'); events=[json.loads(s) for s in (result/'events.jsonl').read_text().splitlines()]
    c,d=v['controller_status'],v['docker_state']
    checks={'plan':sha(out/'data/plan.json')==launch['plan_sha256']==report['plan_sha256']==freeze['plan_sha256'],
      'local_release':all(sha(out/k)==h for k,h in launch['release_hashes'].items()),'remote_release':v['release_hashes_match'],
      'analyzer_frozen':sha(pathlib.Path(__file__))==launch['release_hashes']['analysis/analyze_firc_calibration.py'],
      'metric_helper_frozen':sha(ROOT/'scripts/analyze_firc_tiny.py')==launch['release_hashes']['analysis/analyze_firc_tiny.py'],
      'controller_completed':not v['controller_alive'] and c['state']=='completed' and c['exit_code']==0 and c['guard_stop'] is None,
      'docker_completed':d['Status']=='exited' and d['ExitCode']==0 and not d['OOMKilled'],
      'result_hashes':all(sha(result/k)==h for k,h in read(result/'result_hashes.json').items()),
      'truth_identity':sha(truth_path)==plan['truth_manifest_sha256'],
      'counts_identity':read(out/'data/training_counts.json')=={i:r['count'] for i,r in by_id.items() if r['split']=='train'},
      'no_development_truth_in_inputs':all(not any(k in r for k in ['points','count']) for r in plan['rows']),
      'budget':report['state']=='completed' and report['optimizer_calls']==report['persistent_steps']==0 and report['query_forwards']==130 and report['reference_image_encodings']==8,
      'encoder_unchanged':report['encoder_initial_sha256']==report['encoder_final_sha256'],
      'worker_finite_model':report['finite_model'],
      'threshold_receipt':sha(result/'thresholds_frozen.json')==report['threshold_receipt_sha256'],
      'event_sequence':[e['sequence'] for e in events]==list(range(1,len(events)+1))}
    f=[e for e in events if e['event']=='thresholds_frozen']; assert len(f)==1
    before=[e for e in events if e['event']=='prediction_saved' and e['sequence']<f[0]['sequence']]
    after=[e for e in events if e['event']=='prediction_saved' and e['sequence']>f[0]['sequence']]
    checks['freeze_before_development']=len(before)==84 and len(after)==46 and all(e['phase'].startswith('train') for e in before) and all(e['phase']=='development' for e in after) and freeze['query_forwards_before_freeze']==84 and freeze['development_forwards_before_freeze']==0 and f[0]['sha256']==sha(result/'thresholds_frozen.json')
    checks['input_coverage']=len(by_id)==65 and {r['image_id'] for r in plan['rows']}==set(by_id)
    checks['input_metadata']=all(all(r[k]==by_id[r['image_id']][k] for k in ('width','height','split','roi_xyxy','encoded_sha256','dependency_group')) for r in plan['rows'])
    checks['source_images']=all(sha(out/'data'/r['image'])==r['encoded_sha256'] for r in plan['rows'])
    checks['group_split']=not ({r['dependency_group'] for r in truth if r['split']=='train'} & {r['dependency_group'] for r in truth if r['split']=='development'})
    metrics={}; selections={}
    for arm in plan['arms']:
        a=report['arms'][arm]; entries=a['predictions']; arrays={}
        checks[arm+'_coverage']=len(entries)==65 and len({e['image_id'] for e in entries})==65 and {e['image_id'] for e in entries}==set(by_id)
        checks[arm+'_unchanged']=a['completed'] and a['training_head_unchanged'] and a['initial_head_sha256']==a['final_head_sha256']
        checks[arm+'_checkpoint']=a['checkpoint_sha256']==plan['checkpoints'][arm]['sha256']==sha(out/'data'/plan['checkpoints'][arm]['file'])
        checks[arm+'_parity']=a['exact_endpoint_parity']
        for e in entries:
            assert sha(result/e['file'])==e['sha256']; row=by_id[e['image_id']]; t=e['transform']
            assert e['split']==row['split'] and t['roi']==row['roi_xyxy']
            # Independently reconstruct the resized valid rectangle and pixel-center map.
            rw,rh=row['roi_xyxy'][2]-row['roi_xyxy'][0],row['roi_xyxy'][3]-row['roi_xyxy'][1]
            vw,vh=round(rw*896/max(rw,rh)),round(rh*896/max(rw,rh))
            assert t['valid_hw']==[vh,vw] and t['padded_hw']==[896,896] and t['scale_xy']==[vw/rw,vh/rh]
            with np.load(result/e['file'],allow_pickle=False) as z:
                assert set(z.files)=={'points','probabilities'};xy=z['points'].copy();q=z['probabilities'].copy()
            assert xy.shape==(len(q),2) and np.isfinite(xy).all() and np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
            assert (xy>=0).all() and (xy[:,0]<=vw).all() and (xy[:,1]<=vh).all()
            assert len(q)==len(np.arange(2,vw,4))*len(np.arange(2,vh,4))
            if row['image_id'] in plan['parity_ids']:
                old=plan['parity'][arm][row['image_id']]
                with np.load(out/'data'/old['file'],allow_pickle=False) as z:
                    checks[arm+'_parity'] &= np.array_equal(xy,z['points']) and np.array_equal(q,z['probabilities'])
            arrays[row['image_id']]=(xy,q,t)
        train=[r for r in truth if r['split']=='train']
        scores=[(sum(abs(np.count_nonzero(arrays[r['image_id']][1]>=threshold)-r['count']) for r in train),threshold) for threshold in plan['threshold_grid']]
        error,chosen=min(scores,key=lambda pair:(pair[0],-pair[1])); chosen=float(chosen)
        frozen=freeze['thresholds'][arm]
        checks[arm+'_selection']=chosen==frozen['threshold'] and error/42==frozen['training_mae'] and [{'threshold':float(t),'total_absolute_error':int(e)} for e,t in scores]==frozen['grid_scores']
        selections[arm]={'threshold':chosen,'training_mae':error/42,'total_absolute_error':int(error)}
        metrics[arm]={}
        for mode,threshold in [('fixed',.5),('calibrated',chosen)]:
            per=[]
            for row in truth:
                xy,q,t=arrays[row['image_id']]; pred=xy[q>=threshold].astype(float)/np.asarray(t['scale_xy'])+np.asarray(t['roi'][:2])-.5
                m=evaluate(pred,np.asarray(row['points'],dtype=float).reshape(-1,2),row);m['split']=row['split'];per.append(m)
            train_m=[r for r in per if r['split']=='train'];dev=[r for r in per if r['split']=='development']
            low=[r for r in dev if 1<=r['truth']<=11];high=[r for r in dev if r['truth']>=12]
            metrics[arm][mode]={'threshold':threshold,'training':summarize(train_m),'development':summarize(dev),
               'low':summarize(low),'higher':summarize(high),'positive_frames_with_tp':sum(r['truth']>0 and r['tp']>0 for r in dev),
               'groups':{g:summarize([r for r in dev if r['group']==g]) for g in sorted({r['group'] for r in dev})},'per_image':per}
    gates={};deltas={}
    for arm,m in metrics.items():
        p,b=m['calibrated'],m['fixed'];s,c=p['development'],b['development']
        protect={}
        for scope,keys in [('development',['rmse','game1','game2']),('low',['mae','game1','game2'])]:
            for key in keys:protect[scope+'_'+key]=p[scope][key]-b[scope][key]<=max(.1,.01*b[scope][key])+1e-12
        for g in b['groups']:protect['group_'+g]=p['groups'][g]['mae']-b['groups'][g]['mae']<=max(.1,.01*b['groups'][g]['mae'])+1e-12
        protect['low_f1']=p['low']['positive_macro_f1_at8']>=b['low']['positive_macro_f1_at8']-.01-1e-12
        protect['empty_fp']=s['empty_false_positives']<=c['empty_false_positives']
        relative={'engineering':all(checks.values()),'mae_gain':c['mae']-s['mae']>=max(.5,.1*c['mae'])-1e-12,
                  'positive_macro_f1_gain':s['positive_macro_f1_at8']-c['positive_macro_f1_at8']>=.05-1e-12,**protect}
        absolute={'engineering':all(checks.values()),'mae':s['mae']<=7.71739130435,'pooled_f1':s['pooled_f1_at8']>=.5,
                  'positive_macro_f1':s['positive_macro_f1_at8']>=.5,'positive_frames_tp':p['positive_frames_with_tp']>=14,
                  'fitting_positive_macro_f1':p['training']['positive_macro_f1_at8']>=.8,'empty_fp':s['empty_false_positives']==0}
        gates[arm]={'relative_components':relative,'absolute_components':absolute,'relative_passed':all(relative.values()),'absolute_passed':all(absolute.values())}
        deltas[arm]={scope:{k:p[scope][k]-b[scope][k] for k in ['mae','rmse','bias','pooled_f1_at8','positive_macro_f1_at8','game1','game2','empty_false_positives']} for scope in ['training','development','low','higher']}
    output={'engineering_complete':all(checks.values()),'checks':checks,'selections':selections,'metrics':metrics,'gates':gates,'calibrated_minus_fixed':deltas,
      'primary':'reference_lr','primary_passed':gates['reference_lr']['relative_passed'] and gates['reference_lr']['absolute_passed'],
      'reserved_images_scored':0,'optimizer_calls':0,'full_training_started':False,'final_deployable_accuracy':None,
      'limits':'Standard train-only calibration on heads fitted to four images. Reused development groups; no independent final-test result, no physical quartets, no novelty claim.'}
    with (out/'analysis.json').open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in output.items() if k not in ('metrics','calibrated_minus_fixed')},indent=2))
if __name__=='__main__':main()
