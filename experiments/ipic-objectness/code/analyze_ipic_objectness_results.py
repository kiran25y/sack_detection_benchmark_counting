"""Independent CPU/NumPy verification of frozen three-arm classification test."""
import argparse,datetime,json,math
from pathlib import Path
import numpy as np
from analyze_ipic_transfer_pilot import read,lines,sha,finite,summarize
from verify_ipic_assignment_continuation import compare_objects

def extras(summary):
    rows=summary['all']['rows'];positive=[r for r in rows if r['truth']>0]
    tp=sum(r['point_8']['TP'] for r in rows);pred=sum(r['prediction'] for r in rows);truth=sum(r['truth'] for r in rows)
    return dict(positive_macro_F1=float(np.mean([r['point_8']['F1'] for r in positive])),pooled_F1=2*tp/(pred+truth),
                positive_frames_with_TP=sum(r['point_8']['TP']>0 for r in positive),TP=tp,FP=pred-tp,FN=truth-tp)

def gate(results,t):
    h=results['H']['development'][t];hx=extras(h);checks={};differences={}
    for arm in ['F','U']:
        c=results[arm]['development'][t];cx=extras(c);key=arm+'_'
        checks[key+'MAE_gain']=c['all']['MAE']-h['all']['MAE']>=max(.5,.1*c['all']['MAE'])-1e-9
        checks[key+'positive_F1_gain']=hx['positive_macro_F1']-cx['positive_macro_F1']>=.05-1e-9
        differences[arm]=dict(MAE=h['all']['MAE']-c['all']['MAE'],positive_macro_F1=hx['positive_macro_F1']-cx['positive_macro_F1'])
        for bucket,metrics in [('all',['RMSE','GAME1','GAME2']),('low_1_11',['MAE','GAME1','GAME2'])]:
            for metric in metrics:
                old=c[bucket][metric];new=h[bucket][metric]
                checks[key+bucket+'_'+metric]=new<=old+max(.1,.01*old)+1e-9
                differences[arm][bucket+'_'+metric]=new-old
        checks[key+'low_F1']=h['low_1_11']['F1_8']>=c['low_1_11']['F1_8']-.01-1e-9
        checks[key+'empty_FP']=h['zero']['FP_total']<=c['zero']['FP_total']
        differences[arm]['empty_FP']=h['zero']['FP_total']-c['zero']['FP_total']
        for group in c['groups']:
            old=c['groups'][group]['MAE'];new=h['groups'][group]['MAE']
            checks[key+'group_'+group]=new<=old+max(.1,.01*old)+1e-9
            differences[arm]['group_'+group]=new-old
    absolute=dict(MAE=h['all']['MAE']<=7.71739130435,pooled_F1=hx['pooled_F1']>=.5,
                  positive_frame_coverage=hx['positive_frames_with_TP']>=14,
                  fitting_positive_F1=extras(results['H']['fitting'][t])['positive_macro_F1']>=.7)
    return dict(passed=all(checks.values()) and all(absolute.values()),relative_passed=all(checks.values()),absolute_passed=all(absolute.values()),
                checks=checks,absolute_checks=absolute,signed_differences=differences)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--collected',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists(),'Existing analysis: reuse after validation';col=a.collected
    ver=read(col/'collection_verification.json');controller=read(col/'controller_status.json');spec=read(col/'data/objectness_plan.json')
    assert controller['state']=='completed' and controller['completed_arms']==['F','U','H'] and ver['controller_exited']
    assert ver['source_actual']==read(col/'frozen_hashes.json')
    for n,h in ver['collected_hashes'].items():assert sha(col/n)==h
    for n,h in spec['data_hashes'].items():assert ver['source_actual']['data/'+n]==h
    for n,h in spec['source_hashes'].items():assert ver['source_actual'][n]==h
    assert spec['primary_arm']=='H' and spec['updates_per_arm']==200 and spec['total_new_updates']==600 and spec['thresholds']==[.5,.1] and not spec['threshold_selection']
    results={};replayed=None
    for arm in ['F','U','H']:
        folder=col/'result'/arm;status=read(folder/'status.json');plan=read(folder/'plan.json');hashes=read(folder/'result_hashes.json')
        assert status['state']=='completed' and status['new_durable_steps']==200 and status['finite_model_optimizer'] and status['optimizer_moments_preserved']
        assert plan==dict(spec,arm=arm) and hashes==ver['result_actual'][arm]
        ex=read(col/f'exit_{arm}.json');ds=ver['docker'][arm]['state'];assert ds['ExitCode']==ex['returncode']==0 and not ds['OOMKilled'] and not ex['guard_stop']
        for name in ['counter.pt','resume.pt']:
            meta=ver['checkpoints'][arm][name];m=meta['literal_metadata_without_deserialization']
            assert meta['sha256']==hashes[name] and m['optimizer_steps']==400 and m['new_optimizer_steps']==200 and m['arm']==arm and m['seed']==1729 and m['plan_sha256']==sha(folder/'plan.json')
        assert hashes['counter.pt']==status['checkpoint_sha256'] and hashes['resume.pt']==status['resume_sha256']
        updates=lines(folder/'updates.jsonl');ledger=lines(folder/'augmentation_ledger.jsonl');draws=lines(col/'data/draws_P11.jsonl')
        assert len(updates)==200 and [r['update'] for r in updates]==list(range(1,201)) and finite(updates)
        assert len(ledger)==1600 and [r['draw'] for r in ledger]==draws
        if replayed is None:replayed=ledger
        else:assert ledger==replayed,'Unequal training inputs across arms'
        for u in updates:
            assert u['optimizer_updates']==1 and len(u['microbatch_losses'])==8
            for r in u['microbatch_losses']:
                base=sum(r[v]['classification']+5*r[v]['point'] for v in ['view_a','view_b'])/2
                assert math.isclose(r['total'],base,rel_tol=2e-5,abs_tol=2e-5)
        result={}
        for label,prefix,manifest in [('development','', 'development_manifest.json'),('fitting','fitting','train_manifest.json')]:
            truth={r['image_id']:r for r in read(col/'data'/manifest)};path=folder/prefix
            records=lines(path/'development_predictions_0400.jsonl');saved=read(path/'development_0400.json')
            assert len(records)==len(truth)==len({r['query_id'] for r in records}) and {r['query_id'] for r in records}==set(truth) and finite(records)
            for r in records:
                tr=truth[r['query_id']];assert r['truth']==tr['points'] and r['count']==tr['count'] and r['dependency_group']==tr['dependency_group']
                assert (r['width'],r['height'])==(tr['width'],tr['height']) and len(r['points'])==len(r['probabilities']) and all(0<=p<=1 for p in r['probabilities'])
            scores={str(t):summarize(records,t) for t in [.5,.1]};compare_objects(saved['scores'],scores)
            result[label]=scores;result[label+'_extras']={t:extras(s) for t,s in scores.items()}
        results[arm]=result
    result=dict(state='verified_complete',client_kst=datetime.datetime.now().astimezone().isoformat(),arms=results,
                total_new_updates=600,primary_gate=gate(results,'0.5'),secondary_diagnostic_cannot_rescue_primary=gate(results,'0.1'),
                limitation='One matched continuation from P11, repeated fitting draws, three training and two reused development recording groups, no untouched test or novelty claim. Worker verified tensor finiteness; this analysis does not deserialize model tensors.',automatic_full_training=False)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(state=result['state'],primary_gate=result['primary_gate']),indent=2))
if __name__=='__main__':main()
