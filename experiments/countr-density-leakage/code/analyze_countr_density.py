"""Independently recompute B/P/C/L saved arrays and the frozen loss-repair gate."""
import argparse,datetime,json,math
from pathlib import Path
import numpy as np
from analyze_countr_sack_test import read,sha,compare
from countr_sack_metrics import score,summary
from countr_density_gate import gate
ROOT=Path(__file__).resolve().parents[1]


def metrics(pred,rows,local):
    assert {p.stem for p in pred.glob('*.npy')}=={r['image_id'] for r in rows}
    scored=[]
    for r in rows:
        raw=np.load(pred/(r['image_id']+'.npy'));data=np.load(local/r['array'])
        assert raw.shape==data['mask'].shape and np.isfinite(raw).all()
        scored.append(dict(split=r['split'],**score(raw,data['mask'],r)))
    result={name:summary([r for r in scored if r['split']==split]) for name,split in [('fitting','train'),('development','development')]}
    compare(result,read(pred/'metrics.json'));return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',required=True);args=ap.parse_args();folder=ROOT/'artifacts'/args.attempt;col=folder/'collected';assert not (folder/'analysis.json').exists()
    ver=read(col/'collection_verification.json');spec=read(col/'data/plan.json');ctl=read(col/'controller_status.json')
    assert ctl['state']=='completed' and ctl['completed_arms']==['C','L'] and ver['controller_exited']
    assert ver['source_actual']==read(folder/'frozen_hashes.json') and ver['weights_actual']==spec['weights_sha256']
    for n,h in ver['collected_hashes'].items():assert sha(col/n)==h,n
    for n,h in spec['source_hashes'].items():assert ver['source_actual'][n]==h,n
    for n,h in spec['analysis_hashes'].items():assert sha(ROOT/'scripts'/n)==h,n
    local=ROOT/'artifacts/countr_sack_recipe_v1_20261105'
    for n,h in spec['data_hashes'].items():assert sha(local/n)==h,n
    old=ROOT/'artifacts'/spec['reused_control']['attempt'];oldcol=old/'collected'
    for name,key in [('analysis.json','analysis_sha256'),('collected/collection_verification.json','collection_sha256'),('result_outputs.zip','archive_sha256'),('collected/data/plan.json','plan_sha256')]:assert sha(old/name)==spec['reused_control'][key],name
    oldver=read(oldcol/'collection_verification.json')
    for n,h in oldver['collected_hashes'].items():assert sha(oldcol/n)==h,n
    assert read(old/'analysis.json')['state']=='verified_complete'
    rows=read(col/'data/manifest.json');draws=read(col/'data/draws.json');assert rows==read(oldcol/'data/manifest.json') and draws==read(oldcol/'data/draws.json')
    assert len(rows)==65 and len({r['image_id'] for r in rows})==65
    oldledger=[json.loads(l) for l in (oldcol/'result/P/inputs.jsonl').read_text().splitlines()]
    adm=read(col/'data/cuda_admission.json');cal=adm['calibration'];results={}
    for arm in ['C','L']:
        f=col/'result'/arm;s=read(f/'status.json');ex=read(col/f'exit_{arm}.json');d=ver['docker'][arm]['state']
        assert d['Status']=='exited' and d['ExitCode']==ex['returncode']==0 and not d['OOMKilled'] and not ex['guard_stop']
        assert s['state']=='completed' and s['durable_steps']==300 and s['finite_model_optimizer']
        assert s['initial']==spec['reused_initial_identity'] and s['initial']['encoder']==s['final']['encoder'] and s['plan_sha256']==sha(col/'data/plan.json')
        assert s['calibration_sha256']==adm['calibration_sha256'] and sha(f/'calibration.json')==s['calibration_sha256'] and read(f/'calibration.json')==cal
        meta=ver['checkpoints'][arm]['resume.pt'];m=meta['literal_metadata_without_deserialization']
        assert meta['sha256']==s['resume_sha256'] and m['steps']==300 and m['arm']==arm and m['seed']==1729 and m['plan_sha256']==s['plan_sha256']
        assert m['calibration_sha256']==s['calibration_sha256']
        updates=[json.loads(l) for l in (f/'updates.jsonl').read_text().splitlines()];ledger=[json.loads(l) for l in (f/'inputs.jsonl').read_text().splitlines()]
        assert len(updates)==300 and [r['update'] for r in updates]==list(range(1,301)) and len(ledger)==1200 and ledger==oldledger
        for u in updates:
            assert all(math.isfinite(u[k]) for k in ['loss','gradient_norm','lr','seconds','coefficient']) and u['coefficient']==cal['coefficients'][arm]
            assert len(u['micro_losses'])==len(u['components'])==4
            for loss,parts in zip(u['micro_losses'],u['components']):
                assert all(math.isfinite(v) for v in parts.values())
                assert math.isclose(loss,parts['mse']+u['coefficient']*parts['count' if arm=='C' else 'leakage'],rel_tol=2e-6,abs_tol=2e-7)
        results[arm]=metrics(f/'predictions',rows,local)
    for name,path in [('P','predictions'),('B','baseline_B')]:results[name]=metrics(oldcol/'result/P'/path,rows,local)
    result=dict(state='verified_complete',client_kst=datetime.datetime.now().astimezone().isoformat(),results=results,gate=gate(results),optimizer_calls=600,calibration=cal,
        limitations='One seed,42fitting/23reused development frames in3/2groups. Five reserved groups untouched. Density spatial metrics are not discrete point F1. Worker tensor finiteness is not independent checkpoint deserialization. No novelty/full-training claim.',automatic_full_training=False)
    with (folder/'analysis.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['gate'],indent=2))


if __name__=='__main__':main()
