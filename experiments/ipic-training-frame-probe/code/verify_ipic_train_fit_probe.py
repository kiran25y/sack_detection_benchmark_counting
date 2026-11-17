"""Collect one terminal inference probe, independently score train and saved dev."""
import collections,datetime,hashlib,io,json,subprocess,zipfile
from pathlib import Path
import numpy as np
from verify_ipic_assignment_continuation import calculate
ROOT=Path(__file__).resolve().parents[1]
NAME='ipic_sack_train_fit_probe_v1_20261104'
REMOTE='/project'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(rr,t):
    result=[]
    for r in rr:
        points=np.asarray(r['points']);keep=np.asarray(r['probabilities'])>=t;p=points[keep]
        m=calculate(dict(points=p,probabilities=np.ones(len(p)),valid=np.ones(len(p),bool),
            truth=np.asarray(r['truth']).reshape(-1,2),width=r['width'],height=r['height']))
        result.append(dict(query_id=r['query_id'],**m))
    return dict(threshold=t,MAE=float(np.mean([r['AE'] for r in result])),RMSE=float(np.sqrt(np.mean([r['error']**2 for r in result]))),
        bias=float(np.mean([r['error'] for r in result])),GAME1=float(np.mean([r['GAME1'] for r in result])),GAME2=float(np.mean([r['GAME2'] for r in result])),
        F1_8=float(np.mean([r['point_8']['F1'] for r in result])),F1_16=float(np.mean([r['point_16']['F1'] for r in result])),rows=result)
def main():
    folder=ROOT/'artifacts'/NAME;archive=folder/'result_outputs.zip';col=folder/'collected'
    if (folder/'analysis.json').exists():raise ValueError('Analysis exists; reuse it')
    if not archive.exists():
        code=r'''
import hashlib,io,json,pathlib,subprocess,sys,zipfile
root=pathlib.Path(ROOT);p=root/'research-direction-20261012'/NAME
controller=json.loads((p/'controller_status.json').read_text());ex=json.loads((p/'exit.json').read_text())
assert controller['state']=='completed' and ex['returncode']==0 and not ex['guard_stop']
d=json.loads(subprocess.check_output(['docker','inspect',NAME.replace('_','-')]))[0]
assert d['Id']==ex['container_id'] and d['State']['Status']=='exited' and d['State']['ExitCode']==0 and not d['State']['OOMKilled']
proc=pathlib.Path('/proc')/str(PID)/'stat'
controller_process=proc.read_text().split()[2] if proc.exists() else 'absent'
assert controller_process in ('absent','Z','X')
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
frozen=json.loads((p/'frozen_hashes.json').read_text());assert all(sha(p/n)==h for n,h in frozen.items())
spec=json.loads((p/'data/probe_spec.json').read_text());assert all(sha(root/v['path'])==v['sha256'] for v in spec['checkpoints'].values())
hashes=json.loads((p/'result/result_hashes.json').read_text());assert all(sha(p/'result'/n)==h for n,h in hashes.items())
files=[p/n for n in ['controller_status.json','exit.json','admission.json','launch.json','telemetry.jsonl','frozen_hashes.json','data/probe_spec.json']]+[q for q in (p/'result').iterdir() if q.is_file()]
verification=dict(docker_id=d['Id'],docker_state=d['State'],controller_process=controller_process,all_frozen_hashes=True,all_checkpoint_hashes=True,
    file_hashes={str(q.relative_to(p)):sha(q) for q in files})
b=io.BytesIO()
with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for q in files:z.write(q,str(q.relative_to(p)))
 z.writestr('verification.json',json.dumps(verification,indent=2))
sys.stdout.buffer.write(b.getvalue())
'''
        pid=read(folder/'launch_receipt.json')['controller_pid']
        source=f'ROOT={REMOTE!r}\nNAME={NAME!r}\nPID={pid!r}\n'+code
        p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','training-host','python3','-'],input=source.encode(),capture_output=True,timeout=120)
        if p.returncode:raise RuntimeError(p.stderr.decode())
        with archive.open('xb') as f:f.write(p.stdout)
    col.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            pp=Path(n);assert not pp.is_absolute() and '..' not in pp.parts
        z.extractall(col)
    v=read(col/'verification.json');assert all(sha(col/n)==h for n,h in v['file_hashes'].items())
    report=read(col/'result/report.json');assert report['state']=='completed' and report['query_forwards']==80 and report['reference_batch_forwards']==4 and report['optimizer_steps']==0
    spec=read(folder/'protocol.json');assert spec==read(col/'data/probe_spec.json')
    manifest=read(ROOT/'artifacts/ipic_sack_data_admission_v1_20261103/train_manifest.json');truth={r['image_id']:r for r in manifest}
    results={}
    for model in ['foundation','A','B','E']:
        m=report['models'][model];assert m['state_hash_before']==m['state_hash_after'] and m['checkpoint_sha256']==spec['checkpoints'][model]['sha256']
        rr=rows(col/'result'/f'{model}_training_predictions.jsonl');assert len(rr)==len(truth) and {r['query_id'] for r in rr}==set(truth)
        for r in rr:
            gt=truth[r['query_id']];assert r['truth']==gt['points'] and r['count']==gt['count']
            assert np.isfinite(r['points']).all() and np.isfinite(r['probabilities']).all()
            assert len(r['points'])==len(r['probabilities']) and all(0<=p<=1 for p in r['probabilities'])
        dev_path=ROOT/'artifacts/ipic_sack_screen_v1_20261103/collected/result'/('A' if model=='foundation' else model)/('development_predictions_0000.jsonl' if model=='foundation' else 'development_predictions_1000.jsonl')
        results[model]=dict(training={str(t):score(rr,t) for t in spec['thresholds']},
                           development={str(t):score(rows(dev_path),t) for t in spec['thresholds']},
                           fitting_loss={k:float(np.mean([r['supervised_diagnostic'][k] for r in rr])) for k in ['focal','point','soft_count']})
    output=dict(verified_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        archive_sha256=sha(archive),verification=v,query_forwards=80,reference_batch_forwards=4,optimizer_steps=0,parameters_unchanged=True,
        threshold_selection=False,models=results,limitations='Training results are seen-data fitting diagnostics; development is reused and one dependent group; not independent accuracy.')
    (folder/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps({m:{s:{t:{k:v for k,v in sc.items() if k!='rows'} for t,sc in scores.items()} if s!='fitting_loss' else scores for s,scores in r.items()} for m,r in results.items()},indent=2))
if __name__=='__main__':main()
