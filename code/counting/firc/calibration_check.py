"""One inference-only calibration check. Development labels are not mounted."""
import argparse, json, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .geometry import transform, image_to_model
from .model import FIRC, load_official_encoder
from .preflight import configure
from .tiny_fit import digest, state_digest, write_json


def select_threshold(predictions, counts, grid):
    """Integer total absolute error; larger threshold breaks exact ties."""
    assert set(predictions) == set(counts)
    scores = [{'threshold': t, 'total_absolute_error': sum(abs(int((q >= t).sum()) - counts[i])
               for i, q in predictions.items())} for t in grid]
    best = min(scores, key=lambda x: (x['total_absolute_error'], -x['threshold']))
    return {'threshold': best['threshold'], 'training_images': len(counts), 'grid_scores': scores,
            'training_mae': best['total_absolute_error']/len(counts)}


def run(args):
    data, assets, out = Path(args.data), Path(args.assets), Path(args.output)
    plan = json.loads((data/'plan.json').read_text())
    for k, h in json.loads((data/'hashes.json').read_text()).items():
        assert digest(data/k) == h, k
    configure(); assert torch.cuda.is_bf16_supported()
    rows = plan['rows']; by_id = {r['image_id']: r for r in rows}
    assert len(by_id) == 65 and all('points' not in r and 'count' not in r for r in rows)
    counts = json.loads((data/'training_counts.json').read_text())
    assert set(counts) == {r['image_id'] for r in rows if r['split'] == 'train'} and len(counts) == 42
    checkpoint = assets/'dinov2_vitb14_reg4_pretrain.pth'
    assert digest(checkpoint) == plan['pretrained_sha256']
    model = FIRC(load_official_encoder(checkpoint)).cuda().eval()
    initial_encoder = state_digest(model.encoder.state_dict())
    refs, boxes = [], []
    for i, r in enumerate(plan['references']):
        assert r['split'] == 'train' and digest(data/r['image']) == r['sha256']
        with Image.open(data/r['image']) as im:
            t = transform(*im.size,224); t['padded_hw'] = [224,224]
            refs.append(torch.from_numpy(image_to_model(im,t)))
            h,w = t['valid_hw']; boxes.append([i,0,0,w,h])
    references = torch.stack(refs).cuda(); boxes = torch.tensor(boxes,dtype=torch.float32,device='cuda')
    report = {'state':'running','optimizer_calls':0,'persistent_steps':0,'query_forwards':0,
              'reference_image_encodings':0,'arms':{},'plan_sha256':digest(data/'plan.json'),
              'encoder_initial_sha256':initial_encoder,'development_started':False}
    start = time.monotonic(); write_json(out/'status.json',report)
    events = (out/'events.jsonl').open('x')
    def event(kind, **kw):
        e = {'sequence':event.n,'event':kind,'monotonic_seconds':time.monotonic()-start,**kw}
        event.n += 1; events.write(json.dumps(e,allow_nan=False)+'\n'); events.flush()
    event.n = 1
    def load_arm(arm):
        meta = plan['checkpoints'][arm]; p = data/meta['file']
        assert digest(p) == meta['sha256']
        saved = torch.load(p,map_location='cpu',weights_only=True)
        assert saved['step'] == 300 and saved['new_updates'] == 150 and saved['arm'] == arm
        assert saved['research_initialization_allowed'] is False and saved['plan_sha256'] == plan['source_plan_sha256']
        result = model.load_state_dict(saved['counter'],strict=False)
        assert not result.unexpected_keys and all(k.startswith('encoder.') for k in result.missing_keys)
        assert all(torch.isfinite(v).all() for v in saved['counter'].values())
        expected = state_digest(saved['counter']); del saved
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): rt = model.encode(references)
        report['reference_image_encodings'] += len(references)
        return expected, rt
    def infer(arm, batch, rt, phase):
        images, transforms = [], []
        for r in batch:
            with Image.open(data/r['image']) as im:
                assert im.size == (r['width'],r['height'])
                t = transform(*im.size,896,r['roi_xyxy']); t['padded_hw'] = [896,896]
                images.append(torch.from_numpy(image_to_model(im,t))); transforms.append(t)
        x = torch.stack(images).cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            taps = model.encode(x)
            outputs = model(x,references,boxes,[t['valid_hw'] for t in transforms],taps=taps,reference_taps=rt)
        entries = []
        for r,v,t in zip(batch,outputs,transforms):
            xy,q = v['points'].cpu().numpy(),v['probabilities'].cpu().numpy()
            assert np.isfinite(xy).all() and np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
            name = arm+'/'+r['image_id']+'.npz'
            np.savez_compressed(out/name,points=xy,probabilities=q)
            e = {'image_id':r['image_id'],'file':name,'sha256':digest(out/name),'transform':t,'split':r['split']}
            entries.append(e); report['query_forwards'] += 1
            event('prediction_saved',arm=arm,image_id=r['image_id'],phase=phase,sha256=e['sha256'])
        report['arms'][arm]['predictions'].extend(entries); write_json(out/'status.json',report)
        return entries
    # Complete both training cohorts and freeze both thresholds before touching development.
    for arm in plan['arms']:
        (out/arm).mkdir(exist_ok=False)
        head,rt = load_arm(arm)
        a = {'predictions':[],'initial_head_sha256':head,'checkpoint_sha256':plan['checkpoints'][arm]['sha256']}
        report['arms'][arm] = a
        parity = infer(arm,[by_id[i] for i in plan['parity_ids']],rt,'train_parity')
        for e in parity:
            old = plan['parity'][arm][e['image_id']]
            assert e['transform'] == old['transform']
            with np.load(out/e['file'],allow_pickle=False) as new,np.load(data/old['file'],allow_pickle=False) as saved:
                assert all(np.array_equal(new[k],saved[k]) for k in ('points','probabilities')), (arm,e['image_id'],'parity')
        a['exact_endpoint_parity'] = True
        for r in rows:
            if r['split']=='train' and r['image_id'] not in plan['parity_ids']: infer(arm,[r],rt,'train')
        a['training_head_unchanged'] = head == state_digest({k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')})
        assert a['training_head_unchanged']
    selected = {}
    for arm in plan['arms']:
        predictions = {}
        for e in report['arms'][arm]['predictions']:
            assert e['split']=='train'
            with np.load(out/e['file'],allow_pickle=False) as z: predictions[e['image_id']] = z['probabilities'].copy()
        selected[arm] = select_threshold(predictions,counts,plan['threshold_grid'])
    freeze = {'plan_sha256':report['plan_sha256'],'training_counts_sha256':digest(data/'training_counts.json'),
              'thresholds':selected,'query_forwards_before_freeze':report['query_forwards'],
              'development_forwards_before_freeze':0,'training_predictions':{a:report['arms'][a]['predictions'] for a in plan['arms']}}
    assert report['query_forwards'] == 84
    with (out/'thresholds_frozen.json').open('x') as f: json.dump(freeze,f,indent=2,allow_nan=False)
    frozen_hash = digest(out/'thresholds_frozen.json'); event('thresholds_frozen',sha256=frozen_hash,thresholds={a:v['threshold'] for a,v in selected.items()})
    report['threshold_receipt_sha256'] = frozen_hash; report['development_started'] = True
    for arm in plan['arms']:
        head,rt = load_arm(arm); assert head == report['arms'][arm]['initial_head_sha256']
        for r in rows:
            if r['split']=='development':
                assert digest(out/'thresholds_frozen.json') == frozen_hash
                infer(arm,[r],rt,'development')
        a = report['arms'][arm]
        a['final_head_sha256'] = state_digest({k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')})
        assert head == a['final_head_sha256']; a['completed'] = True
    assert report['query_forwards'] == 130 and report['reference_image_encodings'] == 8
    assert state_digest(model.encoder.state_dict()) == initial_encoder
    report.update(state='completed',encoder_final_sha256=initial_encoder,elapsed_seconds=time.monotonic()-start,
                  finite_model=True,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    event('completed',query_forwards=130,optimizer_calls=0); events.close()
    write_json(out/'status.json',report); write_json(out/'report.json',report)
    write_json(out/'result_hashes.json',{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and p.name not in ('status.json','result_hashes.json')})


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--assets',required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args()
    try: run(args)
    except Exception: write_json(Path(args.output)/'failure.json',{'traceback':traceback.format_exc()}); raise
