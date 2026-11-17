"""One fixed-endpoint train-calibrated development evaluation, no optimization."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .geometry import transform,image_to_model
from .model import FIRC,load_official_encoder
from .preflight import configure
from .tiny_fit import digest,state_digest,write_json
from .calibration_check import select_threshold
def run(args):
    data,assets,out=Path(args.data),Path(args.assets),Path(args.output);plan=json.loads((data/'plan.json').read_text())
    for k,h in json.loads((data/'hashes.json').read_text()).items():assert digest(data/k)==h,k
    configure();assert torch.cuda.is_bf16_supported()
    model=FIRC(load_official_encoder(assets/'dinov2_vitb14_reg4_pretrain.pth')).cuda().eval()
    assert digest(assets/'dinov2_vitb14_reg4_pretrain.pth')==plan['pretrained_sha256']
    payload=torch.load(data/'adaptation.pt',map_location='cpu',weights_only=True)
    assert payload['step']==3000 and payload['stage']=='adaptation' and payload['seed']==1729 and payload['ordinary_supervision_only'] and payload['plan_sha256']==plan['training_plan_sha256']
    missing=model.load_state_dict(payload['counter'],strict=False);assert not missing.unexpected_keys and all(k.startswith('encoder.') for k in missing.missing_keys)
    head=state_digest(payload['counter']);assert head==plan['head_sha256'];del payload
    encoder=state_digest(model.encoder.state_dict());assert encoder==plan['encoder_sha256']
    assert all('points' not in r and 'count' not in r for r in plan['rows'])
    refs=[];boxes=[]
    for i,r in enumerate(plan['references']):
        assert r['split']=='train' and digest(data/r['image'])==r['sha256']
        with Image.open(data/r['image']) as im:
            t=transform(*im.size,224);t['padded_hw']=[224,224];refs.append(torch.from_numpy(image_to_model(im,t)))
            h,w=t['valid_hw'];boxes.append([i,0,0,w,h])
    refs=torch.stack(refs).cuda();boxes=torch.tensor(boxes,dtype=torch.float32,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):rt=model.encode(refs)
    report={'state':'running','query_forwards':0,'optimizer_calls':0,'persistent_steps':0,'reference_image_encodings':len(refs),
      'head_initial_sha256':head,'encoder_initial_sha256':encoder,'plan_sha256':digest(data/'plan.json'),'checkpoint_sha256':digest(data/'adaptation.pt'),'predictions':[]}
    assert report['checkpoint_sha256']==plan['checkpoint_sha256'];counts=json.loads((data/'training_counts.json').read_text())
    assert len(counts)==42 and set(counts)=={r['image_id'] for r in plan['rows'] if r['split']=='train'}
    start=time.monotonic();train_q={};sequence=0;write_json(out/'status.json',report)
    with (out/'events.jsonl').open('x') as events:
        for split in ['train','development']:
            if split=='development':
                selected=select_threshold(train_q,counts,plan['threshold_grid'])
                frozen={'selection':selected,'training_count_sha256':digest(data/'training_counts.json'),'training_prediction_hashes':{e['image_id']:e['sha256'] for e in report['predictions']},'query_forwards':report['query_forwards'],'development_forwards':0}
                assert report['query_forwards']==42
                with (out/'threshold_frozen.json').open('x') as f:json.dump(frozen,f,indent=2)
                report['threshold_receipt_sha256']=digest(out/'threshold_frozen.json');sequence+=1
                events.write(json.dumps({'sequence':sequence,'event':'threshold_frozen','sha256':report['threshold_receipt_sha256']})+'\n');events.flush()
            for row in plan['rows']:
                if row['split']!=split:continue
                if split=='development':assert digest(out/'threshold_frozen.json')==report['threshold_receipt_sha256']
                assert digest(data/row['image'])==row['encoded_sha256']
                with Image.open(data/row['image']) as im:
                    assert im.size==(row['width'],row['height']);t=transform(*im.size,896,row['roi_xyxy'])
                    x=torch.from_numpy(image_to_model(im,t))[None].cuda()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):v=model(x,refs,boxes,[t['valid_hw']],reference_taps=rt)[0]
                xy,q=v['points'].cpu().numpy(),v['probabilities'].cpu().numpy()
                assert np.isfinite(xy).all() and np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
                filename=row['image_id']+'.npz';np.savez_compressed(out/filename,points=xy,probabilities=q)
                e={'image_id':row['image_id'],'split':split,'file':filename,'sha256':digest(out/filename),'transform':t};report['predictions'].append(e)
                if split=='train':train_q[row['image_id']]=q.copy()
                report['query_forwards']+=1;sequence+=1;events.write(json.dumps({'sequence':sequence,'event':'prediction','split':split,'image_id':row['image_id'],'sha256':e['sha256']})+'\n');events.flush();write_json(out/'status.json',report)
    assert report['query_forwards']==65
    report.update(state='completed',head_final_sha256=state_digest({k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')}),encoder_final_sha256=state_digest(model.encoder.state_dict()),finite_model=all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),elapsed_seconds=time.monotonic()-start)
    assert report['head_final_sha256']==head and report['encoder_final_sha256']==encoder and report['finite_model']
    write_json(out/'report.json',report);write_json(out/'status.json',report)
    write_json(out/'result_hashes.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name not in ['status.json','result_hashes.json']})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--assets',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    try:run(args)
    except Exception:write_json(Path(args.output)/'failure.json',{'traceback':traceback.format_exc()});raise
