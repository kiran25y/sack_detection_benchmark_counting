"""Inference-only on all 20 existing training images. No optimizer or selection."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from counting.ipic.model import IPICNet
from counting.ipic.adaptation_data import Data, warp_ids
from counting.ipic.transforms import resize_pad
from counting.ipic.stable_adaptation_runtime import install
from counting.ipic.losses import supervised_loss
from counting.ipic.foundation import write_json

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def state_hash(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():
        h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--package',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    root=Path(args.root);package=Path(args.package);out=Path(args.output);out.mkdir(exist_ok=False)
    spec=json.loads((package/'probe_spec.json').read_text());torch.set_num_threads(2);install();data=Data(root,package)
    assert len(data.train)==20 and {r['image_id'] for r in data.train}==set(spec['training_query_ids'])
    for n,h in spec['frozen_input_hashes'].items():assert sha(package/n)==h
    refs,boxes=data.references('filled_sack');report=dict(state='running',query_forwards=0,reference_batch_forwards=0,optimizer_steps=0,models={})
    write_json(out/'status.json',report)
    for name,item in spec['checkpoints'].items():
        checkpoint=root/item['path'];assert sha(checkpoint)==item['sha256']
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert state['optimizer_steps']==item['steps'] and state['seed']==1729
        if name!='foundation':assert state['arm']==name
        model=IPICNet(detail=False);model.load_state_dict(state['model'],strict=True);del state
        model.cuda().eval();before=state_hash(model);records=[];start=time.monotonic()
        with torch.inference_mode():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                proto=model.encode_references(torch.as_tensor(refs,device='cuda'),torch.as_tensor(boxes,device='cuda'))
            report['reference_batch_forwards']+=1
            for r in data.train:
                path=package/r['image'];assert sha(path)==r['encoded_sha256']
                image=data.image(path);rgb,valid,g=resize_pad(image,1024)
                roi=np.zeros((image.height,image.width),np.int32);x0,y0,x1,y1=r['roi_xyxy'];roi[y0:y1,x0:x1]=1
                valid &= warp_ids(roi,g.record(),0)==1
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    v=model(torch.as_tensor(rgb,device='cuda')[None],torch.as_tensor(valid,device='cuda'),proto)
                points=g.points(v['points'][v['valid']].cpu().numpy(),inverse=True);confidence=v['probabilities'][v['valid']].cpu().numpy()
                assert np.isfinite(points).all() and np.isfinite(confidence).all()
                # Labels used only AFTER prediction, as a fitting diagnostic.
                loss=supervised_loss(v,torch.as_tensor(g.points(r['points']),device='cuda',dtype=torch.float32))
                records.append(dict(query_id=r['image_id'],width=r['width'],height=r['height'],truth=r['points'],count=r['count'],
                      dependency_group=r['dependency_group'],points=points.tolist(),probabilities=confidence.tolist(),
                      supervised_diagnostic={k:float(loss[k]) for k in ['focal','point','soft_count']}))
                report['query_forwards']+=1;write_json(out/'status.json',dict(report,current_model=name,current_query=r['image_id']))
        after=state_hash(model);assert before==after
        with (out/(name+'_training_predictions.jsonl')).open('x') as f:
            for r in records:f.write(json.dumps(r)+'\n')
        report['models'][name]=dict(checkpoint_sha256=sha(checkpoint),state_hash_before=before,state_hash_after=after,
            queries=len(records),seconds=time.monotonic()-start,finite=True)
        del model,proto,records;torch.cuda.empty_cache()
    assert report['query_forwards']==80 and report['reference_batch_forwards']==4 and report['optimizer_steps']==0
    report['state']='completed';write_json(out/'status.json',report);write_json(out/'report.json',report)
    write_json(out/'result_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='result_hashes.json'})
if __name__=='__main__':main()
