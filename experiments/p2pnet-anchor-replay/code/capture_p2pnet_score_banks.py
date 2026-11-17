"""Recover all pre-threshold point coordinates and scores with the archived models."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from models import build_model
from prior_worker import load, tensor, write


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    job=Path('/job')
    out=Path('/output')
    previous=Path('/prior')
    plan=json.loads((job/'design.json').read_text())
    model=build_model(SimpleNamespace(backbone='vgg16_bn',row=2,line=2),training=False).cuda().eval()
    hashes={}
    for arm,sig in plan['weights'].items():
        path=previous/('worker/weights/SHTechA.pth' if arm=='pretrained' else 'result/'+arm+'.pt')
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sig
        state=torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'] if arm=='pretrained' else state,strict=True)
        del state
        sealed_path=previous/'result'/(arm+'.json')
        with sealed_path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==plan['sealed_hashes'][arm+'.json']
        sealed=json.loads(sealed_path.read_text())
        for step,r in enumerate(plan['rows'],1):
            assert 'points' not in r
            rgb,_,scale=load(r)
            output=model(tensor(rgb)[None])
            score=output['pred_logits'][0].softmax(-1)[:,1].cpu().numpy()
            xy=output['pred_points'][0].cpu().numpy()
            assert xy.shape==(len(score),2) and np.isfinite(xy).all() and np.isfinite(score).all()
            assert len(score)==(rgb.shape[0]//8)*(rgb.shape[1]//8)*4
            native=((xy[score>.5]+.5)/scale+np.asarray(r['roi'][:2])-.5)
            expected=np.asarray(sealed[r['id']]).reshape(-1,2)
            assert native.shape==expected.shape and np.allclose(native,expected,rtol=0,atol=1e-5)
            name=arm+'_'+r['id']+'.npz'
            np.savez_compressed(out/name,xy=xy,score=score,scale=scale)
            with (out/name).open('rb') as f:hashes[name]=hashlib.file_digest(f,'sha256').hexdigest()
            if step%10==0 or step==1:write('status.json',dict(arm=arm,frames=step))
    assert len(hashes)==296
    write('report.json',dict(state='completed',optimizer_calls=0,forward_calls=296,verified_native_predictions=296,hashes=hashes))


if __name__=='__main__':
    main()
