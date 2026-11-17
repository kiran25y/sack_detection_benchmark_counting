"""Preserved poststep guard under the validated stable-resampling runtime."""
from .stable_resampling import install

import json
import pathlib
import sys

import torch

from .assignment_continuation import CHECKPOINT_SHA, QUERY_IDS
from .assignment_cudnn_preflight import main as guarded_updates, state_digest
from .confidence_assignment_inspection import sha
from .model import IPICNet
from .preflight import prepare, tensor_digest


def main():
    root=pathlib.Path(sys.argv[sys.argv.index('--root')+1]);out=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
    source=root/'research-direction-20261012/ipic_checks_20261030/pointdiag_v1/resume.pt'
    if sha(source)!=CHECKPOINT_SHA:raise ValueError('Checkpoint changed')
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    rows={r['query_id']:r for r in json.loads((root/'manifests/fsc147_train_groupclean.json').read_text())['queries']}
    ck=torch.load(source,map_location='cpu',weights_only=True)
    model=IPICNet(False).cuda().train();model.load_state_dict(ck['model'],strict=True);del ck
    identity=tensor_digest(model);reference=[];parity=[]
    torch.use_deterministic_algorithms(False)
    with torch.no_grad():
        for q in QUERY_IDS:
            b,_=prepare(rows[q],root,'cuda',1024)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                proto=model.encode_references(b['references'],b['reference_boxes']);v=model(b['a'],b['valid_a'],proto)
            reference.append({k:x.detach().cpu().clone() for k,x in v.items()})
        runtime=install()
        for q,expected in zip(QUERY_IDS,reference):
            b,_=prepare(rows[q],root,'cuda',1024)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                proto=model.encode_references(b['references'],b['reference_boxes']);v=model(b['a'],b['valid_a'],proto)
            for k,x in v.items():torch.testing.assert_close(x.cpu(),expected[k],rtol=0,atol=0)
            parity.append({'query_id':q,'all_forward_fields_exact':True,'field_digest':state_digest(expected)})
    if tensor_digest(model)!=identity:raise ValueError('Forward-only parity changed model')
    with (out/'runtime_amendment.json').open('x') as f:json.dump({'runtime':runtime,'initial_native_vs_stable_forward':parity,'optimizer_calls':0,'four_disposable_guard_updates_follow':True},f,indent=2)
    del model,reference,b,proto,v;torch.cuda.empty_cache()
    guarded_updates()


if __name__=='__main__':main()
