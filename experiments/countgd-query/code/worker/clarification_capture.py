"""Bounded inference only. Uses existing native authors' code and verified weights."""
import argparse
import ast
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from clarification_common import model_input, read, rows, save, sha, validate_inputs


def author_functions(repo):
    # Same AST extraction as the already-reproduced global CountGD++ wrapper.
    path = repo / 'test_dataset.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    for node in nodes:
        for call in ast.walk(node):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == 'torch'
                    and call.func.attr == 'load'):
                call.keywords.append(ast.keyword(arg='weights_only', value=ast.Constant(False)))
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    space = {'__name__': 'countgdpp_author_definitions', 'CONF_THRESH': .23}
    exec(compile(module, str(path), 'exec'), space)
    return SimpleNamespace(**space)


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class CountGD:
    def __init__(self):
        prepared = read('/research/comparator-preparation/countgdpp-v1/prepared.json')
        repo = Path(prepared['release'])
        assert prepared['operator_reference_parity']
        assert (repo / 'checkpoints/bert-base-uncased').exists()
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / 'models/GroundingDINO/ops'))
        self.author = author_functions(repo)
        self.model, self.transform = self.author.build_model_and_transforms(
            SimpleNamespace(device='cuda', pretrain_model_path=prepared['checkpoint']))
        state = torch.load(prepared['checkpoint'], map_location='cpu', weights_only=False)['model']
        missing = set(self.model.state_dict()) - set(state)
        assert all(k.endswith('position_ids') for k in missing), missing
        del state
        self.model = self.model.cuda().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def predict(self, inp, variant, pair=None, reference=False):
        assert set(inp) == {'image', 'exemplars'}
        with Image.open(inp['image']) as f:
            image = f.convert('RGB')
        prompts = {'image': image, 'points': [[*b[:2], 2., *b[2:], 3.] for b in inp['exemplars']]}
        normalized, scores = self.author.predict(self.model, self.transform, image, '', prompts, [], [], torch.device('cuda'))
        normalized = np.asarray(normalized).reshape(-1, 4)
        xywh = normalized.copy()
        xywh[:, :2] -= normalized[:, 2:] / 2
        xywh *= np.asarray([image.width, image.height, image.width, image.height])
        boxes = xywh.copy()
        boxes[:, 2:] += boxes[:, :2]
        # Record fixed author transform identity; no extra model forward.
        x, xe, b = self.author.preprocess(self.transform, image, prompts)
        return {'boxes': boxes.tolist(), 'scores': np.asarray(scores).reshape(-1).tolist(),
                'image_tensor_sha256': tensor_hash(x), 'exemplar_image_tensor_sha256': tensor_hash(xe),
                'transformed_boxes': b.tolist(), 'tensor_shape': list(x.shape),
                'reference_path': 'unchanged author.predict and legacy global wrapper coordinate conversion'}


class Geco:
    def __init__(self):
        from counting.geco import GeCo2
        self.adapter = GeCo2('/research/baselines/geco2',
                            '/research/models/GeCo2-FSCD147/geco2_fscd147.pth',
                            '/research/models/SAM2-base-plus/sam2_hiera_base_plus.pt')
        self.model = self.adapter.model
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def predict(self, inp, variant, pair=None, reference=False):
        from counting.geometry import postprocess
        assert set(inp) == {'image', 'exemplars'}
        with Image.open(inp['image']) as f:
            image = f.convert('RGB')
        fixed = variant == 'three_pair_scale'
        x, b, scale, content = self.adapter.preprocess(image, pair if fixed else inp['exemplars'])
        if fixed:
            b = torch.tensor(inp['exemplars'], dtype=torch.float32, device=x.device).unsqueeze(0) * scale
        if reference:
            assert not fixed
            prediction = self.adapter.predict(inp)  # Existing unchanged native wrapper.
            boxes, scores = prediction['boxes'], prediction['scores']
        else:
            outputs, *_ = self.model(x, b)
            raw = outputs[0]
            cpu = lambda value: value.detach().float().cpu().numpy()
            boxes, scores = postprocess(cpu(raw['pred_boxes']), cpu(raw['box_v']), cpu(raw['scores']), scale, content)
            boxes, scores = boxes.tolist(), scores.tolist()
        return {'boxes': boxes, 'scores': scores, 'image_tensor_sha256': tensor_hash(x),
                'transformed_boxes': b.squeeze(0).cpu().tolist(), 'scale': scale,
                'content_wh': list(content), 'tensor_shape': list(x.shape),
                'reference_path': 'counting.geco.GeCo2.predict' if reference else 'native forward/native postprocess'}


def check_prediction(p, width, height):
    boxes = np.asarray(p['boxes'], dtype=float).reshape(-1, 4)
    scores = np.asarray(p['scores'], dtype=float)
    assert scores.shape == (len(boxes),)
    assert np.isfinite(boxes).all() and np.isfinite(scores).all()
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    assert ((centers >= -1e-6) & (centers <= np.array([width, height]) + 1e-6)).all()
    assert (boxes[:, 2:] >= boxes[:, :2]).all()


def run(a):
    job = Path(a.job)
    out = job / (a.model + '_' + a.mode)
    out.mkdir()  # Exclusive, never resume or replace an existing result.
    calls = 0
    start = time.monotonic()
    try:
        p = read(job / 'protocol.json')
        for rel, expected in read(job / 'release_hashes.json').items():
            assert sha(Path(__file__).parent / rel) == expected, rel
        for rel, expected in read(job / 'input_hashes.json').items():
            assert sha(job / rel) == expected, rel
        external = read(job / 'external_hashes.json')[a.model]
        for path, expected in external.items():
            assert sha(path) == expected, path
        initial, reveal = rows(job / 'initial_inputs.jsonl'), rows(job / 'reveal_inputs.jsonl')
        validate_inputs(initial, reveal)
        byid = {q['query_id']: q for q in reveal}
        if a.mode == 'preflight':
            initial = [q for q in initial if q['query_id'] in p['compatibility_query_ids']]
        for q in initial:
            assert sha(q['image']) == q['image_sha256'], q['query_id']
        torch.set_num_threads(2)
        torch.manual_seed(42 if a.model == 'countgd' else 2027)
        np.random.seed(42 if a.model == 'countgd' else 2027)
        random.seed(42 if a.model == 'countgd' else 2027)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_grad_enabled(False)
        save(out / 'status.json', {'state': 'loading', 'calls': 0, 'optimizer_steps': 0})
        model = CountGD() if a.model == 'countgd' else Geco()
        variants = ['two', 'three'] if a.model == 'countgd' else ['two', 'three', 'three_pair_scale']
        cap = len(initial) * len(variants) * (2 if a.mode == 'preflight' else 1)
        state_versions = {k: v._version for k, v in model.model.named_parameters()}
        checks = []
        with (out / 'predictions.jsonl').open('x') as f:
            for q in initial:
                per_query = {}
                for variant in variants:
                    inp = model_input(q, None if variant == 'two' else byid[q['query_id']]['exemplars'])
                    for repeat in range(2 if a.mode == 'preflight' else 1):
                        assert calls < cap and not torch.is_grad_enabled()
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        # The native three-shot repetition uses the old GeCo wrapper.
                        result = model.predict(inp, variant, pair=q['exemplars'],
                                               reference=a.mode == 'preflight' and repeat == 1 and variant == 'three')
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - before
                        calls += 1
                        check_prediction(result, q['width'], q['height'])
                        if repeat:
                            for key in ('boxes', 'scores', 'transformed_boxes'):
                                np.testing.assert_allclose(result[key], first[key], rtol=1e-5, atol=1e-6)
                            assert result['image_tensor_sha256'] == first['image_tensor_sha256']
                            checks.append({'query_id': q['query_id'], 'variant': variant, 'repeat_and_native_parity': True})
                        else:
                            first = result
                            per_query[variant] = result
                        record = dict(query_id=q['query_id'], variant=variant, repeat=repeat,
                                      model=a.model, pred_count=len(result['boxes']), **result,
                                      seconds_shared_GPU=elapsed, optimizer_steps=0,
                                      model_input_sha256=hashlib.sha256(json.dumps(inp, sort_keys=True).encode()).hexdigest())
                        f.write(json.dumps(record, allow_nan=False) + '\n')
                        f.flush()
                        save(out / 'status.json', dict(state='capturing', calls=calls, prescribed_calls=cap,
                                                      query_id=q['query_id'], variant=variant, optimizer_steps=0))
                if a.model == 'countgd':
                    assert per_query['two']['image_tensor_sha256'] == per_query['three']['image_tensor_sha256']
                    assert per_query['two']['exemplar_image_tensor_sha256'] == per_query['three']['exemplar_image_tensor_sha256']
                else:
                    assert per_query['two']['image_tensor_sha256'] == per_query['three_pair_scale']['image_tensor_sha256']
                    assert per_query['two']['scale'] == per_query['three_pair_scale']['scale']
                print(json.dumps({'model': a.model, 'mode': a.mode, 'calls': calls, 'cap': cap}), flush=True)
        assert calls == cap
        assert state_versions == {k: v._version for k, v in model.model.named_parameters()}
        assert all(torch.isfinite(v).all().item() for v in model.model.parameters())
        report = dict(state='completed', model=a.model, mode=a.mode, query_count=len(initial),
                      calls=calls, optimizer_steps=0, persistent_steps=0, parameters_unmodified=True,
                      finite_parameters=True, checks=checks, elapsed_seconds_shared_GPU=time.monotonic()-start,
                      max_cuda_allocated=torch.cuda.max_memory_allocated(),
                      max_cuda_reserved=torch.cuda.max_memory_reserved(), protocol_sha256=sha(job/'protocol.json'),
                      worker_sha256=sha(__file__), predictions_sha256=sha(out/'predictions.jsonl'),
                      external_files_verified=len(external), truth_loaded=False)
        save(out / 'report.json', report)
        save(out / 'status.json', report)
        save(out / 'result_hashes.json', {name: sha(out/name) for name in ['report.json', 'predictions.jsonl', 'status.json']})
    except Exception:
        save(out / 'status.json', dict(state='failed', calls=calls, optimizer_steps=0, error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job', default='/pipeline')
    p.add_argument('--model', choices=['countgd', 'geco'], required=True)
    p.add_argument('--mode', choices=['preflight', 'bank'], required=True)
    run(p.parse_args())
