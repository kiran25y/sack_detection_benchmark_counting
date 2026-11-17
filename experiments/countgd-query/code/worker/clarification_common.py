"""Information boundary and scoring for the frozen exemplar-reveal feasibility study."""
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def save(path, value):
    p = Path(path)
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    t.replace(p)


def model_input(row, exemplars=None):
    """Model boundary: image and supplied visual examples only, never IDs or truth."""
    boxes = row['exemplars'] if exemplars is None else exemplars
    if len(boxes) not in (2, 3) or len({tuple(b) for b in boxes}) != len(boxes):
        raise ValueError('Require genuine distinct two/three native exemplars')
    for b in boxes:
        if len(b) != 4 or not all(math.isfinite(v) for v in b):
            raise ValueError('Invalid box')
        if not (0 <= b[0] < b[2] <= row['width'] and 0 <= b[1] < b[3] <= row['height']):
            raise ValueError('Exemplar outside native image')
    return {'image': row['image'], 'exemplars': boxes}


def validate_inputs(initial, reveal):
    assert len(initial) == len(reveal) == 128
    assert len({q['query_id'] for q in initial}) == 128
    assert len({q['group_id'] for q in initial}) == 128
    assert [q['query_id'] for q in initial] == [q['query_id'] for q in reveal]
    allowed = {'query_id', 'group_id', 'image', 'image_sha256', 'width', 'height', 'exemplars', 'prompt_ids'}
    for a, b in zip(initial, reveal):
        assert set(a) == allowed and set(b) == {'query_id', 'exemplars', 'prompt_ids'}
        assert len(a['exemplars']) == 2 and len(b['exemplars']) == 3
        assert all(x in b['exemplars'] for x in a['exemplars'])
        model_input(a)
        model_input(a, b['exemplars'])


def stratum(n):
    return 'sparse' if n < 25 else 'medium25' if n < 75 else 'medium75' if n < 150 else 'dense' if n < 900 else 'extreme'


def game(points, centers, width, height, level):
    """Point-mass GAME, with closed outer boundary and half-open interior cells."""
    side = 2 ** level
    def histogram(values):
        h = [0] * (side * side)
        for x, y in values:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError('Nonfinite location')
            # Native endpoint roundoff only; never filter predictions using GT.
            if not (-1e-6 <= x <= width + 1e-6 and -1e-6 <= y <= height + 1e-6):
                raise ValueError('Point center outside image')
            ix = min(side - 1, max(0, int(x / width * side)))
            iy = min(side - 1, max(0, int(y / height * side)))
            h[iy * side + ix] += 1
        return h
    a, b = histogram(points), histogram(centers)
    return sum(abs(x - y) for x, y in zip(a, b))
