"""Verify the extracted review package. Python 3.10+, NumPy >=1.22, SciPy."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
def read(path):
    return json.loads((root/path).read_text(encoding='utf8'))

def check_file(name, *args):
    subprocess.run([sys.executable,'-B',str(root/name),*map(str,args)],check=True,cwd=root)

manifest = read('SHA256.json')
for name, expected in manifest.items():
    with (root/name).open('rb') as f:
        digest = sha256()
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    assert digest.hexdigest()==expected, name
print('Verified SHA-256 for each payload.',flush=True)
check_file('replay.py')
check_file('vote_scale.py')
check_file('verify_paper.py')
check_file('verify_factorial.py')
check_file('full_factorial/sack_cross_seed_review.py','--reviewer',root,'--banks',root/'full_factorial',
           '--check',root/'cross_seed_results.json')
initial=read('annotation_review/initial_points.json')
submitted=read('annotation_review/owner_points.json')
locked=read('labels/test.json')
assert initial.keys()==submitted.keys()==locked.keys() and len(locked)==51
def points(seq):
    return Counter(tuple(x) for x in seq)
retained=sum((points(initial[k])&points(locked[k])).total() for k in locked)
assert (sum(map(len,initial.values())),sum(map(len,locked.values())),retained)==(589,644,589)
assert sum(points(initial[k])==points(locked[k]) for k in locked)==36
assert [k for k in locked if points(submitted[k])!=points(locked[k])]==['H025']
assert len(submitted['H025'])==5 and not locked['H025']
print('Verified draft-to-final point provenance and the H025 adjudication. All review checks passed.')

