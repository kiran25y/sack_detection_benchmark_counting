from pathlib import Path
import sys,json
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'full_factorial'))
import sack_factorial_replay as v
import sack_p2pnet_budget_analysis as pp
def read(p):return json.loads(p.read_text(encoding='utf8'))
d=root/'full_factorial'
assert v.run(root)==read(d/'factorial.json')
assert pp.run(d/'banks.tar',root,read(d/'design.json'),read(d/'capture_verified.json'))==read(d/'analysis.json')
print('Verified four verifier cells, multiplier, all P2PNet score cutoffs, raw coverage and both coordinate-bank comparisons.')
