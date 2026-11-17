"""Four-cell data adapter; original data/geometry is the default path."""
import hashlib,json
from pathlib import Path
from .adaptation_data import Data,sha
from .transfer_protocol import CELLS,validate_manifest,conservative_appearance

class TransferData(Data):
    def __init__(self,root,package,cell):
        if cell not in CELLS:raise ValueError('Unknown fixed cell')
        super().__init__(root,package);self.cell=cell
    def prepare(self,draw,device='cpu'):
        if draw['dataset']=='sacks' and CELLS[self.cell]['coverage']=='original':
            if self.train[draw['row_index']]['dependency_group']!='ipic_dependency_b778d9b86516':raise ValueError('Additional group in original control')
        fn=conservative_appearance if CELLS[self.cell]['appearance']=='conservative' else None
        return super().prepare(draw,device,sack_appearance_function=fn)

def code_digest():
    return hashlib.sha256(json.dumps({p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},sort_keys=True).encode()).hexdigest()

def evidence(root,package):
    package=Path(package);spec=json.loads((package/'pilot_plan.json').read_text())
    if spec['stage']!='ipic_transfer_2x2' or spec['cells']!=CELLS or spec['updates_per_cell']!=200 or spec['total_updates']!=800 or spec['warmup']!=20 or spec['thresholds']!=[.5,.1] or spec['threshold_selection'] is not False:raise ValueError('Frozen pilot recipe changed')
    for n,h in spec['input_hashes'].items():
        if sha(package/n)!=h:raise ValueError('Frozen input changed: '+n)
    rows=json.loads((package/'sack_manifest.json').read_text());coverage=validate_manifest(rows,json.loads((package/'approved_group_allocation.json').read_text()))
    if coverage!=spec['coverage'] or json.loads((package/'label_admission.json').read_text())['passed'] is not True:raise ValueError('Data readiness mismatch')
    draws={c:list(map(json.loads,(package/f'draws_{c}.jsonl').read_text().splitlines())) for c in CELLS}
    for c,dd in draws.items():
        if len(dd)!=1600 or [d['draw_id'] for d in dd]!=list(range(1600)):raise ValueError('Draw sequence mismatch')
    identities=dict(plan=sha(package/'pilot_plan.json'),source_release=code_digest())
    return spec,draws,identities
