"""Freeze draws and audit deterministic inputs, with no neural network calls."""
import argparse,collections,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from .adaptation_data import Data,sha
from .transforms import MEAN,STD

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--package',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=True);torch.set_num_threads(2)
    if (out/'draws.jsonl').exists():raise ValueError('Exclusive audit')
    data=Data(a.root,a.package);draws=data.draws()
    (out/'draws.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in draws))
    # Stratified deterministic manual review set:200 composites, all intervention families.
    selected=[]
    for family,number in [('add_remove',50),('replacement',30),('occlusion',40),('distractor',40),('appearance',20),('geometry',20)]:
        selected.extend([d['draw_id'] for d in draws if d['family']==family][:number])
    selected=set(selected);assert len(selected)==200
    witnesses=set()
    for idx in range(len(data.train)):witnesses.add(next(d['draw_id'] for d in draws if d['dataset']=='sacks' and d['row_index']==idx))
    for source in ('FSC147','D2S'):
        witnesses.update([d['draw_id'] for d in draws if d['dataset']==source and not d.get('composite')][:32])
    inspect=selected|witnesses
    # Reuse byte-verified input records from the superseded, overly broad audit.
    previous=Path(a.root)/'research-direction-20261012/ipic_sack_pair_audit_v4_20261103'
    reuse={}
    if (previous/'result/prepared_ledger.jsonl').exists():
        frozen=json.loads((previous/'frozen_hashes.json').read_text())
        for name in ('adaptation_data.py','renderer.py','public_data.py','transforms.py','adaptation_core.py'):
            if frozen['code/counting/ipic/'+name]!=sha(Path(__file__).with_name(name)):raise ValueError('Cannot reuse different input code')
        if sha(previous/'result/draws.jsonl')!=sha(out/'draws.jsonl'):raise ValueError('Cannot reuse different draws')
        for name in ('sack_manifest.json','asset_background_review.json','references.json'):
            if sha(previous/'data'/name)!=sha(Path(a.package)/name):raise ValueError('Cannot reuse different inputs')
        for r in map(json.loads,(previous/'result/prepared_ledger.jsonl').read_text().splitlines()):reuse[r['draw']['draw_id']]=r
    report=dict(state='running',checked=0,total=len(inspect),planned_draws=len(draws),optimizer_calls=0,model_calls=0,manual_review_pending=True,
        prepared_witness_ids=sorted(witnesses),manual_pair_ids=sorted(selected),reused_checked_draws=0,
        input_hashes={},audit_pairs=[],draws_sha256=sha(out/'draws.jsonl'),failures=[])
    def save():
        temp=out/'status.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(out/'status.json')
    save();page=0;tiles=[]
    try:
        for d in [d for d in draws if d['draw_id'] in inspect]:
            reused=d['draw_id'] in reuse
            if reused:record=reuse[d['draw_id']];batch=None;report['reused_checked_draws']+=1
            else:batch,record=data.prepare(d)
            # Independently preserve per-view array digests to compare replay and all arms.
            if not reused:
                record['array_hashes']={k:__import__('hashlib').sha256(batch[k].numpy().tobytes()).hexdigest()
                    for k in ['a','b','valid_a','valid_b','points_a','points_b']}
            with (out/'prepared_ledger.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if d['draw_id'] in selected:
                if reused:
                    tile=Image.open(previous/'result'/f"pair_{d['draw_id']:04}.jpg").convert('RGB')
                else:tile=Image.new('RGB',(1024,420),'white')
                dr=ImageDraw.Draw(tile)
                dr.text((5,5),f"draw {d['draw_id']} {d['family']} all={d['all_products']} targets {record['count_a']} -> {record['count_b']}",fill='black')
                for j,side in ([] if reused else enumerate(('a','b'))):
                    ar=batch[side][0].numpy();im=Image.fromarray(np.uint8(np.clip(ar.transpose(1,2,0)*STD+MEAN,0,1)*255+.5))
                    drim=ImageDraw.Draw(im)
                    for x,y in batch['points_'+side].numpy():drim.ellipse((x-3,y-3,x+3,y+3),outline='#00ff44',width=2)
                    im.thumbnail((510,380));tile.paste(im,(j*512,32))
                tile.save(out/f"pair_{d['draw_id']:04}.jpg",quality=95);tiles.append(tile)
                report['audit_pairs'].append(dict(draw_id=d['draw_id'],family=d['family'],file=f"pair_{d['draw_id']:04}.jpg",
                    counts=[record['count_a'],record['count_b']],category=record.get('category'),instance_ledger=record.get('instance_ledger'),edit=record.get('edit')))
                if len(tiles)==10:
                    sheet=Image.new('RGB',(2048,2100),'white')
                    for i,im in enumerate(tiles):sheet.paste(im,((i%2)*1024,(i//2)*420))
                    sheet.save(out/f'audit_page_{page:02}.jpg',quality=94);page+=1;tiles=[]
            report['checked']+=1
            if report['checked']%25==0:save()
            del batch,record
        assert not tiles and len(report['audit_pairs'])==200
        report.update(state='completed',prepared_ledger_sha256=sha(out/'prepared_ledger.jsonl'),
            source_draws=dict(collections.Counter(d['dataset'] for d in draws)),D2S_all_products=sum(d['all_products'] for d in draws),
            D2S_zero_queries=sum(d.get('verified_zero_query',False) for d in draws),sack_point_only=True)
        save();(out/'report.json').write_text(json.dumps(report,indent=2))
    except Exception as exc:report.update(state='failed',error=repr(exc),failed_draw=d);save();raise
if __name__=='__main__':main()
