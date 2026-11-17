"""Evaluate the preregistered training-only visual verifier at matched move counts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import hashlib
import json
from pathlib import Path
import numpy as np
from prepare_firc_visual_swap import OUT,ROOT,FIRC,PRIOR,PROTOCOL,read,write,now,sha
from run_firc_uncovered_reallocation import summaries
from analyze_firc_tiny import evaluate,maximum_matches

PRIMARY='center_and_ring'
CONTROLS=('matched_q','matched_coverage_prefix','matched_center_only','matched_shuffled')
OLD=('confidence_71','centering_75','firc_coverage_only')

def normalize(z):
    return z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-8)

def feature_database(frames):
    result={}
    report=read(OUT/'features/report.json')
    gallery={}
    for row in frames:
        key=row['image_id'];meta=report['features'][key];path=OUT/'features'/meta['file']
        assert sha(path)==meta['sha256']
        array=np.load(path,allow_pickle=False).astype(np.float32)
        assert array.shape==(len(row['sample_native']),2,768) and np.isfinite(array).all()
        array=normalize(array)
        n=row['seeds']+len(row['bank']);result[key]=array[:n]
        if row['split']=='train':
            gallery[key]=(row['dependency_group'],array[n:n+row['positives']],array[n+row['positives']:])
        else:assert len(array)==n
    return result,gallery

def scores(query,gallery,exclude):
    """One best similarity per distinct reference image and class, then three-image mean."""
    if not len(query):return np.empty(0),np.empty(0)
    per_class=[[],[]];center_class=[[],[]]
    for group,positive,negative in gallery.values():
        if group==exclude:continue
        for label,reference in enumerate((positive,negative)):
            if not len(reference):continue
            a=query[:,0]@reference[:,0].T
            b=query[:,1]@reference[:,1].T
            per_class[label].append(((a+b)*.5).max(axis=1))
            center_class[label].append(a.max(axis=1))
    def margin(classes):
        means=[]
        for items in classes:
            assert len(items)>=3
            means.append(np.sort(np.stack(items),axis=0)[-3:].mean(axis=0,dtype=np.float64))
        return means[0]-means[1]
    return margin(per_class),margin(center_class)

def order(score,desc=False):
    return np.lexsort((np.arange(len(score)),-score if desc else score))

def main():
    features_ok=read(OUT/'feature_verification.json');assert features_ok['completed']
    prep=read(OUT/'preparation.json')
    assert sha(OUT/'queries.json')==prep['queries_sha256'] and all(sha(ROOT/p)==h for p,h in prep['hashes'].items())
    plan=read(OUT/'queries.json');rows=plan['frames']
    assert plan['protocol_sha256']==sha(PROTOCOL)
    write(OUT/'readout_freeze.json',{'time_kst':now(),'protocol_sha256':sha(PROTOCOL),'code_sha256':sha(Path(__file__)),
          'queries_sha256':sha(OUT/'queries.json'),'features_sha256':sha(OUT/'features/report.json'),'primary':PRIMARY})
    query,gallery=feature_database(rows);assert len(gallery)==42
    old={v['image_id']:v for v in map(json.loads,(PRIOR/'predictions.jsonl').read_text().splitlines())}
    assert sha(PRIOR/'predictions.jsonl')==read(PRIOR/'result.json')['predictions_sha256']
    with (OUT/'predictions.jsonl').open('x') as stream:
        for row in rows:
            k=row['image_id'];n=row['seeds'];group=row['dependency_group']
            full,center=scores(query[k],gallery,group if row['split']=='train' else None)
            donors=order(full[:n]);receivers=order(full[n:],True)
            limit=row['move_cap'];assert limit<=min(len(donors),len(receivers))
            number=sum(full[n+receivers[j]]>full[donors[j]]+1e-9 for j in range(limit))
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(k.encode()).digest()[:8],'big'))
            misaligned=np.r_[rng.permutation(full[:n]),rng.permutation(full[n:])]
            arms=old[k]['arms'];prefix=arms['firc_coverage_only']['moves'][:number]
            index={i:j for j,i in enumerate(row['bank'])}
            rules={PRIMARY:(donors[:number],receivers[:number]),
                   CONTROLS[0]:(order(np.array(row['seed_q']))[:number],order(np.array(row['bank_q']),True)[:number]),
                   CONTROLS[1]:([v['slot'] for v in prefix],[index[v['receiver_candidate']] for v in prefix]),
                   CONTROLS[2]:(order(center[:n])[:number],order(center[n:],True)[:number]),
                   CONTROLS[3]:(order(misaligned[:n])[:number],order(misaligned[n:],True)[:number])}
            positions=np.asarray(row['sample_native'],float).reshape(-1,2)
            outputs={m:{'points':arms[m]['native_points'],'moves':arms[m]['moves']} for m in OLD}
            for method,(remove,replace) in rules.items():
                points=positions[:n].copy();moves=[]
                for d,j in zip(remove,replace):
                    d,j=int(d),int(j);points[d]=positions[n+j]
                    moves.append({'slot':d,'bank_index':j,'candidate':row['bank'][j],'donor_visual_margin':float(full[d]),
                                  'destination_visual_margin':float(full[n+j])})
                assert len(moves)==number and len(points)==n
                outputs[method]={'points':points.tolist(),'moves':moves}
            stream.write(json.dumps({'image_id':k,'split':row['split'],'group':group,'cap':limit,'number':int(number),
                    'visual_scores':full.tolist(),'center_only_scores':center.tolist(),'outputs':outputs},allow_nan=False)+'\n')
    ph=sha(OUT/'predictions.jsonl');write(OUT/'predictions_frozen.json',{'time_kst':now(),'predictions_sha256':ph})
    truth_plan=read(FIRC/'data/plan.json');truth={r['image_id']:r for r in read(ROOT/truth_plan['local_truth_manifest'])}
    records={m:[] for m in (*OLD,PRIMARY,*CONTROLS)}
    with (OUT/'predictions.jsonl').open() as stream:
        for item in map(json.loads,stream):
            row=truth[item['image_id']];gt=np.array(row['points']).reshape(-1,2);scale=1024/max(row['width'],row['height'])
            for m,output in item['outputs'].items():
                pts=np.asarray(output['points'],float).reshape(-1,2);v=evaluate(pts,gt,row)
                v.update(split=row['split'],tp_at4=maximum_matches(pts*scale,gt*scale,4),tp_at16=maximum_matches(pts*scale,gt*scale,16),moves=len(output['moves']))
                records[m].append(v)
    old_result=read(PRIOR/'result.json')
    for m in OLD:
        byid={r['image_id']:r for r in records[m]}
        for r in old_result['per_image'][m]:assert all(byid[r['image_id']][key]==val for key,val in r.items())
    summary={m:summaries(v) for m,v in records.items()};primary=summary[PRIMARY];dv=primary['development']
    gains={g:{m:primary['groups'][g]['tp']-summary[m]['groups'][g]['tp'] for m in OLD} for g in primary['groups']}
    protections={}
    for m in OLD[:2]:
        b=summary[m];checks={'counts':all(i['count']==j['count'] and i['ae']==j['ae'] for i,j in zip(records[m],records[PRIMARY])),
          'low_mae':primary['low']['mae']<=b['low']['mae'],'low_f1':primary['low']['positive_macro_f1_at8']>=b['low']['positive_macro_f1_at8'],
          'empty_fp':dv['empty_false_positives']<=b['development']['empty_false_positives']}
        for x in ('game1','game2'):
            checks[x]=dv[x]<=b['development'][x]+max(.1,.01*b['development'][x])+1e-12
            checks['low_'+x]=primary['low'][x]<=b['low'][x]+1e-12
        protections[m]=checks
    group_pass=all(g[m]>0 for g in gains.values() for m in OLD[:2])
    pooled=all(dv['pooled_f1_at8']-summary[m]['development']['pooled_f1_at8']>=.02-1e-12 for m in OLD[:2])
    specificity={m:dv['pooled_f1_at8']-summary[m]['development']['pooled_f1_at8']>=.02-1e-12 for m in CONTROLS}
    result={'completed_kst':now(),'predictions_sha256':ph,'primary':PRIMARY,'summary':summary,'per_image':records,
        'gains_by_group':gains,'strict_gain_vs_71_75_both_groups':group_pass,'strict_gain_vs_firc_only_both_groups':all(g[OLD[-1]]>0 for g in gains.values()),
        'pooled_gain_vs_71_75':pooled,'protections':protections,'passed':group_pass and pooled and all(all(s.values()) for s in protections.values()),
        'matched_control_margins':specificity,'visual_specificity':all(specificity[m] for m in (CONTROLS[0],CONTROLS[1],CONTROLS[3])),
        'ring_specificity':specificity[CONTROLS[2]],'new_encoder_forwards':65,'optimizer_updates':0,'reserved_scored':0}
    write(OUT/'result.json',result)
    for m,s in summary.items():
        print(m,'TP',s['development']['tp'],'groups',[v['tp'] for v in s['groups'].values()],'F1',s['development']['pooled_f1_at8'],'GAME',s['development']['game1'],s['development']['game2'])
    print('both',group_pass,'pooled',pooled,'protections',protections,'specificity',specificity)

if __name__=='__main__':main()
