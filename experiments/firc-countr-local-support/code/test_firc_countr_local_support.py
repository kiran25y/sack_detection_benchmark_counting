"""One frozen, paired test on existing FIRC points and CounTR densities."""
import datetime
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_firc_tiny import evaluate, summarize
from audit_firc_countr_misses import ROOT, FIRC, COUNTR, DATA, inventory, candidates, mass, sha, read
from firc_countr_support import support_image, rotated

RUN=ROOT/'artifacts/firc_countr_local_support_v1_20261108'
PLAN=ROOT/'study_protocols/firc_countr_local_support_20261108.json'
MODES=('point_only','count_only','rotated','aligned')

def write(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)

def timestamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()

def weak_peaks(xy,q,scale):
    base=np.flatnonzero(q>=.35)
    weak=np.flatnonzero((q>=.025)&(q<.35))
    weak=weak[np.lexsort((weak,-q[weak]))]
    coords=xy*scale
    buckets=defaultdict(list)
    def cell(i):
        return int(np.floor(coords[i,0]/8)),int(np.floor(coords[i,1]/8))
    for i in base:
        buckets[cell(i)].append(i)
    selected=[]
    for i in weak:
        x,y=cell(i)
        neighbors=[j for a in (x-1,x,x+1) for b in (y-1,y,y+1) for j in buckets.get((a,b),())]
        if neighbors and np.any(np.sum((coords[neighbors]-coords[i])**2,axis=1)<=64):continue
        selected.append(i)
        buckets[x,y].append(i)
    return base,np.array(selected,dtype=int)

def features(row,entry,mean_count):
    xy,q=candidates(entry)
    base,weak=weak_peaks(xy,q,1024/max(row['width'],row['height']))
    density=mass(row)
    with np.load(DATA/row['array'],allow_pickle=False) as z:mask=z['mask'].astype(bool)
    def sample(points):
        return support_image(density,mask,points,row['width'],row['height'])/mean_count
    support={'point_only':np.ones(len(weak)),'count_only':np.full(len(weak),density.sum()/mean_count),
             'rotated':sample(rotated(xy[weak],row['roi_xyxy'])),'aligned':sample(xy[weak])}
    odds=q[weak].astype(float)/(1-q[weak].astype(float))
    scores={}
    for mode,prior in support.items():
        weighted=odds*np.clip(prior,.25,4.)
        scores[mode]=weighted/(1+weighted)
        assert np.isfinite(scores[mode]).all()
    return xy,base,weak,scores

def summaries(records):
    dev=[r for r in records if r['split']=='development']
    train=[r for r in records if r['split']=='train']
    return {'training':summarize(train),'development':summarize(dev),
            'low':summarize([r for r in dev if 1<=r['truth']<=11]),
            'higher':summarize([r for r in dev if r['truth']>=12]),
            'positive_frames_tp':sum(r['truth']>0 and r['tp']>0 for r in dev),
            'groups':{g:summarize([r for r in dev if r['group']==g]) for g in sorted({r['group'] for r in dev})}}

def main():
    truth,inputs,entries=inventory()
    RUN.mkdir(exist_ok=False)
    scripts=Path(__file__).resolve().parent
    files=[PLAN,Path(__file__),scripts/'analyze_firc_tiny.py',scripts/'audit_firc_countr_misses.py',
           scripts/'firc_countr_support.py',FIRC/'analysis.json',COUNTR/'analysis.json']
    hashes={f.relative_to(ROOT).as_posix():sha(f) for f in files}
    raw_hash=read(COUNTR/'collected/collection_verification.json')['collected_hashes']
    data_hash=read(COUNTR/'collected/data/plan.json')['data_hashes']
    assert sha(ROOT/read(FIRC/'data/plan.json')['local_truth_manifest'])==read(FIRC/'data/plan.json')['truth_manifest_sha256']
    for key,row in inputs.items():
        pred='result/P/predictions/'+key+'.npy'
        assert sha(COUNTR/'collected'/pred)==raw_hash[pred]
        assert sha(DATA/row['array'])==data_hash[row['array']]
    write(RUN/'frozen_run.json',{'created_kst':timestamp(),'identities':hashes,'primary':'aligned','new_model_forwards':0})
    train=[k for k,r in inputs.items() if r['split']=='train']
    dev=[k for k,r in inputs.items() if r['split']=='development']
    assert len(train)==42 and len(dev)==23
    mean_count=float(np.mean([mass(inputs[k]).sum() for k in train]))
    grid=[i/100 for i in range(10,91,5)]+[1.0]
    bank={k:features(inputs[k],entries[k],mean_count) for k in train}
    thresholds={};profiles={}
    for mode in MODES:
        errors=[sum(abs(len(bank[k][1])+int(np.count_nonzero(bank[k][3][mode]>=t))-truth[k]['count']) for k in train) for t in grid]
        thresholds[mode]=min(zip(errors,grid),key=lambda v:(v[0],-v[1]))[1]
        profiles[mode]=dict(zip(grid,errors))
    write(RUN/'frozen_thresholds.json',{'created_kst':timestamp(),'train_ids':train,'mean_training_countr_total':mean_count,
         'thresholds':thresholds,'total_abs_error':profiles,'development_repair_predictions_scored':0})
    names=['original',*MODES,*['equal_number_'+m for m in MODES if m!='aligned']]
    records={m:[] for m in names}
    with (RUN/'points.jsonl').open('x') as out:
        for key in train+dev:
            row=truth[key]
            xy,base,weak,score=bank.pop(key) if key in bank else features(inputs[key],entries[key],mean_count)
            additions={m:weak[score[m]>=thresholds[m]] for m in MODES}
            k=len(additions['aligned'])
            for m in MODES[:-1]:
                order=np.lexsort((weak,-score[m]))
                additions['equal_number_'+m]=weak[order[:k]]
            additions['original']=np.empty(0,dtype=int)
            gt=np.asarray(row['points'],dtype=float).reshape(-1,2)
            for m,ids in additions.items():
                accepted=np.concatenate((base,ids))
                assert len(np.unique(accepted))==len(accepted)
                rec=evaluate(xy[accepted],gt,row);rec['split']=row['split']
                records[m].append(rec)
                out.write(json.dumps({'image_id':key,'split':row['split'],'method':m,'original_ids':base.tolist(),
                          'added_ids':ids.tolist(),'native_points':xy[accepted].tolist()},allow_nan=False)+'\n')
    summary={m:summaries(v) for m,v in records.items()}
    old=read(FIRC/'analysis.json')['metrics']['calibrated']
    assert records['original']==old['per_image'] or {r['image_id']:r for r in records['original']}=={r['image_id']:r for r in old['per_image']}
    b=summary['original'];a=summary['aligned'];dv=a['development']
    def protected(value,control):
        return value<=control+max(.1,.01*control)+1e-12
    gate={'mae_gain':dv['mae']<=.9*b['development']['mae'],'pooled_f1_gain':dv['pooled_f1_at8']>=b['development']['pooled_f1_at8']+.05,
          'low_mae':protected(a['low']['mae'],b['low']['mae']),'low_f1':a['low']['positive_macro_f1_at8']>=b['low']['positive_macro_f1_at8']-.01,
          'empty_fp':dv['empty_false_positives']<=b['development']['empty_false_positives']}
    for m in MODES[:-1]:
        control='equal_number_'+m
        assert [r['count'] for r in records[control]]==[r['count'] for r in records['aligned']]
        gate[control+'_f1_separation']=dv['pooled_f1_at8']>=summary[control]['development']['pooled_f1_at8']+.02-1e-12
    for metric in ['game1','game2']:
        gate[metric]=protected(dv[metric],b['development'][metric])
    for g in b['groups']:
        gate['group_'+g]=protected(a['groups'][g]['mae'],b['groups'][g]['mae'])
    readiness={'mae':dv['mae']<=7.71739130435,'pooled_f1':dv['pooled_f1_at8']>=.5,
               'positive_macro_f1':dv['positive_macro_f1_at8']>=.5,'positive_frames_tp':a['positive_frames_tp']>=14,
               'training_f1':a['training']['positive_macro_f1_at8']>=.8,'low_mae':a['low']['mae']<=2.,
               'low_f1':a['low']['positive_macro_f1_at8']>=.5,'empty_fp':dv['empty_false_positives']==0}
    write(RUN/'result.json',{'completed_kst':timestamp(),'thresholds':thresholds,'summary':summary,'per_image':records,
          'mechanism_components':gate,'mechanism_passed':all(gate.values()),'practical_components':readiness,
          'practical_readiness':all(readiness.values()),'new_model_forwards':0,'optimizer_updates':0,'reserved_scored':0,
          'hashes':{n:sha(RUN/n) for n in ['frozen_run.json','frozen_thresholds.json','points.jsonl']}})
    for method,s in summary.items():
        d=s['development']
        print(method,thresholds.get(method),{k:round(d[k],4) for k in ['mae','pooled_f1_at8','game1','game2','tp','fp','fn','empty_false_positives']})
    print('mechanism',gate,'readiness',readiness)

if __name__=='__main__':main()
