"""Independently recompute gallery scores, moves, point matches and grid errors."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import hashlib,json
import numpy as np
from pathlib import Path
from audit_firc_public_capacity import maximum_matching
from prepare_firc_visual_swap import OUT,ROOT,FIRC,PRIOR,read,sha,write
from run_firc_visual_swap import PRIMARY,CONTROLS,OLD

def score_again(query,refs):
    if not len(query):return np.empty(0),np.empty(0)
    pos,neg=[],[]
    for _,bank,bank_counts in refs:
        n=bank_counts
        for z,dest in [(bank[:n],pos),(bank[n:],neg)]:
            if not len(z):continue
            # Concatenated normalized vectors implement the average of two cosines.
            wide=query.reshape(len(query),-1)@z.reshape(len(z),-1).T / 2
            single=query[:,0]@z[:,0].T
            dest.append(np.column_stack((wide.max(1),single.max(1))))
    def best_three(v):
        array=np.stack(v);return np.partition(array,-3,axis=0)[-3:].mean(axis=0)
    return (best_three(pos)-best_three(neg)).T

def tp(points,gt,scale,rad):
    distances=np.linalg.norm((gt[:,None]-points[None,:])*scale,axis=2)
    adjacency=[np.flatnonzero(v<=rad).tolist() for v in distances]
    return int((maximum_matching(adjacency,len(points))>=0).sum())

def main():
    r=read(OUT/'result.json');frame_rows=read(OUT/'queries.json')['frames']
    assert sha(OUT/'predictions.jsonl')==r['predictions_sha256']==read(OUT/'predictions_frozen.json')['predictions_sha256']
    assert sha(ROOT/'scripts/run_firc_visual_swap.py')==read(OUT/'readout_freeze.json')['code_sha256']
    source=read(FIRC/'data/plan.json');truth={v['image_id']:v for v in read(ROOT/source['local_truth_manifest'])}
    assert sha(ROOT/source['local_truth_manifest'])==source['truth_manifest_sha256']
    points={v['image_id']:v for v in map(json.loads,(OUT/'predictions.jsonl').read_text().splitlines())}
    prior={v['image_id']:v for v in map(json.loads,(PRIOR/'predictions.jsonl').read_text().splitlines())}
    metrics={m:{v['image_id']:v for v in rows} for m,rows in r['per_image'].items()}
    gallery=[];queries={};report=read(OUT/'features/report.json')
    for row in frame_rows:
        k=row['image_id'];meta=report['features'][k];f=OUT/'features'/meta['file'];assert sha(f)==meta['sha256']
        a=np.load(f,allow_pickle=False).astype(np.float64)
        norms=np.sqrt((a*a).sum(2,keepdims=True));a/=np.maximum(norms,1e-8)
        cut=row['seeds']+len(row['bank']);queries[k]=a[:cut]
        pos=row['positives'];gt=np.asarray(truth[k]['points']).reshape(-1,2)
        assert row['split']==truth[k]['split'] and row['dependency_group']==truth[k]['dependency_group']
        if row['split']=='train':
            native=np.asarray(row['sample_native']).reshape(-1,2)
            assert pos==len(gt) and np.array_equal(native[cut:cut+pos],gt)
            if len(gt) and row['negatives']:
                d=np.linalg.norm((native[cut+pos:,None]-gt)*1024/max(row['width'],row['height']),axis=2)
                assert (d.min(axis=1)>8-1e-9).all()
            gallery.append((row['dependency_group'],a[cut:],pos))
        else:assert cut==len(a) and pos==row['negatives']==0
    assert len(gallery)==42 and len(queries)==65
    rows_checked=0;max_error=0.;moves_audit=[]
    for row in frame_rows:
        k=row['image_id'];item=points[k];n=row['seeds'];nb=len(row['bank']);coords=np.array(row['sample_native']).reshape(-1,2)
        refs=[g for g in gallery if row['split']=='development' or g[0]!=row['dependency_group']]
        full,center=score_again(queries[k],refs)
        for independent,saved in [(full,item['visual_scores']),(center,item['center_only_scores'])]:
            if len(independent):max_error=max(max_error,float(np.max(np.abs(independent-np.asarray(saved)))))
        assert max_error<=2e-5
        cap=row['move_cap'];cost=np.sort(full[:n])[:cap];value=np.sort(full[n:])[::-1][:cap]
        # Where numerical ties could change rank, check the stored swap's margin.
        count=int(np.count_nonzero(value>cost+1e-9))
        assert count==item['number']
        scale=1024/max(row['width'],row['height']);gt=np.array(truth[k]['points']).reshape(-1,2)
        for method,output in item['outputs'].items():
            arr=np.asarray(output['points']).reshape(-1,2);moves=output['moves']
            if method in OLD:
                assert np.array_equal(arr,np.asarray(prior[k]['arms'][method]['native_points']).reshape(-1,2))
            else:
                assert len(moves)==count
                actual=coords[:n].copy();start_matches=tp(actual,gt,scale,8)
                donor=[v['slot'] for v in moves];receiver=[v['bank_index'] for v in moves]
                if method==PRIMARY:
                    assert np.allclose(full[donor],cost[:count],atol=2e-5,rtol=0)
                    assert np.allclose(full[n+np.asarray(receiver,int)],value[:count],atol=2e-5,rtol=0)
                elif method==CONTROLS[1]:
                    prefix=prior[k]['arms']['firc_coverage_only']['moves'][:count]
                    assert donor==[v['slot'] for v in prefix] and [v['candidate'] for v in moves]==[v['receiver_candidate'] for v in prefix]
                else:
                    if method==CONTROLS[0]:s1,s2=np.asarray(row['seed_q']),np.asarray(row['bank_q'])
                    elif method==CONTROLS[2]:s1,s2=center[:n],center[n:]
                    else:
                        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(k.encode()).digest()[:8],'big'))
                        s1,s2=rng.permutation(full[:n]),rng.permutation(full[n:])
                    assert np.allclose(s1[donor],np.sort(s1)[:count],atol=2e-5,rtol=0)
                    assert np.allclose(s2[receiver],np.sort(s2)[::-1][:count],atol=2e-5,rtol=0)
                for v in moves:
                    i,j=v['slot'],v['bank_index'];assert row['bank'][j]==v['candidate']
                    assert np.min(np.linalg.norm((coords[n+j]-coords[:n])*scale,axis=1))>16-1e-9
                    retained=tp(np.delete(actual,i,axis=0),gt,scale,8)
                    actual[i]=coords[n+j];now_matches=tp(actual,gt,scale,8)
                    if method==PRIMARY:
                        moves_audit.append({'image_id':k,'split':row['split'],'group':row['dependency_group'],'lost':start_matches-retained,'gained':now_matches-retained})
                    start_matches=now_matches
                assert np.array_equal(actual,arr)
            val=metrics[method][k];assert len(arr)==n==val['count'] and abs(n-len(gt))==val['ae']
            for rad,key in [(4,'tp_at4'),(8,'tp'),(16,'tp_at16')]:assert tp(arr,gt,scale,rad)==val[key]
            for side,key in [(2,'game1'),(4,'game2')]:
                lo=np.array(row['roi_xyxy'][:2]);hi=np.array(row['roi_xyxy'][2:])
                def hist(x):
                    x=np.clip(x+.5,lo,np.nextafter(hi.astype(float),-np.inf))
                    return np.histogram2d(x[:,0],x[:,1],bins=[np.linspace(lo[j],hi[j],side+1) for j in (0,1)])[0]
                assert abs(hist(gt)-hist(arr)).sum()==val[key]
            rows_checked+=1
    groups={}
    for g in sorted({v['group'] for v in moves_audit if v['split']=='development'}):
        a=[v for v in moves_audit if v['group']==g and v['split']=='development']
        groups[g]={'moves':len(a),'lost':sum(v['lost'] for v in a),'gained':sum(v['gained'] for v in a)}
    assert rows_checked==520
    verification={'records':rows_checked,'independent_4_8_16_and_game_verified':True,'same_count_every_frame':True,
      'gallery_separation_verified':True,'largest_float64_vs_saved_score_error':max_error,'primary_development_swaps':groups,
      'result_sha256':sha(OUT/'result.json')}
    write(OUT/'verification.json',verification);print(json.dumps(verification))

if __name__=='__main__':main()
