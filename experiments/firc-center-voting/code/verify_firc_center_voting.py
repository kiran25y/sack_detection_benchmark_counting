"""Reconstruct coordinate predictions independently and verify the paired metrics."""
import json
import numpy as np
from audit_firc_public_capacity import maximum_matching
from audit_firc_countr_misses import candidates
from run_firc_center_voting import ROOT,FIRC,OUT,PRIOR,CONTROL,read,sha,write


def reproduce_arrays(row, seed_native, xy, probability, radius):
    """Assign by chunked distance matrices, then reconstruct both centroids directly."""
    if not len(seed_native):
        return {m:seed_native.copy() for m in ('uniform','confidence_vote','opposite')}
    scale=1024/max(row['width'],row['height'])
    seed=seed_native*scale
    points=xy*scale
    owners=np.full(len(points),-1,dtype=int)
    for start in range(0,len(points),512):
        chunk=points[start:start+512]
        distances=((chunk[:,None,:]-seed[None,:,:])**2).sum(axis=2)
        closest=distances.argmin(axis=1)
        selected=distances[np.arange(len(chunk)),closest]<=radius**2
        owners[start+np.flatnonzero(selected)]=closest[selected]
    bounds=(np.asarray(row['roi_xyxy'])-.5)*scale
    uniform=seed_native.copy()
    weighted=seed_native.copy()
    for index,point in enumerate(seed):
        mask=owners==index
        if not mask.any():
            continue
        for destination,weights in [(uniform,None),(weighted,probability[mask].astype(float))]:
            if weights is not None and weights.sum()==0:
                continue
            vector=np.average(points[mask],axis=0,weights=weights)-point
            norm=np.linalg.norm(vector)
            if norm==0:continue
            direction=vector/norm
            length=min(8.,norm)
            for axis in (0,1):
                if direction[axis]!=0:
                    clearance=min(point[axis]-bounds[axis],bounds[axis+2]-point[axis])
                    length=min(length,max(0.,clearance)/abs(direction[axis]))
            destination[index]+=direction*length/scale
    return {'uniform':uniform,'confidence_vote':weighted,'opposite':2*seed_native-weighted}


def main():
    result=read(OUT/'result.json')
    freeze=read(OUT/'freeze.json')
    assert all(sha(ROOT/p)==h for p,h in freeze['source_sha256'].items())
    assert sha(OUT/'predictions.jsonl')==result['predictions_sha256']==read(OUT/'prediction_receipt.json')['predictions_sha256']
    truth={r['image_id']:r for r in read(ROOT/read(FIRC/'data/plan.json')['local_truth_manifest'])}
    entries={r['image_id']:r for r in read(FIRC/'collected/result/report.json')['predictions']}
    old={v['image_id']:v for v in map(json.loads,(PRIOR/'points.jsonl').read_text().splitlines()) if v['method']==CONTROL}
    metrics={m:{r['image_id']:r for r in records} for m,records in result['per_image'].items()}
    total=0
    max_coordinate_error=0.
    changes=[]
    with (OUT/'predictions.jsonl').open() as f:
        for rec in map(json.loads,f):
            k=rec['image_id'];r=truth[k];scale=1024/max(r['width'],r['height'])
            gt=np.asarray(r['points'],float).reshape(-1,2)
            xy,q=candidates(entries[k])
            seed=np.asarray(old[k]['native_points'],float).reshape(-1,2)
            assert rec['seed_ids']==old[k]['original_ids']+old[k]['added_ids']
            assert np.array_equal(seed,xy[np.array(rec['seed_ids'],int)])
            reconstructed=reproduce_arrays(r,seed,xy,q,freeze['training_radius'])
            reconstructed['unchanged']=seed
            for m,native in rec['native_points'].items():
                pred=np.asarray(native,float).reshape(-1,2)
                assert len(pred)==len(seed)
                if len(pred):
                    max_coordinate_error=max(max_coordinate_error,float(np.abs(pred-reconstructed[m]).max()))
                assert np.allclose(pred,reconstructed[m],rtol=0,atol=1e-8)
                values=metrics[m][k]
                dist=np.linalg.norm((gt[:,None,:]-pred[None,:,:])*scale,axis=2)
                for radius,metric in [(4,'tp_at4'),(8,'tp'),(16,'tp_at16')]:
                    adjacent=[np.flatnonzero(row<=radius).tolist() for row in dist]
                    assert int((maximum_matching(adjacent,len(pred))>=0).sum())==values[metric]
                assert (values['count'],values['ae'])==(len(pred),abs(len(pred)-len(gt)))
                for side,metric in [(2,'game1'),(4,'game2')]:
                    lo=np.asarray(r['roi_xyxy'][:2],float);hi=np.asarray(r['roi_xyxy'][2:],float)
                    def histogram(pts):
                        data=np.clip(pts+.5,lo,np.nextafter(hi,-np.inf))
                        return np.histogram2d(data[:,0],data[:,1],bins=[np.linspace(lo[j],hi[j],side+1) for j in (0,1)])[0]
                    assert abs(histogram(gt)-histogram(pred)).sum()==values[metric]
                total+=1
            if r['split']=='development':
                changes.append(metrics['confidence_vote'][k]['tp']-metrics['unchanged'][k]['tp'])
    assert total==260 and len(changes)==23
    outcome={'verified_frame_method_rows':total,'one_to_one_matches_rechecked_at':[4,8,16],
             'max_coordinate_difference_native_pixels':max_coordinate_error,'result_sha256':sha(OUT/'result.json'),
             'frozen_protocol_and_inputs_verified':True,'per_frame_counts_identical':True,
             'development_more_same_fewer_matches':[sum(d>0 for d in changes),sum(d==0 for d in changes),sum(d<0 for d in changes)]}
    write(OUT/'verification.json',outcome)
    print(json.dumps(outcome,indent=2))

if __name__=='__main__':main()
