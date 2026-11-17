"""Check fixed counts, score-independent movement, coverage gains and one-to-one metrics."""
import json
from pathlib import Path
import numpy as np
from audit_firc_public_capacity import maximum_matching
from run_firc_uncovered_reallocation import ROOT, FIRC, DATA, CENTER, OUT, PRIMARY, BASELINES, MATCHED, candidates, mass, read, sha, write


def independent_bank(sites,scores,candidates_ids,seeds):
    order=np.lexsort((candidates_ids,-scores))
    allowed=np.ones(len(sites),bool)
    for seed in seeds:
        allowed &= np.linalg.norm(sites-seed,axis=1)>16
    indices=[]
    for i in order:
        if allowed[i]:
            indices.append(i)
            allowed &= np.linalg.norm(sites-sites[i],axis=1)>16
    return np.array(indices,dtype=int)


def local_values(xy,density,mask,row):
    # Direct sums on the original density grid, independently of integral images.
    h,w=density.shape
    dx=w/row['width'];dy=h/row['height'];r=8*max(row['height'],row['width'])/1024
    left=np.clip(np.ceil((xy[:,0]+.5-r)*dx-.5).astype(int),0,w)
    right=np.clip(np.floor((xy[:,0]+.5+r)*dx-.5).astype(int)+1,0,w)
    top=np.clip(np.ceil((xy[:,1]+.5-r)*dy-.5).astype(int),0,h)
    bottom=np.clip(np.floor((xy[:,1]+.5+r)*dy-.5).astype(int)+1,0,h)
    valid=int(mask.sum())
    return np.array([density[t:b,l:u].sum()*valid/max(1,mask[t:b,l:u].sum())
                     for l,u,t,b in zip(left,right,top,bottom)])


def objective(sites,centers,weights):
    peak=np.zeros(len(sites))
    for center in centers:
        d=np.linalg.norm(sites-center,axis=1)
        peak=np.maximum(peak,1-(d/8)**2)
    return float(np.sum(peak*weights))


def verify_moves(data,center,xy,scale,sites,site_ids,bank,weights,greedy):
    pos=center.copy()
    remaining=list(range(len(center)))
    bank_ids=site_ids[bank].tolist()
    remaining_ids=bank_ids.copy()
    if weights.sum():weights=weights/weights.sum()
    else:assert not data['moves']
    gains={j:objective(sites,[xy[j]*scale],weights) for j in remaining_ids} if greedy else {}
    def losses():
        base=objective(sites,pos*scale,weights)
        return {j:base-objective(sites,np.delete(pos,j,axis=0)*scale,weights) for j in remaining}
    for change in data['moves']:
        j=change['slot'];i=change['receiver_candidate']
        assert j in remaining and i in remaining_ids
        assert np.min(np.linalg.norm((center-xy[i])*scale,axis=1))>16-1e-9
        if greedy:
            ll=losses();gg=gains[i];before=objective(sites,pos*scale,weights)
            assert ll[j]<=min(ll.values())+1e-9 and gg>=max(gains[k] for k in remaining_ids)-1e-9
            assert abs(ll[j]-change['coverage_loss'])<1e-8 and abs(gg-change['coverage_gain'])<1e-8
        pos[j]=xy[i]
        if greedy:
            after=objective(sites,pos*scale,weights)
            assert after>before and abs(after-before-(gg-ll[j]))<1e-8
        remaining.remove(j);remaining_ids.remove(i)
    if greedy and remaining and remaining_ids:
        assert max(gains[i] for i in remaining_ids)<=min(losses().values())+1e-9
    assert np.array_equal(pos,np.asarray(data['native_points'],float).reshape(-1,2))


def main():
    frozen=read(OUT/'freeze.json');result=read(OUT/'result.json')
    assert all(sha(ROOT/k)==h for k,h in frozen['sources'].items())
    assert sha(OUT/'predictions.jsonl')==result['predictions_sha256']==read(OUT/'predictions_frozen.json')['sha256']
    entries={e['image_id']:e for e in read(FIRC/'collected/result/report.json')['predictions']}
    start={r['image_id']:r for r in map(json.loads,(CENTER/'predictions.jsonl').read_text().splitlines())}
    rows={r['image_id']:r for r in read(DATA/'manifest.json')}
    truth={r['image_id']:r for r in read(ROOT/read(FIRC/'data/plan.json')['local_truth_manifest'])}
    metrics={m:{r['image_id']:r for r in records} for m,records in result['per_image'].items()}
    n=0;swaps=0;seen=set()
    with (OUT/'predictions.jsonl').open() as f:
        for record in map(json.loads,f):
            k=record['image_id'];row=rows[k];assert k not in seen;seen.add(k)
            scale=1024/max(row['width'],row['height'])
            xy,q=candidates(entries[k]);ids=np.flatnonzero(q>=.025);sites=xy[ids]*scale
            center=np.array(start[k]['native_points']['confidence_vote'],float).reshape(-1,2)
            bank=independent_bank(sites,q[ids],ids,center*scale) if len(center) else np.empty(0,int)
            assert len(bank)==record['destination_bank_size']
            with np.load(DATA/row['array']) as arr:mask=arr['mask'].astype(bool)
            joint=np.sqrt(q[ids].astype(float)*local_values(xy[ids],mass(row),mask,row))
            arms=record['arms'];main_moves=arms[PRIMARY]['moves'];num=len(main_moves)
            donors=[v['slot'] for v in main_moves];receivers=[v['receiver_candidate'] for v in main_moves]
            rank_donors=np.lexsort((np.arange(len(center)),q[np.asarray(start[k]['seed_ids'],int)]))[:num].tolist()
            rank_receivers=ids[bank[:num]].tolist()
            for m,ds,rs in [(MATCHED[0],donors,rank_receivers),(MATCHED[1],rank_donors,receivers),(MATCHED[2],rank_donors,rank_receivers)]:
                assert [v['slot'] for v in arms[m]['moves']]==ds and [v['receiver_candidate'] for v in arms[m]['moves']]==rs
            for m,arm in arms.items():
                pos=np.asarray(arm['native_points'],float).reshape(-1,2)
                assert len(pos)==len(center)
                if m not in BASELINES:
                    greedy=m in (PRIMARY,'firc_coverage_only')
                    verify_moves(arm,center,xy,scale,sites,ids,bank,joint if m==PRIMARY else q[ids].astype(float),greedy)
                    swaps+=len(arm['moves'])
                else:
                    ref=start[k]['native_points']['unchanged' if m==BASELINES[0] else 'confidence_vote']
                    assert np.array_equal(pos,np.asarray(ref,float).reshape(-1,2))
                gt=np.array(truth[k]['points'],float).reshape(-1,2)
                distances=np.linalg.norm((gt[:,None]-pos[None,:])*scale,axis=2)
                values=metrics[m][k]
                assert len(pos)==values['count'] and abs(len(pos)-len(gt))==values['ae']
                for radius,key in [(4,'tp_at4'),(8,'tp'),(16,'tp_at16')]:
                    edges=[np.flatnonzero(d<=radius).tolist() for d in distances]
                    assert int((maximum_matching(edges,len(pos))>=0).sum())==values[key]
                for side,key in [(2,'game1'),(4,'game2')]:
                    lower=np.array(row['roi_xyxy'][:2],float);upper=np.array(row['roi_xyxy'][2:],float)
                    def hist(x):
                        a=np.clip(x+.5,lower,np.nextafter(upper,-np.inf))
                        return np.histogram2d(a[:,0],a[:,1],bins=[np.linspace(lower[j],upper[j],side+1) for j in (0,1)])[0]
                    assert abs(hist(gt)-hist(pos)).sum()==values[key]
                n+=1
    assert len(seen)==65 and n==455
    write(OUT/'verification.json',{'verified_frames':len(seen),'verified_method_frame_records':n,
        'verified_swaps_all_arms_all_splits':swaps,'all_frame_counts_identical':True,
        'independent_direct_density_sums_and_greedy_objective_verified':True,
        'all_4_8_16px_matches_and_GAME1_2_verified':True,'result_sha256':sha(OUT/'result.json')})
    print('Verified',n,'method-frame records and',swaps,'exchanges across controls and splits.')

if __name__=='__main__':main()
