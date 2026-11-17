"""Fail-closed contracts for the owner-authorized 2x2 sack transfer pilot."""
import collections,copy,hashlib,json,math
import numpy as np
from PIL import Image,ImageEnhance

CELLS={'P00':{'coverage':'original','appearance':'current'},
       'P10':{'coverage':'broadened','appearance':'current'},
       'P01':{'coverage':'original','appearance':'conservative'},
       'P11':{'coverage':'broadened','appearance':'conservative'}}
TRAIN_GROUPS={'ipic_dependency_b778d9b86516','ipic_dependency_9945d48d4761','ipic_dependency_a8232acfdcec'}
DEV_GROUPS={'ipic_dependency_483475312362','ipic_dependency_b8c3dda6893f'}
ORIGINAL_GROUP='ipic_dependency_b778d9b86516'
UPDATES=200;WARMUP=20;PAIRS=1600;SEED=1729

def conservative_appearance(image,seed):
    """Independent appearance RNG; geometry and all public inputs are unaffected."""
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),20261104,1]));names=[]
    for name,p in [('brightness',.25),('contrast',.25),('saturation',.2),('gamma',.2)]:
        if rng.random()<p:names.append(name)
    rng.shuffle(names);log=[]
    for name in names[:2]:
        v=float(rng.uniform(.8,1.2))
        if name=='gamma':image=Image.fromarray(np.uint8(255*(np.asarray(image)/255.)**v))
        else:image={'brightness':ImageEnhance.Brightness,'contrast':ImageEnhance.Contrast,'saturation':ImageEnhance.Color}[name](image).enhance(v)
        log.append(dict(name=name,value=v))
    return image,dict(seed=int(seed),seed_stream=[int(seed),20261104,1],operations=log,visibility_review_required=True,policy='conservative_sack_v1')

def validate_manifest(rows,allocation):
    if allocation.get('state')!='owner_approved_revised_split':raise ValueError('Recording reassignment not approved')
    roles={r['dependency_group']:r['split'] for r in allocation['proposed_groups']}
    if {g for g,s in roles.items() if s=='train'}!=TRAIN_GROUPS or {g for g,s in roles.items() if s=='development'}!=DEV_GROUPS:raise ValueError('Approved recording groups changed')
    if sum(s=='reserved' for s in roles.values())!=5:raise ValueError('Five untouched groups required')
    ids=set();images=set();counts=collections.Counter();groups=collections.defaultdict(set)
    for r in rows:
        if r['image_id'] in ids or r['encoded_sha256'] in images:raise ValueError('Duplicate image/query across manifest')
        ids.add(r['image_id']);images.add(r['encoded_sha256'])
        if r['split'] not in ('train','development') or roles.get(r['dependency_group'])!=r['split']:raise ValueError('Group leakage or reserved input')
        if r.get('exhaustive_reviewed') is not True or r.get('points_are_partial',False) or not isinstance(r['count'],int) or r['count']!=len(r['points']):raise ValueError('Unresolved/inconsistent point label: '+r['image_id'])
        if not r.get('annotation_author') or r.get('mask_supervision',False) or r.get('cutout_eligible',False):raise ValueError('Missing label provenance or unapproved mask use')
        x0,y0,x1,y1=r['roi_xyxy'];w,h=r['width'],r['height']
        if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('Invalid common foreground ROI')
        seen=[]
        for p in r['points']:
            if len(p)!=2 or not all(math.isfinite(v) for v in p) or not(x0<=p[0]<x1 and y0<=p[1]<y1):raise ValueError('Point outside shared foreground')
            if any(math.dist(p,q)<1e-6 for q in seen):raise ValueError('Duplicate point')
            seen.append(p)
        band='zero' if r['count']==0 else 'low_1_11' if r['count']<=11 else 'higher_12_plus'
        counts[(r['split'],band)]+=1;groups[r['split']].add(r['dependency_group'])
    if groups['train']!=TRAIN_GROUPS or groups['development']!=DEV_GROUPS:raise ValueError('Missing prescribed recording group')
    for split in ('train','development'):
        for band in ('zero','low_1_11','higher_12_plus'):
            if not counts[(split,band)]:raise ValueError('Unevaluable protected stratum: '+split+'/'+band)
    return dict(queries=len(rows),groups={k:sorted(v) for k,v in groups.items()},strata={s:{b:counts[(s,b)] for b in ('zero','low_1_11','higher_12_plus')} for s in ('train','development')})

def build_draws(original,train):
    """Equal sources/batches. Public draws replay the old audited input exactly."""
    bins={'sacks':[],'FSC147':[],'native':[],'composite':[]}
    for d in original:
        k=d['dataset'] if d['dataset']!='D2S' else 'composite' if d['composite'] else 'native'
        bins[k].append(d)
    needed={'sacks':800,'FSC147':400,'native':200,'composite':200}
    if any(len(bins[k])<v for k,v in needed.items()):raise ValueError('Insufficient original audited draws')
    oldrows=[r for r in train if r['dependency_group']==ORIGINAL_GROUP];byid={r['image_id']:i for i,r in enumerate(train)}
    if len(oldrows)!=20:raise ValueError('Original20 labels required')
    grouped={g:[i for i,r in enumerate(train) if r['dependency_group']==g] for g in sorted(TRAIN_GROUPS)}
    if not all(grouped.values()):raise ValueError('Missing broadened group')
    rng=np.random.default_rng(20261104);broad=[];order=list(grouped)
    for i in range(800):
        group=order[i%3];idx=int(rng.choice(grouped[group]));broad.append(idx)
    output={c:[] for c in CELLS}
    for step in range(UPDATES):
        block=bins['sacks'][step*4:step*4+4]+bins['FSC147'][step*2:step*2+2]+[bins['native'][step],bins['composite'][step]]
        for j,source in enumerate(block):
            for c,spec in CELLS.items():
                d=copy.deepcopy(source);d['source_audit_draw_id']=source['draw_id'];d['draw_id']=step*8+j
                if source['dataset']=='sacks':
                    idx=byid[source['query_id']] if spec['coverage']=='original' else broad[step*4+j]
                    d.update(row_index=idx,query_id=train[idx]['image_id'])
                output[c].append(d)
    for c,draws in output.items():
        assert len(draws)==PAIRS and collections.Counter(d['dataset'] for d in draws)=={'sacks':800,'FSC147':400,'D2S':400}
    return output

def gate(candidate,controls):
    """Signed deltas retained; missing strata cannot pass as zero errors."""
    expected={'P00','P10','P01'}
    if set(controls)!=expected:raise ValueError('All three fixed controls required')
    results={}
    for name,control in controls.items():
        checks={};deltas={}
        for stratum in ('all','zero','low_1_11','higher_12_plus'):
            a=candidate.get(stratum);b=control.get(stratum)
            if not a or not b or not a.get('queries') or a['queries']!=b.get('queries'):raise ValueError('Missing/mismatched protected stratum')
            for k in ('MAE','RMSE','GAME1','GAME2','F1_8'):
                if not all(math.isfinite(x[k]) for x in (a,b)):raise ValueError('Nonfinite metric')
                deltas[stratum+'/'+k]=a[k]-b[k]
        a,b=candidate['all'],control['all']
        checks['MAE_at_least_10_percent']=a['MAE']<=.9*b['MAE'] and a['MAE']<b['MAE']
        checks['F1_8_at_least_0_05']=a['F1_8']>=b['F1_8']+.05
        for k in ('RMSE','GAME1','GAME2'):checks[k+'_protected']=a[k]<=b[k]+max(.1,.01*b[k])
        for k in ('MAE','GAME1','GAME2'):checks['low_'+k+'_protected']=candidate['low_1_11'][k]<=control['low_1_11'][k]+max(.1,.01*control['low_1_11'][k])
        checks['low_F1_protected']=candidate['low_1_11']['F1_8']>=control['low_1_11']['F1_8']-1e-9
        checks['empty_FP_no_increase']=candidate['zero']['FP_total']<=control['zero']['FP_total']
        results[name]=dict(passed=all(checks.values()),checks=checks,signed_candidate_minus_control=deltas)
    return dict(passed=all(r['passed'] for r in results.values()),comparisons=results,primary_cell='P11',threshold=.5,no_alternative_winner=True)
