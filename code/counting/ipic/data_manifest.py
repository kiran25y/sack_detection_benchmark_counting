"""Strict admission for the new task; historical total labels are not point labels."""
import hashlib
import json
from pathlib import Path
import numpy as np

REQUIRED=('image_id','source_version','encoded_sha256','decoded_sha256','parent_id',
          'scene_id','capture_session','camera','site','target','annotation_type',
          'completeness','split','dependency_group','width','height','points',
          'count','point_convention','exhaustive_reviewed','query_mode')


def validate_rows(rows, training=False):
    ids=set();groups={};parents={};hashes={}
    for r in rows:
        missing=set(REQUIRED)-set(r)
        if missing:raise ValueError(f'Missing manifest fields:{sorted(missing)}')
        if r['image_id'] in ids:raise ValueError('Duplicate image query ID')
        ids.add(r['image_id'])
        if r['annotation_type'] not in ('complete_points','complete_masks') or not r['exhaustive_reviewed']:
            raise ValueError('Exact point training requires reviewed exhaustive points/masks')
        if r['completeness'] not in ('complete_image','complete_roi'):raise ValueError('Incomplete coverage')
        if r['completeness']=='complete_roi' and not r.get('roi_rule'):raise ValueError('Missing reviewed ROI rule')
        if r['count']==0 and not r.get('verified_target_absent'):raise ValueError('Unverified zero')
        if r['query_mode']=='all_products' and training and r.get('dataset')!='D2S':
            raise ValueError('All-products training is D2S-only in the prescribed recipe')
        if r['query_mode']=='visual_references' and not 1<=len(r.get('reference_ids',[]))<=3:
            raise ValueError('Visual query needs one to three allowed references')
        pts=np.asarray(r['points'],float).reshape(-1,2)
        if len(pts)!=r['count'] or not np.isfinite(pts).all():raise ValueError('Point/count mismatch')
        if len(pts) and ((pts<0).any() or (pts[:,0]>=r['width']).any() or (pts[:,1]>=r['height']).any()):
            raise ValueError('Out-of-image native point')
        if training and r['split']!='train':raise ValueError('Protected split in training')
        for registry,key in ((groups,r['dependency_group']),(parents,r['parent_id']),(hashes,r['decoded_sha256'])):
            if not key:raise ValueError('Unresolved grouping/ancestry')
            if key in registry and registry[key]!=r['split']:raise ValueError('Cross-split ancestry or duplicate')
            registry[key]=r['split']
    return {'queries':len(rows),'dependency_groups':len(groups),'complete':True}


def assert_stage_ready(receipt, stage):
    required=['official_pretrained_verified','coordinate_tests','finite_loss_gradients',
              'full_assignment_capacity','two_view_resource_profile','tiny_real_set_fit',
              'integer_point_fit_verified']
    if stage=='foundation':required+=['d2s_native_train_verified','fsc_native_train_verified']
    else:required+=['shared_foundation_verified','audited_sack_train_points','audited_sack_development',
                    'paired_assets_and_backgrounds_reviewed','training_pair_manual_audit_200',
                    'draw_ledger_frozen']
    if stage=='confirmatory':required+=['screen_passed','independent_test_protocol_frozen']
    missing=[k for k in required if receipt.get(k) is not True]
    if missing:raise ValueError('Stage admission missing: '+', '.join(missing))
    return True


def read_manifest(path):
    raw=Path(path).read_bytes();rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    return rows,hashlib.sha256(raw).hexdigest()
