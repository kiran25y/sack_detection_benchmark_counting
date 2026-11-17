"""Sack adaptation mechanics; no launcher and no implicit admission or recovery.

This prepares the approved dynamic/no-soft-count recipe for matched arms while
leaving historical losses and frozen releases unchanged. Real-data assembly,
reviewed pairs and an architecture-specific CUDA guard are still prerequisites.
"""
from pathlib import Path
import hashlib

import torch

from .data_manifest import assert_stage_ready
from .losses import ARM_RELATIONS, pair_loss
from .model import IPICNet
from .training import make_optimizer


FOUNDATION_SHA = '3388776d1862789692363cc433f951cd79d07fb614d5f89cb156803ed249a162'
SCREEN_ARMS = {'A': 'B0', 'B': 'B0', 'C': 'B1', 'D': 'B1', 'E': 'B0', 'F': 'B1'}
OBJECTIVE = 'dynamic_focal_plus_5point_no_absolute_soft_count'


def adaptation_pair_loss(a, b, truth_a, truth_b, arm, masks=None,
                         eligible=None, region_valid=None):
    """Preserve the original relation controls; remove only absolute count loss.

    Point-only pairs may supply global delta, never local mask supervision.
    Compute the base directly rather than subtracting two large count terms.
    This function is preparation, not evidence that an arm's CUDA guard passed.
    """
    if arm not in SCREEN_ARMS:
        raise ValueError('Only the prescribed A-F screen is implemented here')
    if masks is None and (eligible is not None or region_valid is not None):
        raise ValueError('Region labels without reviewed ownership masks')
    result = pair_loss(a, b, truth_a, truth_b, arm, masks, eligible, region_valid)
    for name in ('view_a', 'view_b'):
        view = result[name]
        view['total'] = view['focal'] + 5 * view['point']
        view['absolute_count_weight'] = 0.
    result['total'] = .5 * (result['view_a']['total'] + result['view_b']['total'])
    relation = ARM_RELATIONS[arm]
    if relation != 'none':
        result['total'] = result['total'] + .1 * result['global_delta']
    if relation == 'local':
        result['total'] = result['total'] + .25 * result['local_delta']
    return result


def require_runtime(arm, runtime):
    if arm not in SCREEN_ARMS:
        raise ValueError('Unknown screen arm')
    if not runtime.get('strict_deterministic_algorithms') or runtime.get('warn_only'):
        raise ValueError('Strict validated numerical runtime required')
    if SCREEN_ARMS[arm] == 'B1' and runtime.get('supports_B0_only'):
        raise ValueError('B1 requires its own validated resampling runtime')


def require_screen_admission(readiness, arm, guard, identities):
    """Bind new evidence to this data/recipe; old foundation guard cannot admit it.

    The caller must independently hash the evidence files and pass their actual
    identities. This is an extra contract, not a substitute for that verification.
    """
    assert_stage_ready(readiness, 'screen')
    if guard.get('stage') != 'sack_adaptation_cuda' or not guard.get('passed'):
        raise ValueError('Actual sack adaptation CUDA guard required')
    if guard.get('objective') != OBJECTIVE or guard.get('assignment') != 'dynamic':
        raise ValueError('CUDA guard used another objective or assignment')
    if guard.get('arms') != list(SCREEN_ARMS):
        raise ValueError('Full matched A-F runtime coverage is required')
    if not identities or any(not h or len(h) != 64 for h in identities.values()):
        raise ValueError('Missing independently verified identities')
    required = {'foundation', 'training_manifest', 'development_manifest', 'draws',
                'pair_audit', 'asset_background_ledger', 'source_release'}
    if set(identities) != required or identities['foundation'] != FOUNDATION_SHA:
        raise ValueError('Incomplete or wrong experiment identity')
    if guard.get('identities') != identities:
        raise ValueError('Data, draw or code identity changed after CUDA guard')
    for check in ('full_two_view_backward', 'finite_model_optimizer',
                  'authentic_recovery', 'same_shared_initialization',
                  'high_count_and_wide_shape', 'regional_loss_backward'):
        if guard.get(check) is not True:
            raise ValueError('Missing adaptation check: ' + check)
    for candidate in SCREEN_ARMS:
        require_runtime(candidate, guard.get('runtime_by_arm', {}).get(candidate, {}))
    require_runtime(arm, guard['runtime_by_arm'][arm])
    return True


def initialize_from_foundation(path, arm, device='cpu'):
    """Copy the verified seed1729 foundation, then create a fresh Adam optimizer.

    This does not perform any optimization. Call only after the stage's admission.
    Diagnostic checkpoints and checkpoint optimizer moments cannot enter here.
    """
    if arm not in SCREEN_ARMS:
        raise ValueError('Unknown screen arm')
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != FOUNDATION_SHA:
        raise ValueError('Not the verified public foundation checkpoint')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('optimizer_steps') != 2000 or payload.get('seed') != 1729:
        raise ValueError('Foundation metadata mismatch')
    if not payload.get('plan_sha256') or not isinstance(payload.get('model'), dict):
        raise ValueError('Incomplete foundation metadata')
    state = payload['model']
    if not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite or invalid foundation state')
    if any(k.startswith('detail.') for k in state):
        raise ValueError('Expected the B0 shared foundation')
    torch.manual_seed(1729)
    model = IPICNet(detail=SCREEN_ARMS[arm] == 'B1')
    actual = model.state_dict()
    common = {k for k in actual if not k.startswith('detail.')}
    if set(state) != common:
        raise ValueError('Shared model keys differ from the foundation')
    model.load_state_dict(state, strict=SCREEN_ARMS[arm] == 'B0')
    if model.detail is not None and model.detail.alpha.detach().count_nonzero():
        raise ValueError('Detail residual must start at zero')
    model.to(device).train()
    optimizer = make_optimizer(model)
    if optimizer.state:
        raise ValueError('Adaptation must reset optimizer moments')
    return model, optimizer


def validate_prepared_pair(batch, record):
    """Reject incomplete point targets and mask/identity mismatches before update.

    Only image, content validity and shared allowed references enter the model.
    Point/ownership labels remain loss inputs. This verifies a prepared tensor
    contract; it does not manufacture an exhaustive annotation or review receipt.
    """
    if record.get('split') != 'train' or record.get('exhaustive_reviewed') is not True:
        raise ValueError('Only exhaustive audited training pairs may be updated')
    if record.get('label_eligibility_hold') or record.get('annotation_type') not in (
            'complete_points', 'complete_masks'):
        raise ValueError('Unresolved or visibility-held labels cannot train')
    if not record.get('query_id') or not record.get('dependency_group'):
        raise ValueError('Missing query identity or grouping')
    if type(batch.get('all_products')) is not bool:
        raise ValueError('Explicit query mode required')
    if batch['all_products']:
        if record.get('dataset') != 'D2S' or batch.get('references') is not None or batch.get('reference_boxes') is not None:
            raise ValueError('All-products mode is D2S-only without visual references')
    else:
        refs, boxes = batch.get('references'), batch.get('reference_boxes')
        if not torch.is_tensor(refs) or refs.ndim != 4 or not 1 <= refs.shape[0] <= 3 or refs.shape[1:] != (3, 128, 128):
            raise ValueError('One to three shared native reference crops required')
        if not torch.is_tensor(boxes) or boxes.shape != (len(refs), 4) or not torch.isfinite(refs).all() or not torch.isfinite(boxes).all():
            raise ValueError('Invalid shared reference pixels/geometry')
        if (boxes < 0).any() or (boxes > 128).any() or (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError('Reference box outside crop')
    for side in ('a', 'b'):
        image, valid, pts = batch[side], batch['valid_' + side], batch['points_' + side]
        if image.ndim != 4 or image.shape[:2] != (1, 3) or any(n % 32 for n in image.shape[-2:]) or not torch.isfinite(image).all():
            raise ValueError('Invalid padded query image')
        if valid.dtype != torch.bool or valid.shape != image.shape[-2:] or not valid.any():
            raise ValueError('Invalid content validity mask')
        h, w = valid.shape
        if pts.ndim != 2 or pts.shape[1] != 2 or not torch.isfinite(pts).all():
            raise ValueError('Invalid point targets')
        # Continuous pixel-center charts permit a half-pixel image boundary.
        if len(pts) and ((pts < -.5).any() or (pts[:, 0] > w - .5).any() or (pts[:, 1] > h - .5).any()):
            raise ValueError('Point outside the transformed image chart')
        if record.get('count_' + side) != len(pts):
            raise ValueError('Prepared count/point mismatch')
        if not len(pts) and record.get('verified_target_absent_' + side) is not True:
            raise ValueError('Unverified zero target')
    regional = [batch.get(k) is not None for k in ('ownership', 'eligible', 'region_valid')]
    if any(regional) and not all(regional):
        raise ValueError('Incomplete ownership/eligibility/validity ledger')
    if all(regional):
        if record['annotation_type'] != 'complete_masks' or record.get('paired_mask_audit_passed') is not True:
            raise ValueError('Point-only or unreviewed masks cannot supply local supervision')
        if len(batch['eligible']) != 2 or len(batch['ownership']) != 2:
            raise ValueError('Exactly two aligned ownership views required')
        eligible_a, eligible_b = batch['eligible']
        regions = batch['region_valid']
        if eligible_a.ndim != 1 or not len(eligible_a) or eligible_b.shape != eligible_a.shape or regions.shape != eligible_a.shape or regions.dtype != torch.bool:
            raise ValueError('Unaligned instance union')
        ids = record.get('region_ids', [])
        if len(ids) != len(regions) or len(set(ids)) != len(ids) or ids[0] != '__background__':
            raise ValueError('Missing ordered identity union')
        for index, side in enumerate(('a', 'b')):
            own, eligible = batch['ownership'][index], batch['eligible'][index]
            if own.dtype != torch.long or own.shape != batch['valid_' + side].shape or (own < -1).any() or (own >= len(regions)).any():
                raise ValueError('Invalid ownership raster')
            if eligible[0] != 0 or not ((eligible == 0) | (eligible == 1)).all() or eligible.sum() != len(batch['points_' + side]):
                raise ValueError('Eligibility does not match exhaustive target points')
            visible_ids = set(own[batch['valid_' + side]].unique().tolist())
            if any(i not in visible_ids for i in torch.where(eligible == 1)[0].tolist()):
                raise ValueError('Eligible target has no visible ownership region')
            if (own[~batch['valid_' + side]] != -1).any():
                raise ValueError('Padding cannot be supervised as background')
    return True
