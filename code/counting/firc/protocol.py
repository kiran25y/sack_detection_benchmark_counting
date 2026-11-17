"""Fail-closed quartet manifest admission. Booleans require external evidence.

This checker checks consistency and references; it cannot certify physical
independence or annotations from image pixels. It never promotes old frames.
"""
import re
import numpy as np
from .reference import STATES, splat


def validate_quartet(row, field, role, allowed_groups, ancestry_splits, tolerance=1e-8):
    if role not in ("train", "development", "test"):
        raise ValueError("Explicit non-reserved role required")
    if row.get("split") != role or row.get("group_id") not in allowed_groups:
        raise ValueError("Quartet split/group is not admitted")
    if row.get("kind") not in ("physical", "synthetic"):
        raise ValueError("Physical versus synthetic provenance required")
    for key in ("exhaustive_reviewed", "common_coordinates_verified", "partition_reviewed", "visibility_ledger_reviewed"):
        if row.get(key) is not True:
            raise ValueError("Missing prerequisite: " + key)
    if not row.get("review_receipt") or not row.get("source_ancestry"):
        raise ValueError("Review evidence and source parents required")
    if row.get("kind") == "physical":
        error = row.get("max_control_point_error_pixels")
        if error is None or not np.isfinite(error) or not 0 <= error <= 1:
            raise ValueError("Physical registration must pass one-input-pixel control points")
    parents = row["source_ancestry"]
    if any(ancestry_splits.get(p) != role for p in parents):
        raise ValueError("Unknown/cross-split source ancestry")
    states = row.get("states", {})
    if set(states) != set(STATES):
        raise ValueError("Exactly all four factual states required; a temporal triple is insufficient")
    references, transforms, maps, seen_images = [], [], [], {}
    for state in STATES:
        r = states[state]
        if not re.fullmatch("[0-9a-f]{64}", r.get("image_sha256", "")):
            raise ValueError("Missing state image hash")
        references.append(r.get("reference_ids")); transforms.append(r.get("transform_id"))
        if not r.get("reference_ids") or not r.get("transform_id") or not r.get("image_id"):
            raise ValueError("Explicit images, references and geometry required")
        points = np.asarray(r.get("points"), dtype=float)
        if points.size == 0: points = points.reshape(0, 2)
        ids = r.get("instance_ids", [])
        if len(ids) != len(points) or len(set(ids)) != len(ids) or r.get("count") != len(points):
            raise ValueError("Complete count/point/within-state instance-ID agreement required")
        measure = splat(points, np.ones(len(points)), field)
        if r["image_sha256"] in seen_images and not np.array_equal(measure, seen_images[r["image_sha256"]]):
            raise ValueError("Identical image bytes cannot acquire different target measures")
        seen_images[r["image_sha256"]] = measure
        maps.append(measure)
    if any(r != references[0] for r in references) or any(t != transforms[0] for t in transforms):
        raise ValueError("All states must share references and geometry")
    if any(ancestry_splits.get("reference:"+r) != "train" for r in references[0]):
        raise ValueError("Sack reference eligibility requires audited training parents")
    g = np.stack(maps)
    mixed_l1 = float(abs(g[3]-g[1]-g[2]+g[0]).sum())
    independent = row.get("independent_edits_reviewed") is True and row.get("affected_supports_disjoint_reviewed") is True and mixed_l1 <= tolerance
    requested = row.get("relation_role")
    if requested not in ("independent", "interaction_diagnostic"):
        raise ValueError("Explicit independent or interaction-diagnostic role required")
    if requested == "independent" and not independent:
        raise ValueError("Zero count difference/disjoint boxes do not establish spatial independence")
    return dict(eligible_independent=independent, gt_mixed_l1=mixed_l1,
                relation_role=requested, physical_capture_certified_by_code=False,
                unique_state_images=len({states[s]["image_sha256"] for s in STATES}))
