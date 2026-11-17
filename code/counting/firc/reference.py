"""NumPy reference for the attached FIRC four-state proposal.

Float64 mathematics/geometry oracle, not a differentiable training backend.
The explicit VJP validates the intended local splat derivatives before a
separately verified Torch/CUDA implementation. No image generation or fitting.
"""
from dataclasses import dataclass
import math
import numpy as np

STATES = ("00", "10", "01", "11")
EDGES = ((0, 1), (2, 3), (0, 2), (1, 3))
REGIONS = ("EA", "EB", "U", "BG")


@dataclass(frozen=True)
class Field:
    height: int
    width: int
    padded_height: int
    padded_width: int
    stride: int = 4

    def __post_init__(self):
        values = (self.height, self.width, self.padded_height, self.padded_width)
        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("Positive integer dimensions required")
        if self.stride != 4 or self.padded_height % 112 or self.padded_width % 112:
            raise ValueError("Initial proposal uses stride four and padding multiple 112")
        if self.height > self.padded_height or self.width > self.padded_width:
            raise ValueError("Valid rectangle exceeds padding")

    @property
    def shape(self):
        return self.padded_height // 4, self.padded_width // 4


def _stencil(points, field):
    xy = np.asarray(points, dtype=np.float64)
    if xy.size == 0:
        xy = xy.reshape(0, 2)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Finite Nx2 xy coordinates required")
    # Coordinates are in a rectangle with pixel edges [0,width] x [0,height].
    if (xy < 0).any() or (xy[:, 0] > field.width).any() or (xy[:, 1] > field.height).any():
        raise ValueError("Out-of-field points must be handled upstream, not silently lost")
    grid = xy / 4 - .5
    lower = np.floor(grid).astype(np.int64)
    fraction = grid - lower
    max_x, max_y = math.ceil(field.width/4)-1, math.ceil(field.height/4)-1
    entries = []
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        wx = fraction[:, 0] if dx else 1-fraction[:, 0]
        wy = fraction[:, 1] if dy else 1-fraction[:, 1]
        x = np.clip(lower[:, 0]+dx, 0, max_x)
        y = np.clip(lower[:, 1]+dy, 0, max_y)
        entries.append((y, x, wx*wy, (1 if dx else -1)*wy/4, (1 if dy else -1)*wx/4))
    return xy, entries


def splat(points, masses, field):
    xy, entries = _stencil(points, field)
    m = np.asarray(masses, dtype=np.float64)
    if m.shape != (len(xy),) or not np.isfinite(m).all() or (m < 0).any():
        raise ValueError("Finite nonnegative mass per point required")
    out = np.zeros(field.shape, dtype=np.float64)
    for y, x, w, _, _ in entries:
        np.add.at(out, (y, x), m*w)
    if not np.isclose(out.sum(), m.sum(), rtol=1e-12, atol=1e-12):
        raise ArithmeticError("Splat lost mass")
    return out


def splat_vjp(points, masses, field, cotangent):
    xy, entries = _stencil(points, field)
    m = np.asarray(masses, dtype=np.float64)
    c = np.asarray(cotangent, dtype=np.float64)
    splat(xy, m, field)  # shared validation
    if c.shape != field.shape or not np.isfinite(c).all():
        raise ValueError("Finite grid cotangent required")
    dm = np.zeros(len(xy)); dxy = np.zeros_like(xy)
    for y, x, w, dx, dy in entries:
        v = c[y, x]
        dm += v*w
        dxy[:, 0] += v*m*dx
        dxy[:, 1] += v*m*dy
    return dxy, dm


def sum_pool(a, factor=2):
    a = np.asarray(a, dtype=np.float64)
    if type(factor) is not int or factor <= 0 or a.ndim < 2:
        raise ValueError("Positive pooling factor and spatial array required")
    h, w = a.shape[-2:]
    if h % factor or w % factor or not np.isfinite(a).all():
        raise ValueError("Pooling must cover finite arrays exactly")
    return a.reshape(*a.shape[:-2], h//factor, factor, w//factor, factor).sum(axis=(-3, -1))


def pyramid(a):
    a = np.asarray(a, dtype=np.float64)
    b = sum_pool(a)
    return {4: a, 8: b, 16: sum_pool(b)}


def region_memberships(partition, field):
    """One complete categorical partition, -1 ONLY outside the valid rectangle.

    Pool raw pixel coverages at each scale, never normalized child memberships.
    """
    p = np.asarray(partition)
    if p.shape != (field.padded_height, field.padded_width) or p.dtype.kind not in "iu":
        raise ValueError("Integer full-resolution partition required")
    valid = np.zeros_like(p, dtype=bool); valid[:field.height, :field.width] = True
    if ((p[valid] < 0) | (p[valid] > 3)).any() or (p[~valid] != -1).any():
        raise ValueError("Internal unknowns cannot be silently counted as background")
    out = {}
    for stride in (4, 8, 16):
        coverage = sum_pool(valid, stride)
        r = np.stack([sum_pool(p == i, stride) for i in range(4)])
        out[stride] = np.divide(r, coverage, out=np.zeros_like(r), where=coverage > 0)
        if not np.allclose(out[stride].sum(0), coverage > 0):
            raise ArithmeticError("Incomplete region coverage")
    return out


def spatial_norm(maps, memberships, counts):
    if set(maps) != {4, 8, 16} or set(memberships) != {4, 8, 16}:
        raise ValueError("All three declared scales required")
    if len(counts) != 4 or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("GT region counts EA,EB,U,union required")
    weights = np.array([1., 1., 1., .25]) / np.maximum(1, counts)
    total = 0.
    for stride, scale_weight in ((4, .5), (8, .3), (16, .2)):
        v = np.asarray(maps[stride]); r = np.asarray(memberships[stride])
        if r.shape != (4, *v.shape) or not np.isfinite(v).all() or not np.isfinite(r).all():
            raise ValueError("Finite aligned maps required")
        if (r < 0).any() or (r > 1).any():
            raise ValueError("Membership fractions must be in [0,1]")
        total_membership = r.sum(0)
        if not np.all(np.isclose(total_membership, 1) | (total_membership == 0)):
            raise ValueError("Region memberships must partition each valid cell")
        if np.any((r.sum(0) == 0) & (v != 0)):
            raise ValueError("Residual mass outside verified partition")
        total += scale_weight * float((abs(v)[None]*r*weights[:, None, None]).sum())
    return total


def response_terms(predicted, truth, memberships, counts):
    p, g = np.asarray(predicted, dtype=float), np.asarray(truth, dtype=float)
    if p.shape != g.shape or p.ndim != 3 or len(p) != 4 or not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("Four finite aligned states in order00,10,01,11 required")
    if (p < 0).any() or (g < 0).any():
        raise ValueError("Point measures must be nonnegative")
    residual = p-g
    norm = lambda x: spatial_norm(pyramid(x), memberships, counts)
    edges = [norm(residual[b]-residual[a]) for a, b in EDGES]
    mixed = residual[3]-residual[1]-residual[2]+residual[0]
    first = float(np.mean(edges)); mixed_value = norm(mixed)
    if mixed_value > 2*first + 1e-10:
        raise ArithmeticError("Triangle inequality violated with common fixed regional norm")
    return dict(independent=float(np.mean([norm(e) for e in residual])), first=first,
                mixed=mixed_value, scalar_mixed=float(abs(mixed.sum())), edges=edges,
                gt_mixed_l1=float(abs(g[3]-g[1]-g[2]+g[0]).sum()))


def confidence_initialization(candidates, targets, q=.001, count_weight=.10):
    """Analytic shared-logit-shift derivative, not a model parameter gradient.

    Assumes identical probabilities and one matched positive per GT. This is
    an initialization risk calculation, not a measured optimization outcome.
    """
    if not (0 < q < 1) or not (0 <= targets <= candidates) or candidates <= 0:
        raise ValueError("Invalid candidate/target counts")
    n = max(1, targets)
    positive = .25*(2*q*(1-q)**2*np.log(q) - (1-q)**3)
    negative = .75*(-2*q*q*(1-q)*np.log1p(-q) + q**3)
    focal = (targets*positive+(candidates-targets)*negative)/n
    soft = candidates*q
    count = count_weight*np.clip(soft-targets, -1, 1)*candidates*q*(1-q)
    return dict(candidates=candidates, targets=targets, prior=q, initial_soft_count=soft,
                focal_shared_bias_gradient=float(focal), count_shared_bias_gradient=float(count),
                combined_shared_bias_gradient=float(focal+count),
                interpretation="Positive derivative makes gradient descent reduce the common logit; this is not a full-model trajectory.")
