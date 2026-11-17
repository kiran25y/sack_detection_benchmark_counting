"""ConvNeXt-Tiny compatible context core and the prescribed IPIC point model.

The context block layout follows Meta's ConvNeXt (MIT), with state-dict names
preserved. No dropout, stochastic depth, NMS or top-k filtering is used.
"""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, width):
        super().__init__(width, eps=1e-6)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, width, expansion=4, scale=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(width, width, 7, padding=3, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.pwconv1 = nn.Linear(width, expansion * width)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * width, width)
        self.gamma = nn.Parameter(torch.full((width,), scale))

    def forward(self, x):
        y = self.norm(self.dwconv(x).permute(0, 2, 3, 1))
        y = self.pwconv2(self.act(self.pwconv1(y))) * self.gamma
        return x + y.permute(0, 3, 1, 2)


class ContextCore(nn.Module):
    widths = (96, 192, 384, 768)

    def __init__(self):
        super().__init__()
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 96, 4, stride=4), LayerNorm2d(96))])
        self.downsample_layers.extend([
            nn.Sequential(LayerNorm2d(a), nn.Conv2d(a, b, 2, stride=2))
            for a, b in zip(self.widths[:-1], self.widths[1:])])
        self.stages = nn.ModuleList([
            nn.Sequential(*[ConvNeXtBlock(w) for _ in range(n)])
            for w, n in zip(self.widths, (3, 3, 9, 3))])

    def forward(self, x):
        levels = []
        for down, stage in zip(self.downsample_layers, self.stages):
            x = stage(down(x)); levels.append(x)
        return levels

    def load_official(self, path, expected_sha256):
        with open(path, 'rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if digest != expected_sha256:
            raise ValueError('Pretrained checkpoint hash mismatch')
        raw = torch.load(path, map_location='cpu', weights_only=True)['model']
        removed = {'norm.weight', 'norm.bias', 'head.weight', 'head.bias'}
        if set(raw) != set(self.state_dict()) | removed:
            raise ValueError('Checkpoint is not the prescribed official Tiny layout')
        self.load_state_dict({k: v for k, v in raw.items() if k not in removed}, strict=True)
        return {'sha256': digest, 'loaded_tensors': len(self.state_dict()),
                'omitted_classification_tensors': sorted(removed)}


class Pyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Sequential(LayerNorm2d(w), nn.Conv2d(w, 192, 1)) for w in ContextCore.widths])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(192, 192, 3, padding=1), nn.GroupNorm(32, 192), nn.GELU())
            for _ in range(4)])

    def forward(self, levels):
        output, previous = [None] * 4, None
        for i in range(3, -1, -1):
            x = self.lateral[i](levels[i])
            if previous is not None:
                x = x + F.interpolate(previous, size=x.shape[-2:], mode='bilinear', align_corners=False)
            previous = x
            output[i] = self.smooth[i](x)
        return output


class DetailPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), LayerNorm2d(32), nn.GELU(),
                                  nn.Conv2d(32, 64, 3, 2, 1), LayerNorm2d(64), nn.GELU())
        self.projections = nn.ModuleList([nn.Conv2d(w, 64, 1) for w in ContextCore.widths])
        self.norm_d = nn.ModuleList([LayerNorm2d(64) for _ in range(4)])
        self.norm_u = nn.ModuleList([LayerNorm2d(64) for _ in range(4)])
        self.gates = nn.ModuleList([nn.Conv2d(128, 64, 1) for _ in range(4)])
        self.stages = nn.ModuleList([
            nn.Sequential(ConvNeXtBlock(64, 2, .001), ConvNeXtBlock(64, 2, .001)) for _ in range(4)])
        self.out = nn.Conv2d(64, 192, 1)
        self.alpha = nn.Parameter(torch.zeros(()))
        for gate in self.gates:
            nn.init.constant_(gate.bias, -2.)

    def forward(self, image, context):
        d = self.stem(image)
        for i, c in enumerate(context):
            u = F.interpolate(self.projections[i](c), size=d.shape[-2:], mode='bilinear', align_corners=False)
            gate = torch.sigmoid(self.gates[i](torch.cat((self.norm_d[i](d), self.norm_u[i](u)), 1)))
            d = self.stages[i](d + gate * u)
        return self.alpha * self.out(d)


class IPICNet(nn.Module):
    def __init__(self, detail=False):
        super().__init__()
        # Construct all shared modules before the optional path: seed pairing is exact.
        self.context = ContextCore()
        self.pyramid = Pyramid()
        self.all_products_query = nn.Parameter(torch.zeros(192))
        self.film = nn.Sequential(nn.Linear(192, 384), nn.GELU(), nn.Linear(384, 384))
        self.condition_norm = nn.GroupNorm(32, 192)
        self.condition = nn.Conv2d(193, 192, 1)
        self.point_blocks = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(192, 192, 3, padding=1, groups=192),
                          nn.Conv2d(192, 192, 1), nn.GroupNorm(32, 192), nn.GELU()) for _ in range(2)])
        self.point_out = nn.Conv2d(192, 3, 1)
        nn.init.constant_(self.point_out.bias[0:1], math.log(.001 / .999))
        self.detail = DetailPath() if detail else None

    def encode_references(self, crops, boxes):
        """Crops E×3×128×128; boxes E×4 are valid object rectangles in crop pixels."""
        if not 1 <= len(crops) <= 3 or crops.shape[-2:] != (128, 128):
            raise ValueError('Require one to three 128-pixel positive reference crops')
        if boxes.shape != (len(crops), 4) or not torch.isfinite(boxes).all():
            raise ValueError('Invalid reference geometry')
        if (boxes[:, 2:] <= boxes[:, :2]).any() or (boxes < 0).any() or (boxes > 128).any():
            raise ValueError('Empty or out-of-crop reference box')
        p8 = self.pyramid(self.context(crops))[1]
        rois = torch.cat((torch.arange(len(crops), device=boxes.device)[:, None], boxes), 1)
        return roi_align(p8, rois, (3, 3), spatial_scale=1 / 8, sampling_ratio=2, aligned=True).mean((-2, -1))

    def forward(self, image, valid_mask, prototypes=None, all_products=False):
        """One query image per microbatch; reuse the same prototype graph across a pair."""
        if image.ndim != 4 or image.shape[0] != 1 or any(v % 32 for v in image.shape[-2:]):
            raise ValueError('Query must be one image padded to multiples of32')
        if valid_mask.shape != image.shape[-2:] or valid_mask.dtype != torch.bool:
            raise ValueError('Explicit valid content mask required')
        c = self.context(image); f4 = self.pyramid(c)[0]
        if self.detail is not None:
            f4 = f4 + self.detail(image, c)
        if all_products:
            if prototypes is not None:
                raise ValueError('All-products mode does not take visual prototypes')
            q = self.all_products_query
            cosine = torch.zeros_like(f4[:, :1])
        else:
            if prototypes is None or not 1 <= len(prototypes) <= 3:
                raise ValueError('Visual mode needs positive reference prototypes')
            q = prototypes.mean(0)
            cosine = torch.einsum('bchw,ec->behw', F.normalize(f4, dim=1),
                                  F.normalize(prototypes, dim=1)).amax(1, keepdim=True)
        gamma, beta = self.film(q).chunk(2)
        y = (1 + .1 * torch.tanh(gamma))[None, :, None, None] * self.condition_norm(f4)
        y = y + .1 * torch.tanh(beta)[None, :, None, None]
        raw = self.point_out(self.point_blocks(self.condition(torch.cat((y, cosine), 1))))
        raw = raw[0].permute(1, 2, 0).reshape(-1, 3).float()
        h, w = f4.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h, device=raw.device), torch.arange(w, device=raw.device), indexing='ij')
        anchors = torch.stack((4 * xx + 2, 4 * yy + 2), -1).reshape(-1, 2).float()
        valid = valid_mask[anchors[:, 1].long(), anchors[:, 0].long()]
        if not valid_mask.any():
            raise ValueError('Empty valid content')
        ys, xs = torch.where(valid_mask)
        points = anchors + 16 * raw[:, 1:].tanh()
        # Pixel-center convention: raster samples live at integer pixel centers.
        points = torch.stack((points[:, 0].clamp(xs.min(), xs.max()),
                              points[:, 1].clamp(ys.min(), ys.max())), 1)
        return {'logits': raw[:, 0], 'points': points, 'anchors': anchors,
                'valid': valid, 'probabilities': raw[:, 0].sigmoid()}

    @staticmethod
    def predict(output, threshold=.5):
        keep = output['valid'] & (output['probabilities'] >= threshold)
        return {'visible_count': int(keep.sum()), 'points': output['points'][keep],
                'confidence': output['probabilities'][keep]}
