"""Point detection from complementary patch-token views of the same image."""
import torch
from torch import nn
from torch.nn import functional as F


class ContextPointDetector(nn.Module):
    """Share the point head across views, so all four logits are on one learned scale."""

    def __init__(self, point_head, mode):
        super().__init__()
        self.point_head = point_head
        self.mode = mode
        if mode == 'paired':
            # The two reveal scales and the two disagreement penalties are positive.
            self.mix = nn.Parameter(torch.log(torch.expm1(torch.tensor([.5, .5, .25, .25]))))
        elif mode == 'fusion':
            self.mix = nn.Parameter(torch.tensor([.5, .5, 0., 0.]))
        elif mode != 'ordinary':
            raise ValueError(mode)
        if mode != 'ordinary':
            self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, inputs):
        feature, selected = inputs
        if self.mode == 'ordinary':
            raw = self.point_head(feature[:1].float())[0]
            return raw[0], raw[1:].sigmoid().permute(1, 2, 0)
        raw = self.point_head(feature.float())
        full, even_visible, odd_visible, hidden = raw.unbind()
        local_in = torch.where(selected, even_visible, odd_visible)
        local_out = torch.where(selected, odd_visible, even_visible)
        offsets_a = full[1:].sigmoid()
        offsets_b = local_in[1:].sigmoid()
        if self.mode == 'paired':
            reveal_full = full[0] - local_out[0]
            reveal_masked = local_in[0] - hidden[0]
            a, b, disagreement, displacement = F.softplus(self.mix).unbind()
            logit = (self.bias + a*reveal_full + b*reveal_masked
                     - disagreement*(reveal_full-reveal_masked).abs()
                     - displacement*(offsets_a-offsets_b).abs().mean(dim=0))
            weight = (reveal_full-reveal_masked).sigmoid()
        else:
            a, b, c, d = self.mix.unbind()
            logit = self.bias + a*full[0] + b*local_in[0] + c*local_out[0] + d*hidden[0]
            weight = (full[0]-local_in[0]).sigmoid()
        offsets = weight*offsets_a + (1-weight)*offsets_b
        return logit, offsets.permute(1, 2, 0)


@torch.no_grad()
def context_features(encoder, image, valid, null_cache):
    """Keep spatial tokens and position embeddings aligned in the four inputs."""
    h, w = valid.shape
    iy, ix = torch.meshgrid(torch.arange(h,device=image.device),torch.arange(w,device=image.device),indexing='ij')
    even = ((iy//8 + ix//8) % 2 == 0)
    key = (h, w, int(valid.any(dim=1).sum()), int(valid.any(dim=0).sum()))

    def encode(mask):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            tokens = encoder.forward_features(image, masks=None if mask is None else mask.reshape(1,-1))['x_norm_patchtokens']
        return tokens.transpose(1,2).reshape(1,768,h,w).half()

    full = encode(None)
    even_visible = encode(valid & ~even)
    odd_visible = encode(valid & even)
    # All valid tokens are replaced. Every fully padded patch has the same fixed input.
    new_null = key not in null_cache
    if new_null:
        null_cache[key] = encode(valid)
    features = torch.cat([full, even_visible, odd_visible, null_cache[key]],dim=0)
    assert torch.isfinite(features).all()
    return (features, even), 3+int(new_null)
