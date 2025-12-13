import torch
import torch.nn as nn
import torch.nn.functional as F
import einx


class LinearModule(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        variance = 2 / (in_features + out_features)
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(
            W, mean=0.0, std=pow(variance, 0.5), a=-3 * variance, b=3 * variance
        )
        self.W = nn.parameter.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = einx.dot("... in, out in -> ... out", x, self.W)
        return output
