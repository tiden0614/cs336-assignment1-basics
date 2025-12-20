import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import einx
from jaxtyping import Float, Int


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

    def forward(self, x: Tensor) -> Tensor:
        output = einx.dot("... in, out in -> ... out", x, self.W)
        return output


class EmbeddingModule(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        embeddings = torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        )
        nn.init.trunc_normal_(embeddings, mean=0.0, std=1, a=-3, b=3)
        self.embeddings = nn.parameter.Parameter(embeddings)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Tensor:
        return self.embeddings[token_ids]


class RMSNormModule(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        g = torch.empty(self.d_model, device=device, dtype=dtype)
        nn.init.trunc_normal_(g, mean=0.0, std=1, a=-3, b=3)
        self.g = nn.parameter.Parameter(g)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)

        # $$ rms = \sqrt{\frac{1}{d_{model}}\sum_{i=1}^{d_{model}}a_i^2 + \epsilon} $$

        rms = (
            einx.dot("... d_model, ... d_model -> ... 1", x_f32, x_f32) / self.d_model
            + self.eps
        ) ** 0.5

        # $$ RMSNorm(a_i) = \frac{a_i}{RMS(a)}g_i $$

        result = einx.divide("... d_model, ... 1 -> ... d_model", x_f32, rms)
        result = einx.multiply("... d_model, d_model -> ... d_model", result, self.g)
        return result.to(in_dtype)
