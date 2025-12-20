import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import einx
from jaxtyping import Float, Int


def _trunc_norm_init_parameters(
    *dimensions,
    mean: float,
    std: float,
    a: float,
    b: float,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.parameter.Parameter:
    p = torch.empty(*dimensions, device=device, dtype=dtype)
    nn.init.trunc_normal_(p, mean=mean, std=std, a=a, b=b)
    return nn.Parameter(p)


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


class SwigluModule(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device: torch.device = None, dtype: torch.dtype = None
    ):
        super().__init__()

        # $$ d_{ff} = \frac{8}{3}d_{model} $$
        #
        # and d_ff needs to be a multiply of 64 to make efficient use of hardware
        #
        # d_ff = 8 * d_model // 3
        # d_ff = int(math.log(d_ff, 64)) ** 64
        # d_ff = max(d_ff, d_model)

        # $$ W_1, W_3 \in R^{d_{ff} \cross d_{model}} $$
        # $$ W_2 \in R^{d_{model} \cross d_{ff}} $$

        self.w1 = _trunc_norm_init_parameters(
            d_ff, d_model, mean=0.0, std=1, a=-3, b=3, device=device, dtype=dtype
        )
        self.w2 = _trunc_norm_init_parameters(
            d_model, d_ff, mean=0.0, std=1, a=-3, b=3, device=device, dtype=dtype
        )
        self.w3 = _trunc_norm_init_parameters(
            d_ff, d_model, mean=0.0, std=1, a=-3, b=3, device=device, dtype=dtype
        )
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # $$ SwiGLU(x, W_1, W_2, W_3) = W_2(SiLu(W_1x) * W_3x) $$

        # $$ t_1 = W_{1}x $$
        t1 = einx.dot("... d_model, d_ff d_model -> ... d_ff", x, self.w1)

        # $$ t_2 = SiLu(W_{1}x) = SiLu(t_{1}) = t_1 * sigmoid(t_1) $$
        t2 = einx.multiply("... d_ff, ... d_ff -> ... d_ff", t1, torch.sigmoid(t1))

        # $$ t_3 = W_3x $$
        t3 = einx.dot("... d_model, d_ff d_model -> ... d_ff", x, self.w3)

        # $$ t_4 = SiLu(W_1x)*W_3x = t_2 * t_3 $$
        t4 = einx.multiply("... d_ff, ... d_ff -> ... d_ff", t2, t3)

        # $$ result = W_2(SiLu(W_1x) * W_3x) = W_2t_4 $$
        return einx.dot("d_model d_ff, ... d_ff -> ... d_model", self.w2, t4)
