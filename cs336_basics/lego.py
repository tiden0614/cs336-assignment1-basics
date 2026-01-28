import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import einx
from jaxtyping import Float, Int, Bool
from collections.abc import Callable, Iterable
import math


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
        self,
        d_model: int,
        d_ff: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None
    ):
        super().__init__()

        r"""
        Given
        $$
        R_k^i = \begin{bmatrix}\cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\
                                 \sin(\theta_{i,k}) & \cos(\theta_{i,k})\end{bmatrix}
        $$

        
        $$
        R^i = \begin{bmatrix}
        R_1^i & 0 & 0 & \dots & 0 \\
        0 & R_2^i & 0 & \dots & 0 \\
        0 & 0 & R_3^i & \dots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & \dots & R_{d/2}^i
        \end{bmatrix}
        $$
        
        $$ i \in [1,max\_seq\_len] $$

        
        We will store $$C, S \in \real^{max\_seq\_len \cross k}, k = \frac{d}{2}$$
        and C, S store precomputed cos and sin values
        for given position
        """

        self.k = d_k // 2
        i_dim = torch.arange(0, max_seq_len, 1, dtype=torch.float)
        k_dim = torch.arange(1, self.k + 1, 1, dtype=torch.float)

        r"""

        $$
        I_{\text{grid}} = \begin{bmatrix} 
        1 & 1 & \cdots & 1 \\ 
        2 & 2 & \cdots & 2 \\ 
        \vdots & \vdots & \ddots & \vdots \\ 
        i & i & \cdots & i \end{bmatrix}
        $$

        """

        r"""

        $$
        K_{\text{grid}} = \begin{bmatrix} 
        1 & 2 & \cdots & k \\ 
        1 & 2 & \cdots & k \\ 
        \vdots & \vdots & \ddots & \vdots \\ 
        1 & 2 & \cdots & k \end{bmatrix}
        $$

        """
        i_grid, k_grid = torch.meshgrid(i_dim, k_dim, indexing="ij")

        # $$\theta_{i,k} = \frac{i}{\Theta^{(2k - 2)/d}}$$

        denominator = theta ** ((2 * k_grid - 2) / d_k)
        theta_grid = einx.divide("i k, i k -> i k", i_grid, denominator)
        C = torch.cos(theta_grid)
        S = torch.sin(theta_grid)

        self.register_buffer("cos_k", C, persistent=False)
        self.register_buffer("sin_k", S, persistent=False)

    def _rotate_interleaved(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        d = x.shape[-1]
        chunked = x.view(*x.shape[:-1], -1, 2)
        swapped = chunked[..., [1, 0]]
        flattened = swapped.flatten(start_dim=-2, end_dim=-1)
        return flattened * torch.tensor([-1, 1] * (d // 2))

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        r"""
        $$
        x =
        \begin{bmatrix} 
        x_{0,0} & x_{0,1} & x_{0,2} & x_{0,3} & \cdots & x_{0,d-2} & x_{0,d-1} \\
        x_{1,0} & x_{1,1} & x_{1,2} & x_{1,3} & \cdots & x_{1,d-2} & x_{1,d-1} \\
        \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
        x_{seq\_len-1,0} & x_{seq\_len-1,1} & x_{seq\_len-1,2} & x_{seq\_len-1,3} & \cdots & x_{seq\_len-1,d-2} & x_{seq\_len-1,d-1}
        \end{bmatrix}
        $$

        $$
        x\_pair\_rotated =
        \begin{bmatrix} 
        -x_{0,1} & x_{0,0} & -x_{0,3} & x_{0,2} & \cdots & -x_{0,d-1} & x_{0,d-2} \\
        -x_{1,1} & x_{1,0} & -x_{1,3} & x_{1,2} & \cdots & -x_{1,d-1} & x_{1,d-2} \\
        \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
        -x_{seq\_len-1,1} & x_{seq\_len-1,0} & -x_{seq\_len-1,3} & x_{seq\_len-1,2} & \cdots & -x_{seq\_len-1,d-1} & x_{seq\_len-1,d-2}
        \end{bmatrix}
        $$
        """
        x_pair_rotated = self._rotate_interleaved(x)
        r"""
        $$
        cos\_k = 
        \begin{bmatrix} 

        cos(\theta_{0,1}) & cos(\theta_{0,2}) & \cdots & cos(\theta_{0,k}) \\ 
        cos(\theta_{1,1}) & cos(\theta_{1,1}) & \cdots & cos(\theta_{1,k}) \\ 
        \vdots & \vdots & \ddots & \vdots \\ 
        cos(\theta_{seq\_len-1,1}) & cos(\theta_{seq\_len-1,1}) & \cdots & cos(\theta_{seq\_len-1,k})  

        \end{bmatrix}
        $$

        $$
        cos\_d = 
        \begin{bmatrix} 

        cos(\theta_{0,1}) &cos(\theta_{0,1}) & cos(\theta_{0,2}) & cos(\theta_{0,2}) & \cdots & cos(\theta_{0,k}) & cos(\theta_{0,k}) \\ 
        cos(\theta_{1,1}) &cos(\theta_{1,1}) & cos(\theta_{1,2}) & cos(\theta_{1,2}) & \cdots & cos(\theta_{1,k}) & cos(\theta_{1,k}) \\ 
        \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 
        cos(\theta_{seq\_len-1,1}) &cos(\theta_{seq\_len-1,1}) & cos(\theta_{seq\_len-1,2}) & cos(\theta_{seq\_len-1,2}) & \cdots & cos(\theta_{seq\_len-1,k}) & cos(\theta_{seq\_len-1,k}) \\ 

        \end{bmatrix}
        $$
        """
        cos_d = torch.repeat_interleave(self.cos_k[token_positions], repeats=2, dim=-1)
        sin_d = torch.repeat_interleave(self.sin_k[token_positions], repeats=2, dim=-1)
        return (x * cos_d) + (x_pair_rotated * sin_d)

    def forward_slow_sparse_matrix_mul(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        r"""
        $$
        cos\_k = 
        \begin{bmatrix} 

        cos(\theta_{0,1}) & cos(\theta_{0,2}) & \cdots & cos(\theta_{0,k}) \\ 
        cos(\theta_{1,1}) & cos(\theta_{1,1}) & \cdots & cos(\theta_{1,k}) \\ 
        \vdots & \vdots & \ddots & \vdots \\ 
        cos(\theta_{seq\_len-1,1}) & cos(\theta_{seq\_len-1,1}) & \cdots & cos(\theta_{seq\_len-1,k})  

        \end{bmatrix}
        $$
        $$
        diag\_embed(cos\_k[pos]) = 
        \begin{bmatrix} 

        cos(\theta_{pos,1}) & 0 & \cdots & 0 \\ 
        0 & cos(\theta_{pos,1}) & \cdots & 0 \\ 
        \vdots & \vdots & \ddots & \vdots \\ 
        0 & 0 & \cdots & cos(\theta_{pos,k})  

        \end{bmatrix}
        $$
        """
        cos_diag: Float[torch.Tensor, "... seq_len k"] = torch.diag_embed(
            self.cos_k[token_positions]
        )

        r"""
        $$
        template = 
        \begin{bmatrix} 
        1 & 0 \\ 
        0 & 1 \\ 
        \end{bmatrix}
        $$
        """
        template = torch.eye(2)

        r"""
        
        $$
        cos\_R = 
        \begin{bmatrix} 

        cos(\theta_{0,1}) & cos(\theta_{0,2}) & \cdots & cos(\theta_{0,k}) \\ 
        cos(\theta_{1,1}) & cos(\theta_{1,1}) & \cdots & cos(\theta_{1,k}) \\ 
        \vdots & \vdots & \ddots & \vdots \\ 
        cos(\theta_{seq\_len,1}) & cos(\theta_{seq\_len,1}) & \cdots & cos(\theta_{seq\_len,k})  

        \end{bmatrix}
        $$
        $$
        cos\_R = 
        \begin{bmatrix} 

        cos(\theta_{pos,1})\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ \end{bmatrix} & 0 & \cdots & 0 \\ 
        0 & cos(\theta_{pos,2})\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ \end{bmatrix} & \cdots & 0 \\ 
        \vdots & \vdots & \ddots & \vdots \\ 
        0 & 0 & \cdots & cos(\theta_{pos,k})\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ \end{bmatrix}  

        \end{bmatrix}

        =
        
        \begin{bmatrix} 

        cos(\theta_{pos,1}) & 0 & 0 & 0 & \cdots & 0 & 0 \\ 
        0 & cos(\theta_{pos,1}) & 0 & 0& \cdots & 0 & 0\\ 
        0 & 0 & cos(\theta_{pos,2}) & 0 & \cdots & 0 & 0\\ 
        0 & 0 & 0 & cos(\theta_{pos,2})  & \cdots & 0 & 0\\ 
        \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 
        0 & 0 & 0 & 0& \cdots & cos(\theta_{pos,k}) & 0 \\
        0 & 0 & 0 & 0& \cdots & 0 & cos(\theta_{pos,k})  

        \end{bmatrix}
        $$
        """
        cos_R: Float[torch.Tensor, "... seq_len d_k"] = torch.kron(cos_diag, template)

        sin_diag: Float[torch.Tensor, "... seq_len k"] = torch.diag_embed(
            self.sin_k[token_positions]
        )
        template = torch.Tensor([[0.0, -1.0], [1.0, 0.0]])
        r"""
        $$
        sin\_R = 
        \begin{bmatrix} 

        sin(\theta_{pos,1})\begin{bmatrix} 0 & -1 \\ 1 & 0 \\ \end{bmatrix} & 0 & \cdots & 0 \\ 
        0 & sin(\theta_{pos,2})\begin{bmatrix} 0 & -1 \\ 1 & 0 \\ \end{bmatrix} & \cdots & 0 \\ 
        \vdots & \vdots & \ddots & \vdots \\ 
        0 & 0 & \cdots & sin(\theta_{pos,k})\begin{bmatrix} 0 & -1 \\ 1 & 0 \\ \end{bmatrix}  

        \end{bmatrix}

        =
        
        \begin{bmatrix} 

        0 & -sin(\theta_{pos,1}) & 0 & 0 & \cdots & 0 & 0 \\ 
        sin(\theta_{pos,1}) & 0 & 0 & 0& \cdots & 0 & 0\\ 
        0 & 0 & 0 & -sin(\theta_{pos,2}) & \cdots & 0 & 0\\ 
        0 & 0 & sin(\theta_{pos,2}) & 0 & \cdots & 0 & 0\\ 
        \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 
        0 & 0 & 0 & 0& \cdots & 0 & -sin(\theta_{pos,k})  \\
        0 & 0 & 0 & 0& \cdots & sin(\theta_{pos,k}) & 0 

        \end{bmatrix}
        $$
        """
        sin_R: Float[torch.Tensor, "... seq_len d_k"] = torch.kron(sin_diag, template)

        r"""
        $$
        R = cos\_R + sin\_R

        =
        
        \begin{bmatrix} 

        cos(\theta_{pos,1}) & -sin(\theta_{pos,1}) & 0 & 0 & \cdots & 0 & 0 \\ 
        sin(\theta_{pos,1}) & cos(\theta_{pos,1})  & 0 & 0& \cdots & 0 & 0\\ 
        0 & 0 & cos(\theta_{pos,2}) & -sin(\theta_{pos,2}) & \cdots & 0 & 0\\ 
        0 & 0 & sin(\theta_{pos,2}) & cos(\theta_{pos,2}) & \cdots & 0 & 0\\ 
        \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 
        0 & 0 & 0 & 0& \cdots & cos(\theta_{pos,k}) & -sin(\theta_{pos,k})  \\
        0 & 0 & 0 & 0& \cdots & sin(\theta_{pos,k}) &  cos(\theta_{pos,k})

        \end{bmatrix}
        $$
        """
        R = cos_R + sin_R
        return einx.dot("seq_len d_k_1 d_k, ... seq_len d_k -> ... seq_len d_k_1", R, x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True)
    x = x - max_val.values
    exp = torch.exp(x)
    denom = torch.sum(exp, dim=dim, keepdim=True)
    return exp / denom


def attend(
    q: Float[torch.Tensor, "... queries d_k"],
    k: Float[torch.Tensor, "... keys d_k"],
    v: Float[torch.Tensor, "... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] = None,
) -> Float[torch.Tensor, "... d_v"]:
    d_k = k.shape[-1]
    relevance = einx.dot(
        "... queries d_k, ... keys d_k -> ... queries keys",
        q,
        k,
    )
    pre_softmax = relevance / (d_k**0.5)

    if mask is not None:
        pre_softmax = pre_softmax + torch.where(mask, 0.0, float("-inf"))

    s = softmax(pre_softmax, -1)
    return einx.dot("... queries keys, ... keys d_v -> ... queries d_v", s, v)


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        theta: float = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"invalid params d_model={d_model} num_heads={num_heads}"
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.W_o = _trunc_norm_init_parameters(
            d_model, d_model, mean=0.0, std=1.0, a=-3, b=3, device=device, dtype=dtype
        )
        self.W_qkv = _trunc_norm_init_parameters(
            3 * d_model,
            d_model,
            mean=0.0,
            std=1.0,
            a=-3,
            b=3,
            device=device,
            dtype=dtype,
        )

        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device
            )
        else:
            self.rope = None

    def load_weights(
        self,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
    ):
        qkv_stacked = torch.stack([q_proj_weight, k_proj_weight, v_proj_weight])
        qkv = einx.rearrange("three d_k d_in -> (three d_k) d_in", qkv_stacked)
        self.load_state_dict({"W_qkv": qkv, "W_o": o_proj_weight})

    def forward(
        self,
        X: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        seq_len = X.shape[-2]

        # Compute Q, K, V in a single matrix multiplication
        QKV = einx.dot(
            "heads_combined_d_model d_model, ... seq_len d_model -> ... seq_len heads_combined_d_model",
            self.W_qkv,
            X,
        )

        # Causal masking
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        # Rearrange the Q, K, V and send them to the attend function
        # Q, K, V = einx.rearrange(
        #     "... [3 out_heads out_d_h] -> 3 ... out_heads out_d_h",
        #     QKV,
        #     out_heads=self.num_heads,
        #     out_d_h = self.d_k,
        # )
        all_head_features = QKV.shape[:-1]
        qkv_reshaped = QKV.view(*all_head_features, 3, self.num_heads, self.d_k)
        # Move num_heads in front of seq_len
        qkv_reshaped = qkv_reshaped.movedim(-2, -4)
        # Now split the 3 matrices by moving the 3 into the first position
        Q, K, V = qkv_reshaped.movedim(-2, 0)

        # Apply ROPE
        if self.rope is not None and token_positions is not None:
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)

        attention = attend(Q, K, V, mask)
        attention = einx.rearrange(
            "... head seq_len d_h -> ... seq_len head d_h", attention
        )
        attention = einx.rearrange(
            "... seq_len head d_h -> ... seq_len (head d_h)", attention
        )
        result = einx.dot(
            "d_model d_model_1, ... seq_len d_model_1 -> ... seq_len d_model",
            self.W_o,
            attention,
            head=self.num_heads,
        )
        return result


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.rms_attention = RMSNormModule(d_model, device=device, dtype=dtype)
        self.attention = MultiHeadSelfAttentionModule(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.rms_ffn = RMSNormModule(d_model, device=device, dtype=dtype)
        self.ffn_swiglu = SwigluModule(d_model, d_ff, device=device, dtype=dtype)

    def load_weights(
        self,
        attn_q_proj_weight: Float[Tensor, " d_k d_model"],
        attn_k_proj_weight: Float[Tensor, " d_k d_model"],
        attn_v_proj_weight: Float[Tensor, " d_v d_model"],
        attn_o_proj_weight: Float[Tensor, " d_model d_v"],
        rms_attention_weight: Float[Tensor, "d_model"],
        ffn_w1_weight: Float[Tensor, "d_ff d_model"],
        ffn_w2_weight: Float[Tensor, "d_model d_ff"],
        ffn_w3_weight: Float[Tensor, "d_ff d_model"],
        rms_ffn_weight: Float[Tensor, "d_model"],
    ):
        # RMSNorm before attention
        self.rms_attention.load_state_dict({"g": rms_attention_weight})

        # Attention
        self.attention.load_weights(
            attn_q_proj_weight,
            attn_k_proj_weight,
            attn_v_proj_weight,
            attn_o_proj_weight,
        )

        # RMSNorm before FFN
        self.rms_ffn.load_state_dict({"g": rms_ffn_weight})

        # FFN
        self.ffn_swiglu.load_state_dict(
            {"w1": ffn_w1_weight, "w2": ffn_w2_weight, "w3": ffn_w3_weight}
        )

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        rms_attention = self.rms_attention.forward(x)

        seq_len = x.shape[1]
        assert (
            seq_len <= self.max_seq_len
        ), f"cannot infer seq_len={seq_len} > max_seq_len={self.max_seq_len}"
        positions = torch.arange(start=0, end=seq_len)
        attention_result = self.attention.forward(rms_attention, positions)

        ffn_input = x + attention_result
        rms_ffn = self.rms_ffn.forward(ffn_input)
        ffn_result = self.ffn_swiglu.forward(rms_ffn)

        return ffn_input + ffn_result


def _loss_function(
    o: Float[Tensor, "... batch_size vocab_size"],
    x_pos: Float[Tensor, "... batch_size"],
) -> Float[Tensor, "... batch_size"]:
    r"""
    $$
    p(x_{i+1}|x_{1:i}) = softmax(o_i)[x_{i+1}] = \frac{exp(o_i[x_{i+1}])}{\sum_{a=1}^{vocab\_size}exp(o_i[a])}
    $$


    $$
    log\sum_{a=1}^{vocab\_size}exp(o_i[a] - M)
    = log\sum_{a=1}^{vocab\_size}\frac{exp(o_i[a])}{exp(M)}
    = log\frac{\sum_{a=1}^{vocab\_size}{exp(o_i[a])}}{exp(M)}
    = log\sum_{a=1}^{vocab\_size}{exp(o_i[a])} - M
    $$

    $$
    -log(softmax(o_i)[x_{i+1}]) = -log \frac{exp(o_i[x_{i+1}])}{\sum_{a=1}^{vocab\_size}exp(o_i[a])}
    = -o_i[x_{i+1}] + log\sum_{a=1}^{vocab\_size}exp(o_i[a])
    = -o_i[x_{i+1}] + log\sum_{a=1}^{vocab\_size}exp(o_i[a] - M) + M
    $$
    """
    o_max = torch.amax(o, dim=-1, keepdim=True)
    o_exp_sum = torch.sum(torch.exp(o - o_max), dim=-1, keepdim=True)
    o_selected = o.gather(dim=-1, index=x_pos.unsqueeze(-1))
    return -o_selected + torch.log(o_exp_sum) + o_max


def cross_entropy(
    o: Float[Tensor, "... batch_size vocab_size"],
    x_pos: Float[Tensor, "... batch_size"],
) -> float:
    losses = _loss_function(o, x_pos)
    return torch.sum(losses) / torch.numel(losses)


def perplexity(
    o: Float[Tensor, "... batch_size vocab_size"],
    x_pos: Float[Tensor, "... batch_size"],
) -> float:
    losses = _loss_function(o, x_pos)
    return torch.exp(torch.sum(losses) / torch.numel(losses))


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        assert lr >= 0, f"Invalid learning rate {lr}"
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)  # iteration
                grad = p.grad.data  # gradient of loss with respect to p
                p.data -= lr / math.sqrt(t + 1) * grad  # apply gradient
                state["t"] = t + 1

        return loss


def toy_training_loop():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)

    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        alpha: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        lambda_: float,
    ):
        defaults = {
            "alpha": alpha,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "lambda": lambda_,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None if closure is None else closure()

        def get_or_init_state(state, name, init_func):
            if name not in state:
                state[name] = init_func()
            return state[name]

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                t = state.get("t", 1)
                state["t"] = t + 1
                g = param.grad.data

                m = get_or_init_state(state, "m", lambda: torch.zeros(*g.shape))
                m = m * group["beta1"] + (1 - group["beta1"]) * g
                state["m"] = m

                v = get_or_init_state(state, "v", lambda: torch.zeros(*g.shape))
                v = v * group["beta2"] + (1 - group["beta2"]) * g * g
                state["v"] = v

                alpha_t = (
                    group["alpha"]
                    * math.sqrt(1 - group["beta2"] ** t)
                    / (1 - group["beta1"] ** t)
                )
                param.data = param.data - alpha_t * m / (
                    torch.sqrt(v) + group["epsilon"]
                )
                param.data = (1 - group["alpha"] * group["lambda"]) * param.data

        return loss


def cosine_annealing_learning_rate_schedule(
    t: int, a_min: float, a_max: float, T_w: int, T_c: int
) -> float:
    if t < T_w:
        return a_max * t / T_w

    if t > T_c:
        return a_min

    return (
        a_min + (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (a_max - a_min) / 2
    )


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2: float, eps: float = 1e-6):
    # 1. Compute total L2 norm across all tensors
    total_norm = torch.sqrt(sum(torch.sum(param.grad ** 2) for param in params if param.grad is not None))
    
    # 2. Determine the scaling factor
    clip_coeff = max_l2 / max(total_norm, eps)
    
    # 3. Apply the same scale to everything if total_norm > max_l2
    if clip_coeff < 1.0:
        for param in params:
            if param.grad is not None:
                param.grad.mul_(clip_coeff)

