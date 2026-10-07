# position_encoding Rotary copy.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

class PositionEmbeddingRoPE1D(nn.Module):
    """
    RoPE for 1D sequences: input (B, L, D) -> output (B, L, D)
    Uses a precomputed rotation table of shape (max_len, D/2).
    """
    def __init__(self, hidden_size: int, max_len: int, base: int = 10000):
        super().__init__()
        assert hidden_size % 2 == 0, "RoPE requires even hidden_size"
        # 预生成最长长度的旋转表
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.base = base
        # 使用你已有的 RoPENd 来生成旋转表，维度是 (max_len, hidden_size)
        self.rope = RoPENd((max_len, hidden_size), base=base)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: (B, L, D) float
        mask: (B, L) 可忽略，仅为接口兼容
        """
        B, L, D = x.shape
        assert D == self.hidden_size and D % 2 == 0

        # 旋转表（只取前 L）
        rot = self.rope.rotations[:L].to(x.device)

        # —— 关键：不用 view_as_complex ——
        x_pair = x.reshape(B, L, D // 2, 2)
        x_real = x_pair[..., 0]
        x_imag = x_pair[..., 1]
        x_c = torch.complex(x_real, x_imag)                # (B, L, D/2) complex

        rot = rot.to(dtype=x_c.dtype)                      # dtype 对齐（complex64/complex128）
        y_c = x_c * rot                                    # 旋转

        y = torch.stack((y_c.real, y_c.imag), dim=-1).reshape(B, L, D)
        return y


class RoPENd(torch.nn.Module):
    """N-dimensional Rotary Positional Embedding."""
    def __init__(self, shape, base=10000):
        super(RoPENd, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0, f'shape[-1] ({feature_dim}) is not divisible by 2 * len(shape[:-1]) ({2 * len(channel_dims)})'

        # tensor of angles to use
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))

        # create a stack of angles multiplied by position
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # convert to complex number to allow easy rotation
        rotations = torch.polar(torch.ones_like(angles), angles)

        # store in a buffer so it can be saved in model parameters
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        # convert input into complex numbers to perform rotation
        *prefix, D = x.shape
        assert D % 2 == 0
        x_pair = x.reshape(*prefix, D // 2, 2)
        xr, xi = x_pair[..., 0], x_pair[..., 1]
        x_c = torch.complex(xr, xi)
        rot = self.rotations.to(x.device, dtype=x_c.dtype)
        pe_x = rot * x_c
        y = torch.stack((pe_x.real, pe_x.imag), dim=-1).reshape(*prefix, D)
        return y


class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        """
        Args:
            input_feat: (N, L, D)
        """
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images. (To 1D sequences)
    """
    def __init__(self, num_pos_feats=512, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        """
        Args:
            x: torch.tensor, (batch_size, L, d)
            mask: torch.tensor, (batch_size, L), with 1 as valid

        Returns:

        """
        assert mask is not None
        x_embed = mask.cumsum(1, dtype=torch.float32)  # (bsz, L)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t  # (bsz, L, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # (bsz, L, num_pos_feats*2)
        # import ipdb; ipdb.set_trace()
        return pos_x  # .permute(0, 2, 1)  # (bsz, num_pos_feats*2, L)

class ZeroPositionalEncoding(nn.Module):
    """Return zeros with the same shape as x (B, L, D). mask 参数仅为接口对齐。"""
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        return torch.zeros_like(x)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, mask):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim

    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    elif args.position_embedding in ('rope', 'rope1d'):
        # 正统 RoPE：在注意力里旋转 Q/K；视频侧 pos 不再做加法，返回全零即可
        position_embedding = PositionEmbeddingRoPE1D(
                 hidden_size=N_steps, max_len=args.max_v_l, base=getattr(args, "rope_base", 10000)
             )

        # 如果你还想保留“加性 RoPE”的旧实验，可以用一个开关区分：
        # if getattr(args, "rope_in_attention", True):
        #     position_embedding = ZeroPositionalEncoding()
        # else:
        #     position_embedding = PositionEmbeddingRoPE1D(
        #         hidden_size=N_steps, max_len=args.max_v_l, base=getattr(args, "rope_base", 10000)
        #     )

    else:
        raise ValueError(f"not supported {args.position_embedding}")

    # 文本侧：保持你现有逻辑
    txt_pos_embed = TrainablePositionalEncoding(
        max_position_embeddings=args.max_q_l,
        hidden_size=args.hidden_dim, dropout=args.input_dropout)

    if getattr(args, "txt_position_embedding", "learned") in ("rope", "rope1d"):
        txt_pos_embed = PositionEmbeddingRoPE1D(
            hidden_size=N_steps,
            max_len=args.max_q_l,
            base=getattr(args, "rope_base", 10000)
        )

    return position_embedding, txt_pos_embed

