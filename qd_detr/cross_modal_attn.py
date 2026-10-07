import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor

# 复用现有多头注意力函数（来自原代码）
from attention import multi_head_attention_forward


class CrossModalMutualAttention(nn.Module):
    """
    跨模态互注意力模块：实现两个模态（如文本X和视频Y）的双向注意力交互。
    包含两个方向的注意力计算：
    1. X→Y：用模态X的特征作为query，模态Y的特征作为key/value
    2. Y→X：用模态Y的特征作为query，模态X的特征作为key/value
    最终输出两个模态的增强特征。
    """
    def __init__(
        self,
        x_dim: int,  # 模态X的特征维度（如文本）
        y_dim: int,  # 模态Y的特征维度（如视频）
        num_heads: int,  # 注意力头数
        dropout: float = 0.0,
        add_zero_attn: bool = False,
        out_dim: Optional[int] = None  # 输出特征维度（默认与输入模态维度相同）
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.out_dim_x = out_dim if out_dim is not None else x_dim
        self.out_dim_y = out_dim if out_dim is not None else y_dim

        # 检查头维度是否合法
        assert self.x_dim % num_heads == 0, f"x_dim {x_dim} must be divisible by num_heads {num_heads}"
        assert self.y_dim % num_heads == 0, f"y_dim {y_dim} must be divisible by num_heads {num_heads}"
        self.head_dim_x = self.x_dim // num_heads
        self.head_dim_y = self.y_dim // num_heads

        # 输出投影层（分别用于两个模态的增强特征）
        self.out_proj_x = nn.Linear(y_dim, self.out_dim_x)  # X→Y注意力的输出投影（Y的value维度→X的输出维度）
        self.out_proj_y = nn.Linear(x_dim, self.out_dim_y)  # Y→X注意力的输出投影（X的value维度→Y的输出维度）

        # 可选：是否添加零注意力（复用原逻辑）
        self.add_zero_attn = add_zero_attn

        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        # 初始化输出投影层偏置
        nn.init.constant_(self.out_proj_x.bias, 0.)
        nn.init.constant_(self.out_proj_y.bias, 0.)
        # 初始化输出投影层权重
        nn.init.xavier_uniform_(self.out_proj_x.weight)
        nn.init.xavier_uniform_(self.out_proj_y.weight)

    def forward(
        self,
        x: Tensor,  # 模态X特征：(Lx, N, x_dim)，Lx为序列长度，N为batch_size
        y: Tensor,  # 模态Y特征：(Ly, N, y_dim)，Ly为序列长度
        x_mask: Optional[Tensor] = None,  # 模态X的padding掩码：(N, Lx)，True表示padding
        y_mask: Optional[Tensor] = None,  # 模态Y的padding掩码：(N, Ly)，True表示padding
        attn_mask: Optional[Tensor] = None,  # 注意力掩码：可选2D或3D，阻止特定位置交互
        need_weights: bool = False  # 是否返回注意力权重
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        返回：
            x_enhanced: 模态X的增强特征 (Lx, N, out_dim_x)
            y_enhanced: 模态Y的增强特征 (Ly, N, out_dim_y)
            attn_weights_x2y: X→Y的注意力权重（可选）
            attn_weights_y2x: Y→X的注意力权重（可选）
        """
        # --------------------------
        # 1. X→Y注意力：X关注Y
        # query = x，key=value=y
        # --------------------------
        x2y_output, attn_weights_x2y = multi_head_attention_forward(
            query=x,
            key=y,
            value=y,
            embed_dim_to_check=self.x_dim,
            num_heads=self.num_heads,
            in_proj_weight=None,  # 不使用输入投影（直接用原始特征）
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj_x.weight,
            out_proj_bias=self.out_proj_x.bias,
            training=self.training,
            key_padding_mask=y_mask,  # key是Y，用Y的掩码
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,  # 不使用分离的投影权重（直接用原始特征）
            out_dim=self.out_dim_x
        )

        # --------------------------
        # 2. Y→X注意力：Y关注X
        # query = y，key=value=x
        # --------------------------
        y2x_output, attn_weights_y2x = multi_head_attention_forward(
            query=y,
            key=x,
            value=x,
            embed_dim_to_check=self.y_dim,
            num_heads=self.num_heads,
            in_proj_weight=None,  # 不使用输入投影
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj_y.weight,
            out_proj_bias=self.out_proj_y.bias,
            training=self.training,
            key_padding_mask=x_mask,  # key是X，用X的掩码
            need_weights=need_weights,
            attn_mask=attn_mask.transpose(1, 2) if attn_mask is not None else None,  # 掩码转置（Ly×Lx）
            use_separate_proj_weight=False,
            out_dim=self.out_dim_y
        )

        # 增强特征 = 原始特征 + 注意力输出（残差连接）
        x_enhanced = x + x2y_output
        y_enhanced = y + y2x_output

        return x_enhanced, y_enhanced, attn_weights_x2y, attn_weights_y2x

if __name__ == '__main__':
    cross_attn = CrossModalMutualAttention(512,512,8,0.1, )
    m = cross_attn(torch.randn(32,128,512), torch.randn(32,128,512))