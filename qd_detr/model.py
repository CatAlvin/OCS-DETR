# OCS-DETR: gated transport alignment and moment-to-highlight refinement.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from qd_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx

from qd_detr.matcher import build_matcher
from qd_detr.transformer import build_transformer
from qd_detr.position_encoding import build_position_encoding
from qd_detr.misc import accuracy
from qd_detr.interaction.test_CQA import VSLFuser

import numpy as np
def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class OTAligner(nn.Module):
    def __init__(self, d_model, sinkhorn_reg=0.1, sinkhorn_iter=100):
        super().__init__()
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_iter = sinkhorn_iter
        self.cost_proj = nn.Linear(d_model, d_model)

    def sinkhorn(self, cost_matrix, a=None, b=None):
        # cost_matrix: (B, L_v, L_t)；返回 plan: (B, L_v, L_t)
        B, L_v, L_t = cost_matrix.shape
        device = cost_matrix.device
        if a is None: a = torch.full((B, L_v), 1.0/max(L_v,1), device=device)
        if b is None: b = torch.full((B, L_t), 1.0/max(L_t,1), device=device)

        K = torch.exp(-cost_matrix / self.sinkhorn_reg)   # (B, L_v, L_t)
        u = torch.ones_like(a); v = torch.ones_like(b)
        for _ in range(self.sinkhorn_iter):
            Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8  # (B, L_v)
            u = a / Kv
            Ku = torch.bmm(u.unsqueeze(1), K).squeeze(1) + 1e-8    # (B, L_t)
            v = b / Ku
        plan = u.unsqueeze(-1) * K * v.unsqueeze(1)
        plan = plan / (plan.sum(dim=(1,2), keepdim=True) + 1e-8)
        return plan

    def forward(self, src_vid, src_txt, src_vid_mask=None, src_txt_mask=None):
        """
        src_vid: (B, L_v, D), src_txt: (B, L_t, D)
        mask: 1=valid, 0=pad
        return: aligned_vid (B,L_v,D), aligned_txt (B,L_t,D)
        """
        v = self.cost_proj(src_vid)
        t = self.cost_proj(src_txt)
        v_n = F.normalize(v, dim=-1)
        t_n = F.normalize(t, dim=-1)
        cos = torch.einsum('bld,btd->blt', v_n, t_n)      # (B,L_v,L_t)
        cost = 1.0 - cos

        if src_vid_mask is not None and src_txt_mask is not None:
            vid_b = src_vid_mask.bool().unsqueeze(-1)     # (B,L_v,1)
            txt_b = src_txt_mask.bool().unsqueeze(1)      # (B,1,L_t)
            valid = vid_b & txt_b
            cost = cost.masked_fill(~valid, 1e6)

        plan = self.sinkhorn(cost)                        # (B,L_v,L_t)
        aligned_vid = torch.bmm(plan, src_txt)            # (B,L_v,D)
        aligned_txt = torch.bmm(plan.transpose(1,2), src_vid)  # (B,L_t,D)
        return aligned_vid, aligned_txt


class GRUFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(GRUFeatureExtractor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        output, hidden = self.gru(x)

        # If bidirectional, concatenate the forward and backward hidden states
        if self.gru.bidirectional:
            # Combine the hidden states of both directions (forward and backward)
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # If not bidirectional, use the hidden state of the last layer
            hidden = hidden[-1, :, :]

        # Output the hidden state(s) as the global feature(s)
        # For a single layer, hidden shape: (batch_size, hidden_dim)
        # For multiple layers, hidden shape: (batch_size, num_layers * hidden_dim)
        return hidden

class QD_DETR(nn.Module):
    """ QD DETR. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, aud_dim=0, clip_len = 2):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         QD-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.clip_len = clip_len
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj

        self.query_embed = nn.Embedding(num_queries, 2)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)
        self.aux_loss = aux_loss

        self.hidden_dim = hidden_dim
        self.saliency_proj1 = nn.Linear(transformer.d_model, transformer.d_model)

        self.fuser = VSLFuser(transformer.d_model)

        self.gru_extractor = GRUFeatureExtractor(hidden_dim, hidden_dim, num_layers=1, bidirectional=False)
        
        self.ot_aligner = OTAligner(d_model=hidden_dim, sinkhorn_reg=0.1, sinkhorn_iter=100)
        self.ot_gamma = nn.Parameter(torch.zeros(1))   # Residual scale initialized to zero; its sign is learned without constraint.
        
        self.gate_proj_vid = nn.Linear(hidden_dim, hidden_dim)



    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_aud=None, src_aud_mask=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
            
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        # CTC
        src_txt_ed = src_txt
        src_vid_ed = src_vid
        # VTC
        src_vid_cls_ed = src_vid.mean(1)
        src_txt_cls_ed = src_txt.mean(1)
        
        
        #! === 门控 OT 对齐（投影后、fuser 前）===
        # Normalize accepted mask layouts to (batch, sequence).
        if src_vid_mask.dim() == 1:
            src_vid_mask = src_vid_mask.unsqueeze(0).expand(src_vid.size(0), -1)
        elif src_vid_mask.dim() == 3 and src_vid_mask.size(1) == 1:
            src_vid_mask = src_vid_mask.squeeze(1)
        if src_txt_mask.dim() == 1:
            src_txt_mask = src_txt_mask.unsqueeze(0).expand(src_txt.size(0), -1)
        elif src_txt_mask.dim() == 3 and src_txt_mask.size(1) == 1:
            src_txt_mask = src_txt_mask.squeeze(1)

        aligned_vid, aligned_txt = self.ot_aligner(
            src_vid, src_txt,
            src_vid_mask=src_vid_mask,
            src_txt_mask=src_txt_mask
        )
        # 残差融合文本（单标量门控）
        src_txt = src_txt + self.ot_gamma * aligned_txt
        
        # 门控融合视频侧
        gate_vid = torch.sigmoid(self.gate_proj_vid(src_vid + aligned_vid))  # 门控参数由线性层学习
        src_vid = gate_vid * aligned_vid + (1 - gate_vid) * src_vid


        
        src_vid = self.fuser(src_vid, src_txt, src_vid_mask, src_txt_mask)
        
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)

        video_length = src_vid.shape[1]
        
        hs, reference, memory, saliency_scores = self.transformer(src, ~mask, self.query_embed.weight, pos,self.saliency_proj1, video_length=video_length)
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}

        # *Refine the highlight score prediction process using the retrieved moments from moment retrieval
        saliency_scores_refined = self.CMAF(out, memory, video_length, src_vid_ed)
        
        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))
            
            
        # !!! this is code for test
        if src_txt.shape[1] == 0:
            print("There is zero text query. You should change codes properly")
            exit(-1)
        out["saliency_scores"] = saliency_scores_refined
        # print(src_vid_mask.shape, src_vid.shape, vid_mem_neg.shape, vid_mem.shape)
        out["video_mask"] = src_vid_mask
        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))

        out['src_txt_ed'] = src_txt_ed
        out['src_vid_ed'] = src_vid_ed
        out['src_vid_cls_ed'] = src_vid_cls_ed
        out['src_txt_cls_ed'] = src_txt_cls_ed
        
        return out


    def smooth_interval_mask(self, T, starts, ends, tau=2.0, device=None):
        """
        生成平滑区间权重（可导），形状：(B, T)
        starts/ends: (B,) 浮点起止（未取整），T 为时间长度
        tau: 温度，越小越硬，越大越软
        """
        B = starts.size(0)
        if device is None:
            device = starts.device
        t = torch.arange(T, device=device).float().unsqueeze(0)  # (1, T)

        # 到区间两端的“距离”——用 sigmoid 近似阶跃
        left = torch.sigmoid((t - starts.unsqueeze(1)) / tau)      # 进入区间
        right = torch.sigmoid((ends.unsqueeze(1) - t) / tau)       # 未越界
        w = left * right                                           # (B, T), 近似 [start, end] 的指示函数
        # 避免全零
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        return w  # (B, T)

    def cosine_time_attention(self, query_global, seq, mask=None):
        """
        query_global: (B, D) 全局向量（可来自 GRU 的片段汇聚）
        seq: (B, T, D) 全序列特征
        mask: (B, T) 可选，1=valid, 0=invalid（会在 softmax 前加 -inf）
        返回:
        attn: (B, T) 余弦注意力（softmax 后）
        """
        # L2 normalize
        q = F.normalize(query_global, dim=-1)            # (B, D)
        k = F.normalize(seq, dim=-1)                     # (B, T, D)
        # 余弦相似度
        sim = torch.einsum('bd,btd->bt', q, k)           # (B, T)
        if mask is not None:
            sim = sim.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(sim, dim=-1)                    # (B, T)
        return attn
    
    def gated_residual_update(self, memory, src_vid_ed, attn, v_proj=None, gate=None):
        """
        memory: (B, T, D)
        src_vid_ed: (B, T, D)
        attn: (B, T)
        v_proj: 可选的线性层，把 value 做个投影
        gate:   可选的门控参数（标量或 (B,1,1)），默认 0.5
        返回更新后的 memory
        """
        B, T, D = memory.shape
        if v_proj is not None:
            values = v_proj(src_vid_ed)                  # (B, T, D)
        else:
            values = src_vid_ed
        # 聚合值向量： \sum_t attn_t * v_t
        v_agg = torch.einsum('bt,btd->bd', attn, values).unsqueeze(1)  # (B, 1, D)
        if gate is None:
            alpha = 0.5
        else:
            alpha = gate
        return memory + alpha * v_agg.expand(-1, T, -1)
    

    def batched_soft_window_pool(self, spans_xx, video_length, clip_len, T, src_vid_ed, tau_mask=2.0):
        """
        spans_xx: (B, 2)
        video_length: 可能是 int / list / numpy / Tensor；语义是“该样本包含的 clip 数”
        T: src_vid_ed 的时间维长度
        """
        device = src_vid_ed.device
        B = src_vid_ed.size(0)

        # --- NEW: 统一把 video_length 转为 (B,) 的 float32 Tensor ---
        if isinstance(video_length, torch.Tensor):
            vlen = video_length.to(device=device, dtype=torch.float32)
            if vlen.dim() == 0:
                vlen = vlen.repeat(B)
        else:
            # int / list / numpy array
            try:
                # 若是标量，扩成 (B,); 若是序列，直接转 tensor
                import numpy as np
                if np.isscalar(video_length):
                    vlen = torch.full((B,), float(video_length), device=device, dtype=torch.float32)
                else:
                    vlen = torch.as_tensor(video_length, device=device, dtype=torch.float32)
                    if vlen.dim() == 0:
                        vlen = vlen.repeat(B)
            except Exception:
                # 兜底（比如没有 numpy）
                if isinstance(video_length, (int, float)):
                    vlen = torch.full((B,), float(video_length), device=device, dtype=torch.float32)
                else:
                    vlen = torch.tensor(video_length, device=device, dtype=torch.float32)
                    if vlen.dim() == 0:
                        vlen = vlen.repeat(B)

        # 防止 0
        vlen = vlen.clamp(min=1.0)  # (B,)

        # spans_xx: (B, 2), vlen: (B,)
        s0 = spans_xx[:, 0]
        e0 = spans_xx[:, 1]

        # 先下限裁剪到 0（标量 OK）
        s0 = torch.clamp_min(s0, 0.0)
        e0 = torch.clamp_min(e0, 0.0)

        # 再用逐元素最小值限制上限到 vlen（张量 vs 张量）
        # 若 PyTorch 版本较老没有 torch.minimum，可用 torch.min(s0, vlen)
        s0 = torch.minimum(s0, vlen)
        e0 = torch.minimum(e0, vlen)

        # 映射到索引空间 [0, T-1]
        starts_raw = s0 / vlen * (T - 1)
        ends_raw   = e0 / vlen * (T - 1)


        # end < start 切换
        swapped = (ends_raw < starts_raw)
        starts = torch.where(swapped, ends_raw, starts_raw)
        ends   = torch.where(swapped, starts_raw, ends_raw)

        # 软区间权重（可导）
        t = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)  # (1, T)
        left  = torch.sigmoid((t - starts.unsqueeze(1)) / tau_mask)
        right = torch.sigmoid((ends.unsqueeze(1) - t) / tau_mask)
        mask_t = left * right                                       # (B, T)
        mask_t = mask_t / (mask_t.sum(dim=1, keepdim=True) + 1e-6)

        # 窗口内加权平均作为“局部全局”表示
        pooled = torch.einsum('bt,btd->bd', mask_t, src_vid_ed)     # (B, D)
        return pooled, mask_t


    def CMAF(self, out, memory, video_length, src_vid_ed, topk=5, tau_prop=0.5, tau_mask=2.0):
        """
        改进版 MR->HD（Coarse-to-Fine Moment Attention Fusion（CMAF）｜“粗到细片段注意融合”“多提议软窗注意门控回写”）：
        1) 用 top-K proposal + 软选择融合窗口
        2) 用软区间权重在整段上做可导的加权池化（替代硬切片）
        3) 用真正的 cosine 注意力得到时序权重
        4) 用门控残差把“值”写回 memory（而不是放大 memory）
        其余模块（saliency_proj1 等）复用你现有实现
        """
        B, T, D = src_vid_ed.shape
        device = src_vid_ed.device

        # 1) 取分类概率并选 top-K（假设第0类为“目标/前景”）
        prob = F.softmax(out["pred_logits"], dim=-1)              # (B, Q, C)
        scores = prob[..., 0]                                     # (B, Q)
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        k = min(topk, sorted_indices.size(1))
        topk_idx = sorted_indices[:, :k]                          # (B, K)

        # 2) 取对应 span（cxw->xx）并映射到索引空间，做软融合
        pred_spans = out['pred_spans']                            # (B, Q, 2) in cxw
        spans_xx_all = span_cxw_to_xx(pred_spans)                 # (B, Q, 2) in xx
        # 取出 top-K spans
        batch_ids = torch.arange(B, device=device).unsqueeze(-1).expand(B, k)  # (B, K)
        topk_spans_xx = spans_xx_all[batch_ids, topk_idx, :]                    # (B, K, 2)

        # 把 top-K 窗口合成为一个“软窗口”：用 top-K 分数做 softmax 加权
        topk_scores = sorted_scores[:, :k]                         # (B, K)
        prop_w = F.softmax(topk_scores / tau_prop, dim=-1)         # (B, K)

        # 逐个窗口做软池化，再按 prop_w 加权融合
        pooled_list = []
        mask_list = []
        for j in range(k):
            pooled_j, mask_j = self.batched_soft_window_pool(
                topk_spans_xx[:, j, :], video_length, self.clip_len, T, src_vid_ed, tau_mask=tau_mask
            )
            pooled_list.append(pooled_j)
            mask_list.append(mask_j)
        pooled_stack = torch.stack(pooled_list, dim=1)             # (B, K, D)
        mask_stack   = torch.stack(mask_list, dim=1)               # (B, K, T)

        # 融合 top-K
        pooled_global = torch.einsum('bk,bkd->bd', prop_w, pooled_stack)   # (B, D)
        fused_mask    = torch.einsum('bk,bkt->bt', prop_w, mask_stack)     # (B, T)  作为注意力可用的时间先验

        # 3) 用“局部全局”向量做真正的 cosine 注意力（结合先验 mask）
        # 注意：mask 用 logit 加法方式融合（把无效位置 -inf）
        prior_logits = torch.log(fused_mask + 1e-6)                # (B, T)
        # 先算纯 cosine
        attn_cos = self.cosine_time_attention(pooled_global, src_vid_ed, mask=None)  # (B, T)
        # 融合先验：cosine 的 logits + 先验 logits，再 softmax
        logits = torch.log(attn_cos + 1e-6) + prior_logits
        attn = F.softmax(logits, dim=-1)                           # (B, T)

        # 4) 门控残差写回 memory（引入 Value 分支，而不是对 memory 纯尺度放大）
        # 可选值投影：self.value_proj（建议你在 __init__ 里加一个 nn.Linear(D,D)）
        v_proj = getattr(self, 'value_proj', None)
        gate = getattr(self, 'hd_gate', None)  # 可是标量超参/nn.Parameter(1).sigmoid()等
        memory_updated = self.gated_residual_update(memory, src_vid_ed, attn, v_proj=v_proj, gate=gate)

        # 5) 计算 saliency（建议：用你之前那种“与文本全局向量点积/缩放”的方式更合理；
        # 这里保留你原先的写法，但你可以把 memory_updated 与 text_global 做缩放点积）
        saliency_scores = torch.sum(self.saliency_proj1(memory_updated), dim=-1) / (D ** 0.5)

        return saliency_scores




class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, use_matcher=True):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        
        # for tvsum,
        self.use_matcher = use_matcher

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries) # 1
        target_classes[idx] = self.foreground_label # 0
        target_classes = F.one_hot(target_classes, num_classes=2).permute(0, 2, 1) # (32, 10, 2)
        src_logits = src_logits.to(torch.float32)
        target_classes = target_classes.to(torch.float32)
        import torchvision  # Used by the training classification loss only.
        loss_ce = torchvision.ops.focal_loss.sigmoid_focal_loss(src_logits.transpose(1, 2), target_classes, alpha=0.5, gamma=2.5, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses
    
    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        vid_token_mask = outputs["video_mask"]
        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]
        saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

        tau = 0.5
        loss_rank_contrastive = 0.

        # for rand_idx in range(1, 13, 3):
        #     # 1, 4, 7, 10 --> 5 stages
        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3

            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12
        
        # margin: Moment-DETR
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
        loss_saliency = loss_saliency + loss_rank_contrastive 
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        # TODO (1)  align vid_mem and txt_mem;
        # TODO (2) change L1 loss as CE loss on 75 labels, similar to soft token prediction in MDETR
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        # only for HL, do not use matcher
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
            losses_target = self.losses
        else:
            indices = None
            losses_target = ["saliency"]

        # Compute all the requested losses
        losses = {}
        # for loss in self.losses:
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets)
                    losses_target = self.losses
                else:
                    indices = None
                    losses_target = ["saliency"]    
                # for loss in self.losses:
                for loss in losses_target:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/qd_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)
    # 根据是否使用音频来构建模型S
    if args.a_feat_dir is None:
        model = QD_DETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            clip_len=args.clip_length
        )
    else:
        model = QD_DETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            aud_dim=args.a_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            clip_len=args.clip_length
        )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
        
    # For tvsum dataset
    use_matcher = not (args.dset_name == 'tvsum')
        
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin, use_matcher=use_matcher,
    )
    criterion.to(device)
    return model, criterion
