import torch
import torch.nn as nn

import numpy as np
from math import sqrt
from .utils import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # B:批次大小, L:序列长度, H:头数, E:特征维度
        _, S, _, D = values.shape  # S:源序列长度, D:值的特征维度
        scale = self.scale or 1. / sqrt(E)  # 如果没有预设缩放因子，就使用 1/√E，这是 Transformer 中的标准做法，用于稳定训练。
        """
            einsum函数的参数"blhe,she->bhls"定义了张量之间的运算规则
            Einsum 根据维度标签自动进行以下操作：
                相同标签出现在两个输入中但不在输出中 → 对该维度求和（内积）
                标签只在一个输入中出现 → 保留该维度
                标签在输出中出现 → 该维度会被保留。
        """
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                # 因果掩码是一个下三角掩码，用于确保在序列建模时，当前位置只能"看到"之前的位置，不能"看到"未来的位置。
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            """
                # 原始分数矩阵（假设）
                    scores = [
                        [2.1, 1.5, 0.8, 1.2],
                        [1.3, 2.0, 1.1, 0.9], 
                        [0.7, 1.4, 2.2, 1.0],
                        [1.1, 0.6, 1.3, 1.8]
                    ]
                    
                    # 应用掩码后（True的位置变成-∞）
                    masked_scores = [
                        [2.1, -∞,  -∞,  -∞ ],
                        [1.3, 2.0, -∞,  -∞ ],
                        [0.7, 1.4, 2.2, -∞ ],
                        [1.1, 0.6, 1.3, 1.8]
                    ]
                    softmax后
                    [1.0, 0.0, 0.0, 0.0],  # "我"只关注自己
                    [0.27, 0.73, 0.0, 0.0], # "爱"关注"我"和自己
                    [0.09, 0.18, 0.73, 0.0], # "学"关注前三个词
                    [0.20, 0.12, 0.24, 0.44] # "习"关注所有词
            """
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # 注意力权重 × 值矩阵
        V = torch.einsum("bhls,bshd->blhd", A, values)
        """
            是否把注意力权重输出出去，作用：
                4.1 分析模型在关注什么和可视化
                # 可视化注意力模式
                import matplotlib.pyplot as plt
                plt.imshow(attn_weights[0, 0].detach().numpy())  # 显示第一个头的注意力
                plt.title("注意力权重热力图")
                plt.show()
                
                4.2 调试和研究
                理解模型行为：看模型在关注输入的哪些部分
                发现问题：检查注意力是否合理分布
                论文研究：分析不同层的注意力模式
                
                4.3 多任务学习
                
                注意力蒸馏：将教师模型的注意力传递给学生模型
                注意力正则化：约束注意力的分布模式
        """
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
