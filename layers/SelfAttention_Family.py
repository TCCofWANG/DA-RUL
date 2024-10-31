import torch
import torch.nn as nn
import torch.nn.functional as F



import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import os


class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous()



class M_FullAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(M_FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.inf = -2**32+1

    def forward(self, queries, keys, values,attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1,H,1,1)
            scores.masked_fill_(attn_mask, -np.inf)

        mask = torch.ones_like(scores[0])
        mask = mask.tril(diagonal=0)
        scores = torch.where(mask > 0, scores, (torch.ones_like(mask) * self.inf))

        # 对行做softmax没毛病，取决于下一步和v怎么乘
        A = torch.softmax(scale * scores, dim=-1)

        attention_output = A

        V = torch.einsum("bhls,bshd->blhd", attention_output, values)

        V = V.contiguous()
        return V


class GaussianAttention(nn.Module):
    def __init__(self, input_dim, attention_dropout=0.1):
        super(GaussianAttention, self).__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()

        # Compute Gaussian kernel
        distances = torch.norm(query.unsqueeze(2) - key.unsqueeze(1), dim=-1)  # Pairwise distances
        weights = torch.exp(-distances)  # Gaussian kernel weights

        # Apply mask if provided
        if mask is not None:
            weights = weights.masked_fill(mask == 0, 0)

        # Normalize weights
        attention_weights = F.softmax(weights, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        output = torch.matmul(attention_weights, value)

        return output

class M_FullGaussianAttention_smilarty(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(M_FullGaussianAttention_smilarty, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.inf = -2**32 + 1

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Compute pairwise squared distances
        queries = queries / sqrt(E)  # Normalization
        keys = keys / sqrt(E)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        dist_scores = -torch.cdist(queries.view(B * H, L, E), keys.view(B * H, S, E), p=2).pow(2).view(B, H, L, S)

        # Compute Gaussian kernel
        gauss_scores = torch.exp(dist_scores)

        # Combine original attention scores with Gaussian scores
        scores = scores * gauss_scores

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, H, 1, 1)
            scores.masked_fill_(attn_mask, -float('inf'))

        # Apply causal mask
        mask = torch.ones_like(scores[0])
        mask = mask.tril(diagonal=0)
        scores = torch.where(mask > 0, scores, torch.ones_like(mask) * self.inf)

        A = torch.softmax(scale * scores, dim=-1)
        A = self.dropout(A)

        V = torch.einsum('bhls,bshd->blhd', A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous()

class M_FullGaussianAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.3, output_attention=False):
        super(M_FullGaussianAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.inf = -2**32 + 1

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Compute pairwise squared distances
        queries = queries / sqrt(E)  # Normalization
        keys = keys / sqrt(E)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        dist_scores = -torch.cdist(queries.view(B * H, L, E), keys.view(B * H, S, E), p=2).pow(2).view(B, H, L, S)

        # Compute Gaussian kernel
        gauss_scores = torch.exp(dist_scores)
        # Compute inverse Gaussian kernel
        #inverse_gauss_scores = 1 - torch.exp(-dist_scores)  # Higher scores for larger distances
        # Combine original attention scores with Gaussian scores
        scores = scores * gauss_scores

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, H, 1, 1)
            scores.masked_fill_(attn_mask, -float('inf'))

        # Apply causal mask
        mask = torch.ones_like(scores[0])
        mask = mask.tril(diagonal=0)
        scores = torch.where(mask > 0, scores, torch.ones_like(mask) * self.inf)

        A = torch.softmax(scale * scores, dim=-1)
        A = self.dropout(A)

        V = torch.einsum('bhls,bshd->blhd', A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous()

class DegradationAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super(DegradationAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.inf = -2**32 + 1

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / sqrt(E)

        # Compute pairwise squared distances
        # queries = queries / sqrt(E)  # Normalization
        # keys = keys / sqrt(E)
        # values = values / sqrt(E)  #normalize score is optional
        #scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        dist_scores = torch.cdist(queries.view(B * H, L, E), keys.view(B * H, S, E), p=2).pow(2).view(B, H, L, S)
        dist_scores = self.dropout(dist_scores)
        # Compute Gaussian kernel
        #gauss_scores = torch.exp(dist_scores)
        # Compute inverse Gaussian kernel
        inverse_scores = 1 - torch.exp(-dist_scores)  # Higher scores for larger distances
        scores = self.dropout(inverse_scores)
        # Combine original attention scores with Gaussian scores
        #scores = scores * gauss_scores

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, H, 1, 1)
            scores.masked_fill_(attn_mask, -float('inf'))

        # Apply causal mask
        mask = torch.ones_like(scores[0])
        mask = mask.tril(diagonal=0)
        scores = torch.where(mask > 0, scores, torch.ones_like(mask) * self.inf)

        A = torch.softmax(scale * scores, dim=-1)
        A = self.dropout(A)

        V = torch.einsum('bhls,bshd->blhd', A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous()

class D_FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(D_FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.degradation_net = nn.Sequential(
            nn.Linear(128 , 64),  # Input size is H * D
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def calculate_degradation(self, values):
        """
        Calculate degradation scores using a neural network.
        """
        B, S, H, D = values.shape
        values_flat = values.view(B * S, H * D)  # Flatten for linear layer
        degradation_scores = self.degradation_net(values_flat)  # (B*S, 1)
        degradation_scores = degradation_scores.view(B, S)  # Reshape to (B, S)
        degradation_scores = torch.clamp(degradation_scores, min=0.01, max=0.99)  # Clip scores to avoid extreme values
        return degradation_scores

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries, keys, values: (B, L, H, E)
        attn_mask: (B, L, S)
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Ensure inputs do not contain NaNs
        if torch.isnan(queries).any() or torch.isnan(keys).any() or torch.isnan(values).any():
            raise ValueError("Input contains NaNs")

        # Calculate attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Calculate degradation scores
        degradation_scores = self.calculate_degradation(values)  # (B, S)
        degradation_scores = degradation_scores.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        degradation_scores = degradation_scores.expand(B, H, L, S)  # (B, H, L, S)


        # Subtract the maximum value in scores for numerical stability
        degradation_scores = degradation_scores - degradation_scores.max(dim=-1, keepdim=True).values


        # Apply degradation scores to attention scores
        scores = scores + degradation_scores

        # Subtract the maximum value in scores for numerical stability
        #scores = scores - scores.max(dim=-1, keepdim=True).values

        # Apply softmax and dropout to the scores
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous()



class T2V_FullAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(T2V_FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values,x_date,y_date):
        B, L, E = queries.shape
        _, S, D = values.shape
        _, _, k = x_date.shape
        scale = self.scale or 1. / sqrt(k)

        scores = torch.einsum("ble,bhe->blh", x_date, y_date)
        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("ble,blh->bhe", values, A)


        return V.contiguous()

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

    def forward(self, queries, keys, values,attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        return out


class T2V_AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(T2V_AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values,x_date,y_date):
        B, L, _ = queries.shape
        _, S, _ = keys.shape

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        out = self.inner_attention(
            queries,
            keys,
            values,
            x_date,
            y_date
        )

        return self.out_projection(out)



class FullAttention_Full_t2v(nn.Module):
    '''完成了多头映射，多头Attention，Feedforward，normal部分。'''
    def __init__(self, args,factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention_Full_t2v, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.activation = nn.GELU()
        self.n_heads=args.n_heads
        # Feed Forward
        self.conv1 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_ff, out_channels=args.d_model, kernel_size=1)
        # Attention norm
        self.norm1 = nn.BatchNorm1d(args.d_model)
        self.norm2 = nn.BatchNorm1d(args.d_model)
        # Attention projection

        # hidden_size 干脆放一个更高维的了
        hidden_size = args.n_heads * (args.d_model // args.n_heads)
        self.project_in_q = nn.Linear(args.d_model, hidden_size)
        self.project_in_k = nn.Linear(args.d_model, hidden_size)
        self.project_in_v = nn.Linear(args.d_model, hidden_size)
        self.project_out = nn.Linear(hidden_size, args.d_model)

    def forward(self, queries, keys, values):
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = self.project_in_q(queries).reshape(B,L,self.n_heads,-1)
        keys = self.project_in_k(keys).reshape(B, S, self.n_heads, -1)
        values = self.project_in_v(values).reshape(B, S, self.n_heads, -1)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # reshape
        B,L,H,D = V.shape
        V = V.reshape(B,L,-1)
        # Attention projection
        V = self.project_out(V)
        # Attention norm
        x = V
        y = x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        # Feed forward
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # ResNet+Attention norm
        y = self.norm2((x + y).permute(0, 2, 1)).permute(0, 2, 1)
        return y