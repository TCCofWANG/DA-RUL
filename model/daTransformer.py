import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer, IEncoderLayer, IEncoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer, DegradationAttention, M_FullGaussianAttention
from layers.Embed import DataEmbedding_inverted
import numpy as np


class daTransformer(nn.Module):

    def __init__(self, sequence_len, feature_len, hidden_dim,
                 fc_layer_dim, rnn_num_layers, fc_activation,fc_dropout=0.3,
                 feature_head_num=None, factor=3, output_attention=False):
        super(daTransformer, self).__init__()
        self.seq_len = sequence_len
        self.pred_len = sequence_len
        self.output_attention = output_attention
        self.use_norm = False
        self.n_heads = rnn_num_layers
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(sequence_len, hidden_dim,  fc_dropout)#
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = IEncoder(
            [
                IEncoderLayer(
                    AttentionLayer(
                        #FullAttention(False, 1, attention_dropout=fc_dropout, output_attention=False),
                        DegradationAttention(self.n_heads, attention_dropout=fc_dropout, output_attention=self.output_attention),
                        hidden_dim, feature_head_num),
                    hidden_dim,
                    fc_layer_dim,
                    dropout=fc_dropout,
                    activation=fc_activation
                ) for l in range(self.n_heads)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_dim)
        )
        self.decoder = nn.Linear(hidden_dim, sequence_len, bias=True)

    def forward(self, x_enc, x_mark_enc=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        x_enc = x_enc.permute(0,2,1)
        emb_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules

        # B N E -> B N S -> B S N
        #pro_out = self.projector(enc_out) #just changed
        #
        # if self.use_norm:
        #     # De-Normalization from Non-stationary Transformer
        #     dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #     dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))




        # def forward(self, x_enc, x_mark_enc=None, mask=None):
        #     dec_out = self.forecast(x_enc, x_mark_enc)
        #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        # if self.output_attention:
        #     enc_out, att_weight = self.encoder(emb_out, attn_mask=None)
        #     return enc_out.permute(0, 2, 1), att_weight
        # else:
        enc_out = self.encoder(emb_out, attn_mask=None)
        #dec_out = self.decoder(enc_out) #just changed
        return enc_out.permute(0, 2, 1)