from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, IEncoderLayer, IEncoder
from layers.SelfAttention_Family import M_FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.RevIN import RevIN


class vTransformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self, sequence_len, feature_num, hidden_dim,
                 fc_layer_dim, rnn_num_layers, output_dim, fc_activation,
                 feature_head_num=None, dropout=float(0.3)):
        super().__init__()
        self.seq_len = sequence_len
        self.pred_rul = output_dim
        self.output_attention = False
        self.attention_used_time = None
        self.revin = True
        self.dropout = dropout
        self.channels = feature_num
        self.d_model = hidden_dim
        self.n_heads = feature_head_num

        self.activation = fc_activation
        self.d_ff = fc_layer_dim
        self.e_layers = rnn_num_layers
        self.factor = 1 / sqrt(hidden_dim)  # 0.3
        # Embedding

        self.timeEmbedding = DataEmbedding(self.channels, d_model=self.d_model)

        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(self.d_model, feature_num)
        self.decoder = nn.Linear(self.d_model, self.pred_rul)

        if self.revin:
            self.revin_layer = RevIN(self.d_model, affine=False, subtract_last=False)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        M_FullAttention(self.factor, attention_dropout=dropout,
                                        output_attention=False), self.d_model, self.n_heads, d_keys=self.d_model,
                        d_values=self.d_model),
                    self.d_model,
                    self.d_ff,
                    dropout=dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],

            norm_layer=torch.nn.BatchNorm1d(self.d_model)
        )

    def forward(self, x_enc):

        if self.revin:
            # [B,L,D]
            x_enc = self.revin_layer(x_enc, 'norm')

        # enc_out = self.enc_embedding(x_enc) #[B,L,d_model]

        enc_out = self.timeEmbedding(x_enc)
        enc_out = self.encoder(enc_out, attn_mask=None)

        enc_out = self.dropout(enc_out)
        output_projection = self.output_projection(enc_out)
        # output2 = self.decoder(output1.squeeze()[:, -1, :])
        # final_out = output2.squeeze()
        # rul_prediction = self.linear(output2.squeeze()[:, -1, :])

        return output_projection  # [B,L]
