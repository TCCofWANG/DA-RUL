from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer,IEncoderLayer,IEncoder
from layers.SelfAttention_Family import M_FullAttention, AttentionLayer, GaussianAttention, M_FullGaussianAttention, \
    DegradationAttention
from layers.Embed import DataEmbedding
from layers.RevIN import RevIN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ITransformer_Enc(nn.Module):
    def __init__(self, sequence_len, feature_len, hidden_dim,
                 fc_layer_dim, rnn_num_layers, fc_activation,fc_dropout=0.3,
                 feature_head_num=None, factor=3):
        super(ITransformer_Enc, self).__init__()
        self.seq_len = sequence_len
        #self.pred_rul = output_dim
        self.output_attention = False
        self.attention_used_time=None
        self.revin = True
        self.dropout = fc_dropout
        self.channels = feature_len
        self.d_model = hidden_dim
        self.n_heads = feature_head_num

        #self.attention_order = attention_order
        self.activation = fc_activation
        self.d_ff = fc_layer_dim
        self.e_layers = rnn_num_layers
        self.factor =  1 / sqrt(hidden_dim) #0.3

        self.output_attention = False
        self.attention_used_time=None
        self.revin = True


        # Embedding

        self.enc_embedding = DataEmbedding(self.seq_len , self.d_model, 'timeF', 'h',
                                           self.dropout)

        self.temporal_embedding = nn.Linear(self.seq_len, self.d_model)

        self.temporal_output = nn.Linear(self.d_model, self.d_model)

        #self.output = nn.Linear(self.channels,self.pred_rul)

        if self.revin:self.revin_layer = RevIN(self.d_model, affine=False, subtract_last=False)

        # Encoder-only architecture
        self.encoder = IEncoder(
            [
                IEncoderLayer(
                    AttentionLayer(
                        DegradationAttention(self.n_heads, attention_dropout=fc_dropout),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=fc_dropout,
                    activation=fc_activation
                ) for l in range(self.n_heads)
            ],

            norm_layer=torch.nn.BatchNorm1d(self.channels)
        )

    def forward(self, x_enc):

        # if self.revin:
        #     #[B,L,D]
        #     x_enc = self.revin_layer(x_enc, 'norm')

        # x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.enc_embedding(x_enc) #[B,L,d_model]

        #enc_out = self.temporal_embedding(enc_out)
        enc_out = self.encoder(enc_out, attn_mask=None)

        enc_out = enc_out
        #outputs = self.output(self.temporal_output(enc_out.permute(0,2,1)).permute(0,2,1))
        #sequeezout= outputs[:, -1, :]
        return enc_out
class ITransformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, cell_type, sequence_len, feature_num, hidden_dim,
                 fc_layer_dim, rnn_num_layers, output_dim, fc_activation,
                 feature_head_num=None,
                 fc_dropout=0.1):
        super(ITransformer, self).__init__()
        self.seq_len = sequence_len
        self.seq_len = sequence_len
        self.pred_rul = output_dim
        self.output_attention = False
        self.attention_used_time=None
        self.revin = True
        self.dropout = fc_dropout
        self.channels = feature_num
        self.d_model = hidden_dim
        self.n_heads = feature_head_num

        self.activation = fc_activation
        self.d_ff = fc_layer_dim
        self.e_layers = rnn_num_layers
        self.factor =  1 / sqrt(hidden_dim) #0.3

        self.output_attention = False
        self.attention_used_time=None
        self.revin = True

        self.channels = feature_num

        # Embedding

        # self.enc_embedding = DataEmbedding(self.channels , configs.d_model, 'timeF', 'h',
        #                                    configs.dropout)

        self.temporal_embedding = nn.Linear(self.seq_len, self.d_model)

        self.temporal_output = nn.Linear(self.d_model, self.seq_len)

        self.output = nn.Linear(self.channels,self.pred_rul)

        if self.revin:self.revin_layer = RevIN(self.d_model, affine=False, subtract_last=False)

        # Encoder
        self.encoder = IEncoder(
            [
                IEncoderLayer(
                    AttentionLayer(
                        M_FullAttention(self.factor, attention_dropout=self.dropout,
                                      output_attention=False), self.d_model, self.n_heads,d_keys=self.d_model,d_values=self.d_model),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            # ITransformer的源码是这么对这一维做norm
            norm_layer=torch.nn.BatchNorm1d(self.channels)
        )

    def forward(self, x_enc):

        if self.revin:
            #[B,L,D]
            x_enc = self.revin_layer(x_enc, 'norm')


        # enc_out = self.enc_embedding(x_enc) #[B,L,d_model]

        enc_out = self.temporal_embedding(x_enc.permute(0,2,1))
        enc_out = self.encoder(enc_out.permute(0,2,1), attn_mask=None)

        enc_out = enc_out.permute(0,2,1)
        #outputs = self.output(self.temporal_output(enc_out.permute(0,2,1)).permute(0,2,1))
        #sequeezout= outputs[:, -1, :]
        return enc_out


















