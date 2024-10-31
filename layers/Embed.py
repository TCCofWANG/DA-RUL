import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class DataEmbedding(nn.Module):
    def __init__(self, c_in,d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        # 这里仅仅是对于x的值进行编码，没有对于时间mark进行编码
        # value就是点聚合卷积了一下，position就是拿固定频率的cos，sin来拟合
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class BWEmbedding(nn.Module):
    def __init__(self, d_model, batch_size, num_tokens):
        super(BWEmbedding, self).__init__()

        self.batch_embed = nn.Embedding(batch_size, d_model)
        self.token_embed = nn.Embedding(num_tokens, d_model)

    def forward(self, x):
        batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand(x.size(0), x.size(2))
        token_indices = torch.arange(x.size(2), device=x.device).unsqueeze(0).expand(x.size(0), x.size(2))

        batch_embeddings = self.batch_embed(batch_indices)
        token_embeddings = self.token_embed(token_indices)

        return batch_embeddings + token_embeddings
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        b = self.emb(x).detach()
        return b

class CyclicEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='embedding', freq='w'):
        super(CyclicEmbedding, self).__init__()

        window_size=40
        features_size=16

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        self.window_embed = Embed(window_size, d_model)
        self.featuer_embed = Embed(features_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        print(x[:, :, 3])
        hour_x = self.hour_embed(x[:, :, 3])

        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='embedding', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        print(x[:, :, 3])
        hour_x = self.hour_embed(x[:, :, 3])

        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='embedding', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        print(x[:, :, 3])
        hour_x = self.hour_embed(x[:, :, 3])

        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in,d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        # 这里仅仅是对于x的值进行编码，没有对于时间mark进行编码
        # value就是点聚合卷积了一下，position就是拿固定频率的cos，sin来拟合
        value_embedding = self.value_embedding(x)
        position_embedding = self.position_embedding(x)
        x = value_embedding + position_embedding

        return self.dropout(x)

class DataEmbedding_onlypos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_onlypos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # try:
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        # except:
        #     a = 1
        return self.dropout(x)


class TimeEmbedding(nn.Module):
    def __init__(self, d_mark, d_model):
        super(TimeEmbedding, self).__init__()
        self.embed = nn.Linear(d_mark, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


'''pyraformer'''




class TokenEmbedding_pyraformer(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding_pyraformer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x




class TimeFeatureEmbedding_pyraformer(nn.Module):
    def __init__(self, d_model):
        super(TimeFeatureEmbedding_pyraformer, self).__init__()

        d_inp = 4
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)

'''Aliformer and Deepar'''
class TokenEmbedding_Aliformer(nn.Module):
    def __init__(self, d_feature, d_model):
        super(TokenEmbedding_Aliformer, self).__init__()
        self.embed = nn.Linear(d_feature, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding_time_token(nn.Module):
    def __init__(self, d_feature, d_mark, d_model):
        super(DataEmbedding_time_token, self).__init__()

        self.value_embedding = TokenEmbedding_Aliformer(d_feature=d_feature, d_model=d_model)
        self.time_embedding = TimeEmbedding(d_mark=d_mark, d_model=d_model)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.time_embedding(x_mark)
        return x



"""Embedding modules. The DataEmbedding is used by the ETT dataset for long range forecasting."""


class DataEmbedding_pyraformer(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_pyraformer, self).__init__()

        self.value_embedding = TokenEmbedding_pyraformer(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding_pyraformer(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)


"""The CustomEmbedding is used by the electricity dataset and app flow dataset for long range forecasting."""


class CustomEmbedding(nn.Module):
    def __init__(self, c_in, d_model, temporal_size, seq_num, dropout=0.1):
        super(CustomEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding_pyraformer(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = nn.Linear(temporal_size, d_model)
        self.seqid_embedding = nn.Embedding(seq_num, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark[:, :, :-1]) \
            + self.seqid_embedding(x_mark[:, :, -1].long())

        return self.dropout(x)

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        #x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]

        if x_mark is None:
            x = self.value_embedding(x) #+ self.position_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)