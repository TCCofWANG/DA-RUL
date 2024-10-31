import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float).to(Q.device))
    return torch.softmax(m, -1)

def Sensor_a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float).to(Q.device))
    return torch.softmax(m, -1)

def time_step_a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float).to(Q.device))
    return torch.softmax(m, -1)

# attention
def attention(Q, K, V):
    a = a_norm(Q, K)
    return torch.matmul(a, V)

def Sensor_attention(Q, K, V):
    a = Sensor_a_norm(Q, K)
    return torch.matmul(a, V)

def time_step_attention(Q, K, V):
    a = time_step_a_norm(Q, K)
    return torch.matmul(a, V)

# AttentionBlock
class AttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)

    def forward(self, x, kv=None):
        if kv is None:
            return attention(self.query(x), self.key(x), self.value(x))
        return attention(self.query(x), self.key(kv), self.value(kv))

class Sensor_AttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(Sensor_AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)

    def forward(self, x, kv=None):
        if kv is None:
            return Sensor_attention(self.query(x), self.key(x), self.value(x))
        return Sensor_attention(self.query(x), self.key(kv), self.value(kv))

class time_step_AttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(time_step_AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)

    def forward(self, x, kv=None):
        if kv is None:
            return time_step_attention(self.query(x), self.key(x), self.value(x))
        return time_step_attention(self.query(x), self.key(kv), self.value(kv))

# Multi-head self-attention
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = nn.ModuleList([AttentionBlock(dim_val, dim_attn) for _ in range(n_heads)])
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias=False)

    def forward(self, x, kv=None):
        a = [h(x, kv=kv) for h in self.heads]
        a = torch.stack(a, dim=-1)
        a = a.flatten(start_dim=2)
        x = self.fc(a)
        return x

class Sensor_MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(Sensor_MultiHeadAttentionBlock, self).__init__()
        self.heads = nn.ModuleList([Sensor_AttentionBlock(dim_val, dim_attn) for _ in range(n_heads)])
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias=False)

    def forward(self, x, kv=None):
        a = [h(x, kv=kv) for h in self.heads]
        a = torch.stack(a, dim=-1)
        a = a.flatten(start_dim=2)
        x = self.fc(a)
        return x

class TimeStepMultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(TimeStepMultiHeadAttentionBlock, self).__init__()
        self.heads = nn.ModuleList([time_step_AttentionBlock(dim_val, dim_attn) for _ in range(n_heads)])
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias=False)

    def forward(self, x, kv=None):
        a = [h(x, kv=kv) for h in self.heads]
        a = torch.stack(a, dim=-1)
        a = a.flatten(start_dim=2)
        x = self.fc(a)
        return x

# Query, Key and Value
class Value(nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_val, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x

class Key(nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x

class Query(nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)

    def forward(self, x):
        x = self.fc1(x.to(device))
        return x

# PositionalEncoding (from Transformer)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)


