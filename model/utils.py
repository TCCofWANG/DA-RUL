from torch import nn
RNN_MAP= {
    'rnn': nn.RNN,
    'lstm': nn.LSTM,
    'gru': nn.GRU,
    'conv' : nn.Conv1d
}
ACTIVATION_MAP = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'elu': nn.ELU,
    'leakyrelu': nn.LeakyReLU,
    'silu': nn.SiLU
}