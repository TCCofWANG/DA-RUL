"""
mLSTM: Matrix Long Short-Term Memory

This module implements the mLSTM (matrix LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The mLSTM extends the traditional LSTM by using a matrix memory state and exponential gating,
allowing for enhanced storage capacities and improved performance on long-range dependencies.

Author: Mudit Bhargava
Date: June 2024
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class mLSTM(nn.Module):
    """
    mLSTM layer implementation.

    This layer applies multiple mLSTM cells in sequence, with optional dropout between layers.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
        num_layers (int): Number of mLSTM layers.
        dropout (float, optional): Dropout probability between layers. Default: 0.0.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([mLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                                     for i in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass of the mLSTM layer.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_state (tuple of Tensors, optional): Initial hidden state. Default: None.

        Returns:
            tuple: Output sequence and final hidden state.
        """
        batch_size, seq_length, _ = input_seq.size()

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)

        outputs = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            for layer_idx, layer in enumerate(self.layers):
                h, C, n = hidden_state[layer_idx]
                h, C, n = layer(x, (h, C, n))
                hidden_state[layer_idx] = (h, C, n)
                x = self.dropout_layer(h) if layer_idx < self.num_layers - 1 else h
            outputs.append(x)

        return torch.stack(outputs, dim=1), hidden_state

    def init_hidden(self, batch_size):
        """Initialize hidden state for all layers."""
        return [(torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device),
                 torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=self.layers[0].weight_ih.device),
                 torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=self.layers[0].weight_ih.device))
                for _ in range(self.num_layers)]


class mLSTMCell(nn.Module):
    """
    mLSTM cell implementation.

    This cell uses a matrix memory state and exponential gating as described in the xLSTM paper.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
    """

    def __init__(self, input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(3 * hidden_size))

        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)

        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)

    def forward(self, input, hx):
        """
        Forward pass of the mLSTM cell.

        Args:
            input (Tensor): Input tensor of shape (batch_size, input_size).
            hx (tuple of Tensors): Previous hidden state and cell state.

        Returns:
            tuple: New hidden state and cell state.
        """
        h_prev, c_prev, n_prev = hx
        #gates = F.linear(input, self.weight_ih, self.bias) + F.linear(h_prev, self.weight_hh)
        #i, f, o = gates.chunk(3, 1)
        input = input.unsqueeze(-1) # (batch_size, input_size, 1)
        q = torch.matmul(self.W_q.weight, input) + self.W_q.bias
        k = (1/ math.sqrt(self.hidden_size)) * (torch.matmul(self.W_k.weight, input) + self.W_k.bias) #self.W_k(input)
        v = torch.matmul(self.W_v.weight, input) + self.W_v.bias #self.W_v(input)

        v = v.squeeze(-1)  # (batch_size, mem_dim)
        k = k.squeeze(-1)  # (batch_size, mem_dim)


        i = torch.exp(torch.matmul(self.W_i.weight, input).squeeze() + self.W_i.bias) #torch.exp(i)  # Exponential input gate
        f = torch.sigmoid(torch.matmul(self.W_f.weight, input).squeeze() + self.W_f.bias) #torch.exp(f)  # Exponential forget gate
        o = torch.sigmoid(torch.matmul(self.W_o.weight, input) + self.W_o.bias) #torch.sigmoid(o)


        h_tilde = torch.bmm(q, c_prev)
        n_prev_t = n_prev.transpose(1, 2)
        denominator = torch.max(torch.abs(torch.matmul(n_prev_t, q)), torch.tensor(1.0, device=h_tilde.device))
        h = o * (h_tilde / denominator)

        n = f * n_prev + i * k

        c = f.unsqueeze(2) * c_prev + i.unsqueeze(2) * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))


        return h, c, n
