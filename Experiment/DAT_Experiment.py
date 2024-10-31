import torch
from torch import nn
from model.daTransformer import daTransformer

from model.utils import ACTIVATION_MAP, RNN_MAP
from model.vTransformer import vTransformer
from xLSTM import sLSTM, mLSTM


class DAT_Experiment(nn.Module):
    def __init__(self, cell_type, sequence_len, feature_num, hidden_dim, lstm_dim,
                 fc_layer_dim, rnn_num_layers, output_dim, fc_activation,
                 attention_type, feature_head_num=None, sequence_head_num=None,
                 fc_dropout=0, rnn_dropout=0, bidirectional=False, return_attention_weights=False, alpha=0.5, beta=0.5):
        super().__init__()
        assert cell_type in ['rnn', 'lstm', 'gru', 'grucell', 'slstm', 'mlstm', 'xlstm']
        assert fc_activation in ['tanh', 'gelu', 'relu', 'silu', 'leakyrelu']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.attention_type = attention_type
        self.return_attention_weights = return_attention_weights
        self.feature_num = feature_num
        self.sequence_len = sequence_len

        self.cell = cell_type
        self.hidden_dim = hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.bidirectional = bidirectional

        self.fc_layer_dim = fc_layer_dim
        self.fc_dropout = fc_dropout
        self.fc_activation = fc_activation

        self.output_dim = output_dim
        self.rnn_dropout = rnn_dropout

        self.feature_head_num = feature_head_num
        self.sequence_head_num = sequence_head_num
        self.alpha = alpha
        self.beta = beta

        self.dropout = nn.Dropout(self.fc_dropout)

        if attention_type == 'deg_attention':
            # Initialize the sequence attention module
            self.attention_enc = daTransformer(sequence_len, feature_num, hidden_dim,
                                               fc_layer_dim, rnn_num_layers, fc_activation,
                                               fc_dropout, sequence_head_num, return_attention_weights)

        elif attention_type == 'vanilla_attention':
            # Initialize the feature attention module
            self.attention_enc = vTransformer(sequence_len, feature_num, hidden_dim,
                                              fc_layer_dim, rnn_num_layers, output_dim, fc_activation, feature_head_num,
                                              fc_dropout)

        if cell_type == 'slstm':
            self.RNN = sLSTM(
                input_size=feature_num,
                hidden_size=lstm_dim,  # hidden size ke jaga fc_layer dim
                num_layers=rnn_num_layers,
            )
        elif cell_type == 'mlstm':
            self.RNN = mLSTM(
                input_size=feature_num,
                hidden_size=lstm_dim,  # hidden size ke jaga fc_layer dim
                num_layers=rnn_num_layers,
            )
        else:
            # Initialize the RNN
            self.RNN = RNN_MAP[cell_type](
                input_size=feature_num,  # Adjust input size based on fusion output
                hidden_size=lstm_dim,
                batch_first=True,
                bidirectional=bidirectional,
                num_layers=rnn_num_layers,
                dropout=rnn_dropout
            )

        # Fully connected layers
        self.linear = nn.Sequential(
            nn.Linear(lstm_dim * 2 if bidirectional else lstm_dim, fc_layer_dim),
            ACTIVATION_MAP[fc_activation](),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_layer_dim, output_dim),
        )
        # input_size, hidden_size, hidden_size, num_layers)

    def forward(self, x):
        # x = self.timeEmbedding(x).to(self.device)

        fused_attention = self.attention_enc(x)

        dropout_fused_attention = self.dropout(fused_attention)

        # RNN processing
        rnn_output, _ = self.RNN(dropout_fused_attention)

        dropout_output = self.dropout(rnn_output)  # output_mean = dropout_output.mean(dim=1)

        fc_output = self.linear(dropout_output)
        rul_prediction = torch.abs(fc_output[:, -1, :])  # self.regressor(fc_output)

        # Returning attention weights if needed
        if self.return_attention_weights:
            return rul_prediction, fused_attention
        else:
            return rul_prediction
