import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from dataset import CMAPSSDataset
from torch.nn import functional as F
import pandas as pd
from pytorch_lightning.metrics.functional import mean_squared_error

from Experiment.DAT_Experiment import DAT_Experiment

from utils.RiskAwareLoss import RiskAwareLoss

from utils.log_result import log_args_and_metrics_to_csv
from utils.metrics import score
import random
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Module(pl.LightningModule):
    def __init__(self, lr, alpha=.0005, **kwargs):
        super(Module, self).__init__()
        # self.device ='cuda'
        self.net = DAT_Experiment(**kwargs)
        self.lr = lr
        #self.risk_aware_loss = RiskAwareLoss(1.5, 1, -13, 10)
        # print(self.net)
        # print('lr', self.lr)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self.net(x)
        loss = F.mse_loss(y_pred, y)
        #loss = self.risk_aware_loss(y_pred, y)
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        x = self.net(x)
        loss = F.mse_loss(x, y, reduction='sum')
        return torch.tensor([loss.item(), len(y)])

    def test_step(self, batch, batch_idx, reduction='sum'):
        x, y, id = batch
        x = self.net(x)
        # x = x.cpu()
        return torch.cat([id, torch.floor(x), y], dim=1)

    def test_epoch_end(self, step_outputs):
        t = torch.cat(step_outputs, dim=0)
        t = t.cpu()
        rmse = torch.sqrt(mean_squared_error(t[:, 1], t[:, 2]))
        s = score(t[:, 1], t[:, 2])
        self.log('test_rmse', rmse)
        self.log('test_score', s)
        # Example: Save predictions
        predictions = t[:, 1]  # Assuming these are your model predictions
        actuals = t[:, 2]  # Actual values
        df = pd.DataFrame({'Prediction': predictions, 'Actual': actuals})
        df.to_csv('../results/FD003_predictions.csv', index=False)

    def validation_epoch_end(self, val_step_outputs):
        t = torch.stack(val_step_outputs)
        t = torch.sum(t, dim=0)
        self.log('val_rmse', torch.sqrt(t[0] / t[1]), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
        return [optimizer], [lr_scheduler]


def main(args):
    model_kwargs = {
        'sequence_len': args.sequence_len,
        'feature_num': args.feature_num,
        'hidden_dim': args.hidden_dim,
        'lstm_dim': args.lstm_dim,
        'cell_type': args.cell_type,
        'fc_layer_dim': args.fc_layer_dim,
        'rnn_num_layers': args.rnn_num_layers,
        'output_dim': 1,
        'fc_activation': args.fc_activation,
        'attention_type': args.attention_type,
        'bidirectional': args.bidirectional,
        'feature_head_num': args.feature_head_num,
        'sequence_head_num': args.sequence_head_num,
        'fc_dropout': args.fc_dropout
    }
    DAST_model_kwargs = {
        'dim_val': args.hidden_dim,  # General hidden dimension
        'dim_val_s': args.hidden_dim,  # Hidden dimension for the spatial transformer (feature attention)
        'dim_val_t': args.hidden_dim,  # Hidden dimension for the temporal transformer (sequence attention)
        'dim_attn': args.feature_head_num,  # General attention dimension
        'dim_attn_s': args.feature_head_num,  # Number of attention heads for feature attention
        'dim_attn_t': args.sequence_head_num,  # Number of attention heads for sequence attention
        'time_step': args.sequence_len,  # Length of the input sequences
        'input_size': args.feature_num,  # Number of input features
        'dec_seq_len': 10,  # Decoder sequence length (set explicitly)
        'out_seq_len': 1,  # Output sequence length (set explicitly)
        'n_encoder_layers': args.rnn_num_layers,  # Number of RNN layers in the encoder
        'n_decoder_layers': 1,  # Number of RNN layers in the decoder (set explicitly)
        'n_heads': 1,  # Number of attention heads (set explicitly)
        'dropout': args.fc_dropout  # Dropout rate for the fully connected layer
    }

    train_loader, test_loader, valid_loader = CMAPSSDataset.get_data_loaders(
        dataset_root=args.dataset_root,
        sequence_len=args.sequence_len,
        sub_dataset=args.sub_dataset,
        norm_type=args.norm_type,
        max_rul=args.max_rul,
        cluster_operations=args.cluster_operations,
        norm_by_operations=args.norm_by_operations,
        use_max_rul_on_test=args.use_max_rul_on_test,
        validation_rate=args.validation_rate,
        exclude_cols=args.exclude_cols,
        return_id=True,
        use_only_final_on_test=not args.save_attention_weights,
        loader_kwargs={'batch_size': args.batch_size}
    )

    model = Module(
        lr=args.lr,
        # gpus=1
        **model_kwargs
    )
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_rmse',
        min_delta=0.00,
        patience=args.patience,
        verbose=False,
        mode='min'
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_rmse',
        filename='checkpoint-{epoch:02d}-{val_rmse:.4f}',
        save_top_k=1,
        mode='min'
    )
    # checkpoint_callback = CustomModelCheckpoint(
    # monitor='val_rmse',
    # filename='checkpoint-{epoch:02d}-{val_rmse:.4f}',
    # save_top_k=5,
    # mode='min'
    # )

    trainer = pl.Trainer(
        default_root_dir='../checkpoints/{0}/'.format(args.sub_dataset),
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        callbacks=[early_stop_callback, checkpoint_callback],
        # checkpoint_callback=False,
        # logger=False,
        # progress_bar_refresh_rate=0
    )
    trainer.fit(model, train_loader, val_dataloaders=valid_loader or test_loader)
    result = trainer.test(test_dataloaders=test_loader)
    result = result[0]
    log_file_path = "../results/{0}_experiment_results.csv".format(args.sub_dataset)
    log_args_and_metrics_to_csv(log_file_path, args, Exe_Time=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                RMSE=result['test_rmse'], Score=result['test_score'])


class CustomModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_validation_end(self, trainer, pl_module):
        # Get validation RMSE and training RMSE
        val_rmse = trainer.callback_metrics.get("val_rmse", float('inf'))
        train_rmse = trainer.callback_metrics.get("train_rmse", float('inf'))

        # Move tensors to the same device if needed
        if isinstance(val_rmse, torch.Tensor) and isinstance(train_rmse, torch.Tensor):
            if val_rmse.device != train_rmse.device:
                val_rmse = val_rmse.to(train_rmse.device)

        # Check if validation RMSE is less than training RMSE
        if val_rmse < train_rmse:
            # If validation RMSE is less than training RMSE, call the parent method to save the checkpoint
            super().on_validation_end(trainer, pl_module)


if __name__ == '__main__':
    seed = 8
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description='DegFormer')
    parser.add_argument('--Model-spec', type=str, default="DAT-RUL")  # Trans+Itrans>MetaAttentionFS
    parser.add_argument('--sequence-len', type=int, default=40)  # done
    parser.add_argument('--feature-num', type=int, default=16)  # done

    parser.add_argument('--attention-type', default='deg_attention', action='append',
                        help="'deg_attention', 'vanilla_attention'")
    parser.add_argument('--cell-type', type=str, default='slstm', help='lstm, slstm')  # done
    parser.add_argument('--fc-activation', type=str, default='gelu', help='relu, tanh, gelu, silu, leakyrelu')

    parser.add_argument('--rnn-num-layers', type=int, default=1)  # The number of RNN layers.

    parser.add_argument('--hidden-dim', type=int, default=40,
                        help='hidden dims(d_model)')  # The dimensionality of the hidden state (d_model).
    parser.add_argument('--lstm-dim', type=int, default=16, help='lstm hidden dims')
    parser.add_argument('--fc-layer-dim', type=int, default=64)  # The dimensionality of the fully connected layer (d_ff).

    parser.add_argument('--feature-head-num', type=int, default=2)  # done
    parser.add_argument('--sequence-head-num', type=int, default=2)  # done
    parser.add_argument('--fc-dropout', type=float, default=0.25)  # done

    parser.add_argument('--lr', type=float, default=1e-03)  # done
    parser.add_argument('--validation-rate', type=float, default=0.2, help='validation set ratio of train set')
    parser.add_argument('--batch-size', type=int, default=128)  # done
    parser.add_argument('--patience', type=int, default=50, help='Early Stop Patience')  # done
    parser.add_argument('--max-epochs', type=int, default=200)

    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--save-attention-weights', action='store_true', default=False)
    parser.add_argument('--dataset-root', type=str, default='D:\\Datasets\\CMAPSS\\raw_data',
                        help='The dir of CMAPSS dataset')
    parser.add_argument('--sub-dataset', type=str, default='FD003', help='FD001/2/3/4')
    parser.add_argument('--norm-type', type=str, default='z-score', help='z-score, -1-1 or 0-1')  # done
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL upper limit')  # done
    parser.add_argument('--cluster-operations', action='store_true', default=False)
    parser.add_argument('--norm-by-operations', action='store_true', default=False)
    parser.add_argument('--use-max-rul-on-test', action='store_true', default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--exclude_cols', default=['op3', 's1', 's5', 's6', 's10', 's16', 's18', 's19'],
                        action='append', help="cols to exclude")

    args = parser.parse_args()
    main(args)
