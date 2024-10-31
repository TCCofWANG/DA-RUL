import argparse
import random
import time

import numpy as np
import torch
import pytorch_lightning as pl
from dataset import ModifiedCMAPSSDataset, CMAPSSDataset
from model import MultiHeadAttentionLSTM
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import mean_squared_error

from Experiment.DAT_Experiment import DAT_Experiment
from utils.log_result import log_experiment_results, log_args_and_metrics_to_csv
from utils.metrics import score

from itertools import product


class Module(pl.LightningModule):
    def __init__(self, lr, **kwargs):
        super(Module, self).__init__()
        self.net = DAT_Experiment(**kwargs)
        self.lr = lr
        # print(self.net)
        # print('lr', self.lr)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        # print('x', x)
        # print('y', y)
        x = self.net(x)
        loss = F.mse_loss(x, y)
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
        return torch.cat([id, x, y], dim=1)

    def test_epoch_end(self, step_outputs):
        t = torch.cat(step_outputs, dim=0)
        t = t.cpu()
        rmse = torch.sqrt(mean_squared_error(t[:, 1], t[:, 2]))
        s = score(t[:, 1], t[:, 2])
        self.log('test_rmse', rmse)
        self.log('test_score', s)
        # Example: Save predictions
        # predictions = t[:, 1]  # predictions values
        # actuals = t[:, 2]  # Actual values

    def validation_epoch_end(self, val_step_outputs):
        t = torch.stack(val_step_outputs)
        t = torch.sum(t, dim=0)
        self.log('val_rmse', torch.sqrt(t[0] / t[1]), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
        return [optimizer], [lr_scheduler]


def train_model_with_hyperparameters(args, model_kwargs):
    seeds = [0, 4, 8, 12, 16,20, 24,28, 32,36, 40, 44,48]
    seed = random.choice(seeds)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
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

    model = Module(**model_kwargs)
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
        mode='min',
    )

    trainer = pl.Trainer(
        default_root_dir='../checkpoints/param_test/{0}/'.format(args.sub_dataset),
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_dataloaders=valid_loader or test_loader)
    result = trainer.test(test_dataloaders=test_loader)

    return result[0]  # trainer.callback_metrics['val_rmse'].item()


def main(args):
    # Define hyperparameters grid
    hyperparameters_grid = {
        #'lr': [1e-04, 4e-04, 7e-04, 1e-03, 4e-03, 7e-03, 1e-02],
        #'sequence_len': [20, 25, 30, 35, 40, 45, 50],
        #'lr': [1e-03, 1e-03, 1e-03, 1e-03, 1e-03]
        'attention_type': ['vanilla_attention', 'deg_attention'],
        'cell_type': ['lstm', 'slstm'],
        'lr': [1e-03, 1e-03, 1e-03]
    }

    best_performance = float('inf')
    best_hyperparameters = None
    log_file_path = "../results/{1}_{0}_experiment_results.csv".format(str(time.time()), args.sub_dataset)

    for combination in product(*hyperparameters_grid.values()):
        hyperparameters = dict(zip(hyperparameters_grid.keys(), combination))
        print(f"Testing combination: {hyperparameters}")

        # Update args and model_kwargs with the current combination of hyperparameters
        for key, value in hyperparameters.items():
            setattr(args, key, value)

        model_kwargs = {
            'sequence_len': args.sequence_len,
            'feature_num': args.feature_num,
            'hidden_dim': args.hidden_dim,#(args.sequence_len + 7) // 8 * 8,  # rounding sequence_len up to the nearest multiple of 8
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
            'fc_dropout': args.fc_dropout,
            'lr': args.lr  # Assuming the learning rate is directly used in Module's initialization
        }
        args.hidden_dim = model_kwargs['hidden_dim']
        performance = train_model_with_hyperparameters(args, model_kwargs)
        log_args_and_metrics_to_csv(log_file_path, args, RMSE=performance['test_rmse'], Score=performance['test_score'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DegFormer+sLSTM')

    parser.add_argument('--Model-spec', type=str, default="DAT-RUL")  # done AOA + Sequence-> Feature
    parser.add_argument('--sequence-len', type=int, default=40)  # done
    parser.add_argument('--feature-num', type=int, default=20)  # done

    parser.add_argument('--attention-type', default='deg_attention', action='append', help="'deg_attention', 'vanilla_attention'")
    parser.add_argument('--cell-type', type=str, default='slstm', help='lstm, mlstm, slstm, gru or rnn')  # done
    parser.add_argument('--fc-activation', type=str, default='gelu', help='relu, tanh, gelu, silu, leakyrelu')

    parser.add_argument('--rnn-num-layers', type=int, default=1)  # The number of RNN layers (d_hidden layers).

    parser.add_argument('--hidden-dim', type=int, default=40,
                        help='hidden dims(d_model)')  # The dimensionality of the hidden state (d_model).
    parser.add_argument('--lstm-dim', type=int, default=16, help='lstm hidden dims')
    parser.add_argument('--fc-layer-dim', type=int,
                        default=32)  # The dimensionality of the fully connected layer (d_ff).

    # parser.add_argument('--hidden-dim', type=int, default=40, help='hidden dims(d_model)')  # The dimensionality of the hidden state (d_model).
    # parser.add_argument('--lstm-dim', type=int, default=16, help='lstm hidden dims')
    # parser.add_argument('--fc-layer-dim', type=int, default=64)  # The dimensionality of the fully connected layer (d_ff).

    parser.add_argument('--feature-head-num', type=int, default=2)  # done
    parser.add_argument('--sequence-head-num', type=int, default=2)
    parser.add_argument('--fc-dropout', type=float, default=0.25)  # done

    parser.add_argument('--lr', type=float, default=1e-03)  # done
    parser.add_argument('--validation-rate', type=float, default=0.2, help='validation set ratio of train set')
    parser.add_argument('--batch-size', type=int, default=128)  # done
    parser.add_argument('--patience', type=int, default=50, help='Early Stop Patience')  # done
    parser.add_argument('--max-epochs', type=int, default=200)


    parser.add_argument('--save-attention-weights', action='store_true', default=False)
    parser.add_argument('--dataset-root', type=str, default='D:\\Datasets\\CMAPSS\\raw_data',
                        help='The dir of CMAPSS dataset')
    parser.add_argument('--sub-dataset', type=str, default='FD002', help='FD001/2/3/4')
    parser.add_argument('--norm-type', type=str, default='z-score', help='z-score, -1-1 or 0-1')  # done
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL')  # done
    parser.add_argument('--cluster-operations', action='store_true', default=True)
    parser.add_argument('--norm-by-operations', action='store_true', default=True)
    parser.add_argument('--use-max-rul-on-test', action='store_true', default=True)
    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--gpus', type=int, default=1)
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--exclude_cols', default=['s13', 's16', 's18', 's19'], action='append', help="cols to exclude")

    args = parser.parse_args()
    main(args)
