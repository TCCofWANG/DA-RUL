import re
import argparse
import random
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import mean_squared_error

import sys
import os
# Get the parent directory of the current directory and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from dataset import CMAPSSDataset
from Experiment import DAT_Experiment
from utils.metrics import score

class Module(pl.LightningModule):
    def __init__(self, lr, **kwargs):
        super(Module, self).__init__()
        self.net = DAT_Experiment(**kwargs)
        self.lr = lr

    def forward(self, x):
        return self.net(x)


    def test_step(self, batch, batch_idx, reduction='sum'):
        x, y, id = batch
        x = self.net(x)
        return torch.cat([id, torch.floor(x), y], dim=1)

    def test_epoch_end(self, step_outputs):
        t = torch.cat(step_outputs, dim=0)
        t = t.cpu()
        rmse = torch.sqrt(mean_squared_error(t[:, 1], t[:, 2]))
        s = score(t[:, 1], t[:, 2])
        self.log('test_rmse', rmse)
        self.log('test_score', s)

    def validation_epoch_end(self, val_step_outputs):
        t = torch.stack(val_step_outputs)
        t = torch.sum(t, dim=0)
        self.log('val_rmse', torch.sqrt(t[0] / t[1]), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
        return [optimizer], [lr_scheduler]

def load_model(checkpoint_path, model_kwargs, lr):
    model = Module(lr=lr, **model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def main():
    seed = 8
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser( description=__doc__)
    parser.add_argument('--Model-spec', type=str, default="DAT+sLSTM")
    parser.add_argument('--sequence-len', type=int, default=30, help='(30|40) sequence length)')
    parser.add_argument('--feature-num', type=int, default=16)

    parser.add_argument('--attention-type', default='deg_attention', action='append', help="'deg_attention', 'vanilla_attention'")
    parser.add_argument('--cell-type', type=str, default='slstm', help='lstm, slstm')
    parser.add_argument('--fc-activation', type=str, default='gelu', help='relu, gelu, silu')

    parser.add_argument('--rnn-num-layers', type=int, default=1)

    parser.add_argument('--hidden-dim', type=int, default=32, help='(32|40)hidden dims(d_model)')
    parser.add_argument('--lstm-dim', type=int, default=16, help='lstm hidden dims')
    parser.add_argument('--fc-layer-dim', type=int,default=32, help='(32|64)')

    parser.add_argument('--feature-head-num', type=int, default=2)
    parser.add_argument('--sequence-head-num', type=int, default=2)
    parser.add_argument('--fc-dropout', type=float, default=0.25)

    parser.add_argument('--lr', type=float, default=1e-03)
    parser.add_argument('--validation-rate', type=float, default=0.2, help='validation set ratio of train set')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=50, help='Early Stop Patience')
    parser.add_argument('--max-epochs', type=int, default=150)

    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--save-attention-weights', action='store_true', default=False)
    parser.add_argument('--dataset-root', type=str, default='D:\\Datasets\\CMAPSS\\raw_data',  help='The dir of CMAPSS dataset')
    parser.add_argument('--sub-dataset', type=str, default='FD001', help='FD001/3')
    parser.add_argument('--norm-type', type=str, default='z-score', help='z-score, -1-1 or 0-1')
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL upper limit')
    parser.add_argument('--cluster-operations', action='store_true', default=False)
    parser.add_argument('--norm-by-operations', action='store_true', default=False)
    parser.add_argument('--use-max-rul-on-test', action='store_true', default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--exclude_cols', default=['op3', 's1', 's5', 's6', 's10', 's16', 's18', 's19'], action='append', help="cols to exclude")

    args = parser.parse_args()

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

    # Load test data
    _, test_loader, _ = CMAPSSDataset.get_data_loaders(
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

    # Directory where the checkpoints are stored
    subset = 'FD001'
    try:
        lightning_logs_dir = os.path.join('./checkpoints/{0}/lightning_logs/'.format(subset))
        os.listdir(lightning_logs_dir)
    except:
        lightning_logs_dir = os.path.join('../checkpoints/{0}/lightning_logs/'.format(subset))
    # Initialize a dictionary to store results for each checkpoint
    results = {}
    i=1
    for folder_version in os.listdir(lightning_logs_dir):
        checkpoint_dir = '{0}/{1}/checkpoints/'.format(lightning_logs_dir, folder_version)


        # Iterate over files in the checkpoint directory
        for filename in os.listdir(checkpoint_dir):
            # Check if the file is a checkpoint file
            if filename.endswith('.ckpt'):
                # Extract checkpoint information using regular expression
                match = re.match(r'checkpoint-epoch=(\d+)-val_rmse=(\d+\.\d+)', filename)
                if match:
                    epoch = int(match.group(1))
                    val_rmse = float(match.group(2))
                    # Construct full path to checkpoint file
                    checkpoint_path = os.path.join(checkpoint_dir, filename)

                    # Load the model
                    model = load_model(checkpoint_path, model_kwargs, lr=args.lr)

                    # Initialize the Lightning Trainer
                    trainer = pl.Trainer(gpus=args.gpus)

                    # Perform testing
                    result = trainer.test(model, test_dataloaders=test_loader)

                    # Store the result in the dictionary
                    results["checkpoint_"+folder_version] = result


    # Print results
    for filename, result in results.items():
        print(f'Result for {filename}: {result}')

if __name__ == '__main__':
    main()
