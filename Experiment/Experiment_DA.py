import sys

sys.path.append("..")
import os
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yaml
from dataset.load_data_CMAPSS import cmapss_data_train_vali_loader,da_cmapss_data_train_vali_loader
from CMAPSS_Related.CMAPSS_Dataset import CMAPSSData

from torch.utils.data import DataLoader

from model.Transfomer_domain_adaptive import Transformer_domain,Discriminator,backboneDiscriminator

from Experiment.Early_Stopping import EarlyStopping
from Experiment.learining_rate_adjust import adjust_learning_rate_class
from Experiment.HTS_Loss_Function import HTSLoss
from Experiment.HTS_Loss_Function import Weighted_MSE_Loss, MSE_Smoothness_Loss

from tool.Write_csv import *
import datetime

from tqdm import tqdm

"""
This file only used for CMAPSS Datase
"""

def advLoss(source, target, device):

    sourceLabel = torch.ones(len(source)).double()
    targetLabel = torch.zeros(len(target)).double()
    Loss = nn.BCELoss()
    if device == 'cuda':
        Loss = Loss.cuda()
        sourceLabel, targetLabel = sourceLabel.cuda(), targetLabel.cuda()
    #print("sd={}\ntd={}".format(source, target))
    loss = Loss(source, sourceLabel) + Loss(target, targetLabel)
    return loss*0.5




class Exp_DA(object):
    def __init__(self, args):
        self.args = args

        self.device = self._acquire_device()

        # load CMAPSS dataset
        if self.args.dataset_name == "CMAPSS":
            self.s_train_data, self.t_train_data, self.s_train_loader, self.t_train_loader, self.s_vali_data, self.t_vali_data, self.s_vali_loader, self.t_vali_loader = self._get_data_CMPASS(flag='train')
            self.test_data, self.test_loader, self.input_feature = self._get_data_CMPASS(flag='test')


        # build the Inception-Attention Model:
        self.model = self._get_model()
        self.D1 = Discriminator(self.args.input_length).double().to(self.device)
        self.D2 = backboneDiscriminator(self.args.input_length,self.args.d_model).double().to(self.device)


        # What optimisers and loss functions can be used by the model
        self.optimizer_dict = {"Adam": optim.Adam}
        self.criterion_dict = {"MSE": nn.MSELoss, "CrossEntropy": nn.CrossEntropyLoss, "WeightMSE": Weighted_MSE_Loss,
                               "smooth_mse": MSE_Smoothness_Loss}

    # choose device
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:0')
            print('Use GPU: cuda:0')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    # ------------------- function to build model -------------------------------------
    def _get_model(self):

        if self.args.model_name == 'Transformer_domain':
            model = Transformer_domain(self.args, input_feature=self.input_feature)


        print("Parameter :", np.sum([para.numel() for para in model.parameters()]))

        return model.double().to(self.device)

    # --------------------------- select optimizer ------------------------------
    def _select_optimizer(self):
        if self.args.optimizer not in self.optimizer_dict.keys():
            raise NotImplementedError

        model_optim = self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # ---------------------------- select criterion --------------------------------
    def _select_criterion(self):
        if self.args.criterion not in self.criterion_dict.keys():
            raise NotImplementedError

        criterion = self.criterion_dict[self.args.criterion]()
        return criterion

    # ------------------------ get Dataloader -------------------------------------
    # 要同时输入source 和 Target 意味着 输出也是要两个
    #  funnction of load CMPASS Dataset
    def _get_data_CMPASS(self, flag="train"):
        args = self.args
        if flag == 'train':
            # train and validation dataset
            s_x_train, t_x_train, s_x_vali, t_x_vali, t_y_train, t_y_vali = da_cmapss_data_train_vali_loader(data_path=args.data_path_CMAPSS,
                                                                                s_id=args.source_domain,
                                                                             t_id=args.target_domain,
                                                                             flag="train",
                                                                             sequence_length=args.input_length,
                                                                             MAXLIFE=args.MAXLIFE_CMAPSS,
                                                                             is_difference=args.is_diff,
                                                                             normalization=args.normalization_CMAPSS,
                                                                             validation=args.validation)


            s_train_data_set = CMAPSSData(s_x_train, t_y_train)
            t_train_data_set = CMAPSSData(t_x_train, t_y_train)

            s_vali_data_set = CMAPSSData(s_x_vali, t_y_vali)
            t_vali_data_set = CMAPSSData(t_x_vali, t_y_vali)


            s_train_data_loader = DataLoader(dataset=s_train_data_set,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           drop_last=True)

            t_train_data_loader = DataLoader(dataset=t_train_data_set,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           drop_last=True)

            s_vali_data_loader = DataLoader(dataset=s_vali_data_set,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          drop_last=True)

            t_vali_data_loader = DataLoader(dataset=t_vali_data_set,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          drop_last=True)

            return s_train_data_set, t_train_data_set, s_train_data_loader, t_train_data_loader, s_vali_data_set, t_vali_data_set, s_vali_data_loader, t_vali_data_loader

        else:
            # test dataset
            X_test, y_test = cmapss_data_train_vali_loader(data_path=args.data_path_CMAPSS,
                                                           Data_id=args.target_domain,
                                                           flag="test",
                                                           sequence_length=args.input_length,
                                                           MAXLIFE=args.MAXLIFE_CMAPSS,
                                                           is_difference=args.is_diff,
                                                           normalization=args.normalization_CMAPSS,
                                                           validation=args.validation)
            input_fea = X_test.shape[-1]
            test_data_set = CMAPSSData(X_test, y_test)
            test_data_loader = DataLoader(dataset=test_data_set,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          drop_last=False)

            return test_data_set, test_data_loader, input_fea

    def save_hparam(self, args, path):
        # args: args from argparse return
        value2save = {k: v for k, v in vars(args).items() if not k.startswith('__') and not k.endswith('__')}
        with open(os.path.join(path, 'hparam.yaml'), 'a+') as f:
            f.write(yaml.dump(value2save))

    def train(self, save_path):

        # save address
        path = './logs/' + save_path
        if not os.path.exists(path):
            os.makedirs(path)

        model_path = path + '/' + self.args.model_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # how many step of train and validation:
        train_steps = len(self.t_train_loader)
        vali_steps = len(self.t_vali_loader)
        print("train_steps: ", train_steps)
        print("validaion_steps: ", vali_steps)

        # TODO 保存模型的超参数
        self.save_hparam(args=self.args, path=model_path)

        # initial early stopping
        early_stopping = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)

        # initial learning rate
        learning_rate_adapter = adjust_learning_rate_class(self.args, True)
        # choose optimizer
        model_optim = self._select_optimizer()

        # choose loss function
        loss_criterion = torch.nn.MSELoss()

        # training process
        print("start training")
        for epoch in range(self.args.train_epochs):
            start_time = time()

            iter_count = 0
            train_loss = []

            self.model.train()
            s_iter = iter(self.s_train_loader)
            t_iter = iter(self.t_train_loader)
            l = min(len(s_iter), len(t_iter))
            for _ in range(l):
                s_d, batch_y = next(s_iter)
                t_d, _ = next(t_iter)
                model_optim.zero_grad()
                s_d = s_d.double().to(self.device)  # [B,window_size,D]
                t_d = t_d.double().to(self.device)  # [B,window_size]
                batch_y = batch_y.double().to(self.device)
                s_features, s_out = self.model(s_d)
                t_features, t_out = self.model(t_d)

                if self.args.is_minmax:   # 训练的时候要拿source的output去和source的GT去比，因为认为场景下target是得不到GT的
                    # 只预测窗口内的最后一个rul
                    batch_y_norm = batch_y / 120
                    loss1 = loss_criterion(s_out, batch_y_norm)

                else:
                    loss1 = loss_criterion(s_out, batch_y)

                if self.args.type == 1 or self.args.type == 0:
                    if self.args.type == 1:
                        s_domain = self.D2(s_features)
                        t_domain = self.D2(t_features)
                    else:
                        s_domain = self.D1(s_out)
                        t_domain = self.D1(t_out)
                    # loss2 = advLoss(s_domain.squeeze(1), t_domain.squeeze(1), 'cuda')
                    # loss = loss1 + 0.1*loss2

                    #Block all classifiers
                    loss = loss1

                elif self.args.type == 2:
                    s_domain_bkb = self.D2(s_features)
                    t_domain_bkb = self.D2(t_features)
                    s_domain_out = self.D1(s_out)
                    t_domain_out = self.D1(t_out)
                    if epoch >= 5:  # 迭代初期不引入混合loss
                        fea_loss = advLoss(s_domain_bkb.squeeze(1), t_domain_bkb.squeeze(1), 'cuda')
                        out_loss = advLoss(s_domain_out.squeeze(1), t_domain_out.squeeze(1), 'cuda')

                        loss = loss1 + 0.1 * fea_loss + 0.5 * out_loss
                    else:
                        loss = loss1

                loss.backward()
                model_optim.step()

            end_time = time()
            epoch_time = end_time - start_time
            train_loss = np.average(train_loss)  # avgerage loss

            # validation process:
            vali_loss = self.validation(self.s_vali_loader, self.t_vali_loader, loss_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}. it takse {4:.7f} seconds".format(
                epoch + 1, train_steps, train_loss, vali_loss, epoch_time))

            # At the end of each epoch, Determine if we need to stop and adjust the learning rate

            early_stopping(vali_loss, self.model, model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            learning_rate_adapter(model_optim, vali_loss)

        # 读取最优的参数
        check_point = torch.load(model_path + '/' + 'best_checkpoint.pth')
        self.model.load_state_dict(check_point)

        # test:
        if self.args.dataset_name == "CMAPSS":
            average_enc_loss, average_enc_overall_loss, overall_score = self.test(self.test_loader)
            print("CMAPSS: RMSE test performace of enc is: ", average_enc_loss, " of enc overall is: ",
                  average_enc_overall_loss, 'socre of'
                                            'enc', overall_score)

        log_path = path + '/da_experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                           'batch_size','best_last_RMSE', 'score','windowsize', 'source_d',
                           'target_d','type', 'info']]
            write_csv(log_path, table_head, 'w+')

        time_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  # 获取当前系统时间

        a_log = [{'dataset': save_path, 'model': self.args.model_name, 'time': time_now,
                  'LR': self.args.learning_rate,
                  'batch_size': self.args.batch_size,
                  'best_last_RMSE': average_enc_loss,
                  'score': overall_score, 'windowsize': self.args.input_length, 'source_d': self.args.source_domain,
                  'target_d': self.args.target_domain,
                  'type': self.args.type,
                  'info': self.args.info}]
        write_csv_dict(log_path, a_log, 'a+')

    # ---------------------------------- validation function -----------------------------------------
    def validation(self, s_vali_loader, t_vali_loader, criterion):
        self.model.eval()
        total_loss = []

        s_iter = iter(self.s_vali_loader)
        t_iter = iter(self.t_vali_loader)
        l = min(len(s_iter), len(t_iter))
        for _ in range(l):
            s_d, _ = next(s_iter)
            t_d, batch_y = next(t_iter)

            s_d = s_d.double().to(self.device)  # [B,window_size,D]
            t_d = t_d.double().to(self.device)  # [B,window_size]
            batch_y = batch_y.double().to(self.device)

            s_features, s_out = self.model(s_d)
            t_features, t_out = self.model(t_d)

            if self.args.is_minmax:
                outputs_denorm = t_out * 120
                loss = criterion(outputs_denorm[:, -1:], batch_y[:, -1:])
            else:
                loss = criterion(t_out, batch_y)

            total_loss.append(loss.item())

        average_vali_loss = np.average(total_loss)

        self.model.train()
        return average_vali_loss

    # ----------------------------------- test function ------------------------------------------
    def test(self, test_loader):
        self.model.eval()
        enc_pred = []
        gt = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.double().double().to(self.device)
            batch_y = batch_y.double().double().to(self.device)

            if self.args.is_minmax:
                t_features, t_out = self.model(batch_x)
                outputs = t_out * 120

            else:
                t_features, t_out = self.model(batch_x)  # outputs[B,window_size]
                outputs = t_out

            batch_y = batch_y[:, -1:].detach().cpu().numpy()
            enc = outputs[:, -1:].detach().cpu().numpy()

            gt.append(batch_y)
            enc_pred.append(enc)

        gt = np.concatenate(gt).reshape(-1, 1)
        enc_pred = np.concatenate(enc_pred).reshape(-1, 1)

        if self.args.dataset_name == "CMAPSS":
            # 算的就是RMSE
            average_enc_loss = np.sqrt(mean_squared_error(enc_pred, gt))
            # average_enc_loss = np.sqrt(mean_squared_error(enc_pred[:, -1], gt[:, -1]))  #取wz最后一个位置的预测结果 #这个的输出应该就是最真实的RMSE
            average_enc_overall_loss = np.sqrt(mean_squared_error(enc_pred, gt))

            # 计算score
            overall_score = self.score_compute(enc_pred, gt)

            return average_enc_loss, average_enc_overall_loss, overall_score

    def score_compute(self, pred, gt):
        # pred [B] gt[B]
        B = pred.shape
        score = 0
        score_list = np.where(pred - gt < 0, np.exp(-(pred - gt) / 13) - 1, np.exp((pred - gt) / 10) - 1)

        # 这里有的paper求均值，有的求和。实验里面先全都求和计算score
        score = np.sum(score_list)

        return score




