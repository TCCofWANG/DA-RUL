import numpy as np
from torch import nn
import torch


def score(predict, label):
    a1 = 13
    a2 = 10
    error = predict - label
    pos_e = np.exp(-error[error < 0] / a1) - 1
    neg_e = np.exp(error[error >= 0] / a2) - 1
    return sum(pos_e) + sum(neg_e)


def pn_rul_compute(predictor, f_pos, f_neg):
    """
    Used to compute Rul of the positive and negative samples. Because the Weighted Info
    NCE LOSS needs all the positive and negative rul to compute the final loss.

    :param predictor: The predictor layer
    :param f_pos: The positive samples with shape (batch, features)
    :param f_neg: The negative samples with shape (batch, nums, features), where nums indicates
                  the number of negative samples.
    :return: All the rul with shape (batch, nums+1)
    """
    out_all = predictor(f_pos)
    neg_nums = f_neg.shape[1]
    neg_out = []
    for neg_i in range(neg_nums):
        neg_out.append(predictor(f_neg[:, neg_i]))
    neg_out = torch.concat(neg_out, dim=-1)
    return torch.concat([out_all, neg_out], dim=-1)