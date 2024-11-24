import os
import random

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import kendalltau

import numpy as np
import torch
import torch.nn as nn

def set_random_seed(random_seed = 42):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # multi-GPU
    

def compute_metrics(preds, labels):
    R_squre = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    pearson = float(np.corrcoef(labels, preds)[1,0])
    # tau, pval = kendalltau(labels, preds)
    
    return {
        'R_squre': R_squre,
        'mae' : mae,
        'mse' : mse,
        'rmse' : rmse,
        'Pearson Q' : pearson,
        # 'Kendall tau' : tau,
    }