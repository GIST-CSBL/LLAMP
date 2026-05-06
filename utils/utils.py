import os
import random

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import kendalltau

import numpy as np
import torch
import torch.nn as nn
from Bio import SeqIO

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
       
def count_mer(mer, seq):
    mer_count = {}
    for oligo in mer:
        mer_count[oligo] = seq.count(oligo)
    return mer_count    

def normalize(x):
    norm = np.linalg.norm(x)
    return x/norm

def get_features(seqs):
    features1 = 0
    features2 = 0
    features3 = 0
    features4 = 0
    for seq in seqs:
        nucleotide = ['A', 'T', 'G', 'C']
        mer_1 = nucleotide
        mer_2 = []
        for i in nucleotide:
            for j in nucleotide:
                mer_2.append(i+j)
        mer_3 = []
        for i in nucleotide:
            for j in nucleotide:
                for k in nucleotide:
                    mer_3.append(i+j+k)
        mer_4 = []
        for i in nucleotide:
            for j in nucleotide:
                for k in nucleotide:
                    for l in nucleotide:
                        mer_4.append(i+j+k+l)

        count_1 = count_mer(mer_1, seq)
        count_2 = count_mer(mer_2, seq)
        count_3 = count_mer(mer_3, seq)
        count_4 = count_mer(mer_4, seq)

        features1 += np.array(list(count_1.values()))
        features2 += np.array(list(count_2.values()))
        features3 += np.array(list(count_3.values()))
        features4 += np.array(list(count_4.values()))
    features1 = normalize(features1)
    features2 = normalize(features2)
    features3 = normalize(features3)
    features4 = normalize(features4)
    
    return np.concatenate((features1, features2, features3, features4))

def get_genome_feature_from_file(file_path):
    genome_seqs = []
    descriptions = []
    for seq_record in SeqIO.parse(file_path, 'fasta'):
        descriptions.append(seq_record.description)
        genome_seqs.append(seq_record.seq)

    return get_features(genome_seqs)
