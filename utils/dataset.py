import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader

from transformers import EsmTokenizer

tokenizer = EsmTokenizer.from_pretrained('Daehun/peptide_tuned_ESM-2')

def sequence_to_input(seqs):    
    input_seqs = []
    for i in seqs:
        for j in range(len(i)-1):
            i = i[:j+j+1]+ ' ' + i[j+j+1:]
        input_seqs.append(i)
    
    inputs = tokenizer.batch_encode_plus(input_seqs, add_special_tokens=True, pad_to_max_length=True)
    input_ids = torch.tensor(inputs['input_ids'])
    attention_mask = torch.tensor(inputs['attention_mask'])
    
    return input_ids, attention_mask

def data_loader(seqs, genome_feats, label, BATCH_SIZE, NUM_THREADS=0, shuffle=False):
    input_ids, attention_mask = sequence_to_input(seqs)
    dataset = TensorDataset(input_ids, attention_mask, genome_feats, label)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=NUM_THREADS, shuffle=shuffle)
    
    return loader

def get_features(data, feature_dic):
    encoded = []
    for token in data:
        encoded.append(torch.as_tensor(feature_dic[token][0], dtype=torch.float32))
    return torch.stack(encoded)