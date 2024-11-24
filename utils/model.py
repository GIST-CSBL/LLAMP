import torch
import torch.nn as nn

from transformers import EsmModel

class LLAMP(nn.Module):
    def __init__(self, hidden_feat = 256, pooling = None, pretrained_model='Daehun/peptide_tuned_ESM-2'):
        super(LLAMP, self).__init__()
        self.bert = EsmModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        
        self.genome_linear = nn.Sequential(nn.Linear(340, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 128))
        
        self.peptide_linear = nn.Sequential(nn.Linear(480, 256))
        
        if hidden_feat == None:
            self.linear = nn.Sequential(nn.Linear(384, 1))
        else:
            self.linear = nn.Sequential(nn.Linear(384, hidden_feat),
                                        nn.ReLU(),
                                        nn.Linear(hidden_feat, hidden_feat),
                                        nn.ReLU(),
                                        nn.Linear(hidden_feat, 1))
        
    def forward(self, input_ids, attention_mask, genome_feat):
        embedding = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        if self.pooling == 'mean':
            hidden = torch.sum((embedding * attention_mask.unsqueeze(-1)), dim =1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
        elif self.pooling == 'sum': #'sum'
            hidden = torch.sum((embedding * attention_mask.unsqueeze(-1)), dim =1)
        elif self.pooling == 'CLS':
            hidden = embedding[:, 0, :]
        
        hidden = self.peptide_linear(hidden)
        genome_feat = self.genome_linear(genome_feat)
            
        in_feats = torch.cat([hidden, genome_feat], dim=1)
        predict = self.linear(in_feats)
        return predict.squeeze()