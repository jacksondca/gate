import torch
import torch.nn.functional as F
from torch import nn


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, embedding, pos, neg):
        pos_sim = torch.sum(embedding * pos, dim=-1)
        neg_sim = torch.sum(embedding * neg, dim=-1)
        
        pos_sim = pos_sim / self.temperature
        neg_sim = neg_sim / self.temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(embedding.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss