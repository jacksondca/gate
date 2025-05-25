import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cluster.model.train.fusion import TransformerFusion


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_layers=2, num_heads=4):
        super(Model, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        
        self.fusion = TransformerFusion(output_dim, input_dim, hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, node_feature, graph_feature):
        x = self.fusion(node_feature, graph_feature)
        return self.encoder(x)
