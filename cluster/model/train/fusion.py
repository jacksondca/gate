import torch
import torch.nn as nn

class TransformerFusion(nn.Module):
    def __init__(self, in_channels1, in_channels, out_channels, num_heads=8, num_layers=2, dim_feedforward=2048, dropout=0.0):
        super(TransformerFusion, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
        self.fc1 = nn.Linear(in_channels1, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.fc1(x1)

        x = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)
        
        x = self.transformer_encoder(x)
        
        fused_features = x[:, -1, :]
        
        fused_features = self.fc(fused_features)
        
        return fused_features
