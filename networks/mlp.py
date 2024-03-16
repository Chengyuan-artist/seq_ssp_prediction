import torch
import torch.nn as nn
# import torch.nn.functional as F

class MLP(nn.Module):
  def __init__(self, feat_dim=3):
    self.encoder = nn.Sequential(
        nn.Linear(dim_in, dim_in),
        nn.ReLU(inplace=True),
        nn.Linear(dim_in, feat_dim)
    )
  def forward(self, x):
    feat = self.encoder(x)
    # feat = F.normalize(self.head(feat), dim=1)
    return feat