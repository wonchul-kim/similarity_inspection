import torch, torch.nn as nn, torch.nn.functional as F
import os

class FusionMLP(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def fuse_signals(feats, mlp: FusionMLP=None):
    """
    feats: dict with keys ['global_cos', 'local_ratio', 'ocr_sim']
    returns score in [0,1] via sigmoid
    """
    x = torch.tensor([[feats.get("global_cos",0.0),
                       feats.get("local_ratio",0.0),
                       feats.get("ocr_sim",0.0)]], dtype=torch.float32)
    if mlp is None:
        mlp = FusionMLP()
    with torch.no_grad():
        logits = mlp(x)
    return torch.sigmoid(logits).item()
