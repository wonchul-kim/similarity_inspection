import torch
from .fusion_mlp import FusionMLP

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
