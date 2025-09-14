import argparse
import yaml
import os.path as osp
from pathlib import Path
FILE = Path(__file__)
ROOT = FILE.parent.resolve()

from copyright.src.embedder import LoFTR

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/loftr.yaml"))
    return ap.parse_args()

def main():
    args = get_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    cfg['embedding']['output_dir'] = osp.join(cfg['output_dir'], cfg['embedding']['output_dir'])
    
    loftr = LoFTR(output_dir=cfg['output_dir'], 
                  pretrained=cfg['pretrained'],
                  target_height=cfg['target_height'])
    results = loftr.run(cfg['fname1'], cfg['fname2'])
    loftr.visualize()