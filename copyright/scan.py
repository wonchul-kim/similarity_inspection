import time
import os.path as  osp
import argparse, os, json, numpy as np, tqdm, cv2, yaml, math
from copyright.src import (VisualEmbedder, EmbeddingIndex)
from copyright.utils.functionals import load_image_bgr, save_debug_matches

from pathlib import Path
FILE = Path(__file__)
ROOT = FILE.parent.resolve()

def cosine(a,b): return float((a*b).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    cfg["faiss"]["output_dir"] = osp.join(cfg["output_dir"], cfg["faiss"]["output_dir"])
    os.makedirs(cfg["faiss"]["output_dir"], exist_ok=True)
    
    ve = VisualEmbedder(name=cfg["embedding"]["name"],
                        device=cfg["device"],
                        input_size=cfg["embedding"]["input_size"],
                        normalize=cfg["faiss"]["normalize"])

    idx = EmbeddingIndex.load(cfg['faiss']['index_file'], 
                              cfg['faiss']['catalog_file'], 
                              normalize=cfg["faiss"]["normalize"])
    idx.index.nprobe = cfg["faiss"]["faiss_nprobe"]

    res_paths = []
    for root,_,files in os.walk(cfg['reseller_image_dir']):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp")):
                res_paths.append(os.path.join(root,f))
    res_paths.sort()

    with open(os.path.join(cfg["faiss"]["output_dir"], 'scan.jsonl'), "w", encoding="utf-8") as fout:
        for p in tqdm.tqdm(res_paths, desc="Scanning"):
            img_r = load_image_bgr(p)
            e_r = ve.embed_np_bgr(img_r).astype("float32")[None,:]
            D, I, metas = idx.search(e_r, topk=cfg["faiss"]["topk"])
            # for topk candidates, compute signals and fuse
            entries = []
            for rank, (d,i,m) in enumerate(zip(D[0], I[0], metas[0])):
                cand_path = m["path"]
                img_o = load_image_bgr(cand_path)
                e_o = ve.embed_np_bgr(img_o)
                gcos = cosine(e_r[0], e_o)

                feats = {"global_cos": gcos}
                entry = {
                    "reseller": p,
                    "original": cand_path,
                    "rank": rank,
                    "faiss_score": float(d),
                    "signals": feats,
                    "score": float(gcos)
                }

                entries.append(entry)

            entries.sort(key=lambda x: x["score"], reverse=True)
            best = entries[0]
            best["decision"] = "copied" if best["score"]>=cfg["thresholds"]["copied"] else "uncertain"
            fout.write(json.dumps(best, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
