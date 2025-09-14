import time
import argparse, os, json, numpy as np, tqdm, cv2, yaml, math
from copyright.src import (VisualEmbedder, EmbeddingIndex)
from copyright.utils.functionals import load_image_bgr, save_debug_matches

from pathlib import Path
FILE = Path(__file__)
ROOT = FILE.parent.resolve()

def cosine(a,b): return float((a*b).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resellers", default=str(ROOT / '../assets/reseller'))
    ap.add_argument("--index", default=str(ROOT / '../outputs/index/originals.faiss'), help="FAISS index path")
    ap.add_argument("--catalog", default=str(ROOT / '../outputs/index/originals.jsonl'), help="JSONL meta for originals")
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--out", default=str(ROOT / '../outputs/scan'), help="Output JSONL")
    ap.add_argument("--save-viz", action="store_true", default='True', help="Save LoFTR viz images")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    
    
    ve = VisualEmbedder(name=cfg["embedding"]["name"],
                        device=cfg["device"],
                        input_size=cfg["embedding"]["input_size"],
                        normalize=cfg["faiss"]["normalize"])

    idx = EmbeddingIndex.load(args.index, args.catalog, normalize=cfg["faiss"]["normalize"])
    idx.index.nprobe = cfg["faiss"]["faiss_nprobe"]

    res_paths = []
    for root,_,files in os.walk(args.resellers):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp")):
                res_paths.append(os.path.join(root,f))
    res_paths.sort()

    with open(os.path.join(args.out, 'scan.jsonl'), "w", encoding="utf-8") as fout:
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
