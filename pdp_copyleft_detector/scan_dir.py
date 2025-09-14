import time
import argparse, os, json, numpy as np, tqdm, cv2, yaml, math
from copyleft_detector import (VisualEmbedder, EmbeddingIndex, 
                               LoFTRMatcher, 
                               FusionMLP, fuse_signals)
# , OCRTextEncoder,  
from copyleft_detector.utils import load_image_bgr, save_debug_matches

from pathlib import Path
FILE = Path(__file__)
ROOT = FILE.parent.resolve()

def cosine(a,b): return float((a*b).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resellers", default=str(ROOT / '../assets/reseller'))
    ap.add_argument("--index", default=str(ROOT / '../outputs/pdp/index/originals.faiss'), help="FAISS index path")
    ap.add_argument("--catalog", default=str(ROOT / '../outputs/pdp/index/originals.jsonl'), help="JSONL meta for originals")
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--out", default=str(ROOT / '../outputs/pdp'), help="Output JSONL")
    ap.add_argument("--save-viz", action="store_true", default='True', help="Save LoFTR viz images")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ve = VisualEmbedder(name=cfg["visual_backbone"]["name"],
                        device=cfg["device"],
                        input_size=cfg["visual_backbone"]["input_size"],
                        normalize=cfg["retrieval"]["normalize"])

    idx = EmbeddingIndex.load(args.index, args.catalog, normalize=cfg["retrieval"]["normalize"])
    idx.index.nprobe = cfg["retrieval"]["faiss_nprobe"]
    
    matcher = None
    # matcher = LoFTRMatcher(device=cfg["device"]) if cfg["local_match"]["enable"] else None
    
    ocr = None
    # ocr = OCRTextEncoder(langs=cfg["ocr"]["lang"], sentence_model=cfg["ocr"]["sentence_model"], device=cfg["device"]) if cfg["ocr"]["enable"] else None

    mlp = None
    if os.path.exists(cfg["fusion"]["checkpoint"]):
        import torch
        mlp = FusionMLP(hidden=cfg["fusion"]["hidden"])
        mlp.load_state_dict(torch.load(cfg["fusion"]["checkpoint"], map_location="cpu"))
        mlp.eval()

    res_paths = []
    for root,_,files in os.walk(args.resellers):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp")):
                res_paths.append(os.path.join(root,f))
    res_paths.sort()

    os.makedirs("results/viz", exist_ok=True)

    with open(os.path.join(args.out, 'scan.json'), "w", encoding="utf-8") as fout:
        for p in tqdm.tqdm(res_paths, desc="Scanning"):
            img_r = load_image_bgr(p)
            e_r = ve.embed_np_bgr(img_r).astype("float32")[None,:]
            D, I, metas = idx.search(e_r, topk=cfg["retrieval"]["topk"])
            # for topk candidates, compute signals and fuse
            entries = []
            for rank, (d,i,m) in enumerate(zip(D[0], I[0], metas[0])):
                cand_path = m["path"]
                img_o = load_image_bgr(cand_path)
                e_o = ve.embed_np_bgr(img_o)
                gcos = cosine(e_r[0], e_o)

                lratio, mm = (0.0, (None,None,None))
                if matcher is not None:
                    lratio, mm = matcher.match_ratio(img_o, img_r, max_size=cfg["local_match"]["max_size"])

                osim, texts = (0.0, ("",""))
                if ocr is not None:
                    osim, texts = ocr.semantic_sim(img_o, img_r, max_words=cfg["ocr"]["max_words"])

                feats = {"global_cos": gcos, "local_ratio": lratio, "ocr_sim": osim}
                score = fuse_signals(feats, mlp)

                entry = {
                    "reseller": p,
                    "original": cand_path,
                    "rank": rank,
                    "faiss_score": float(d),
                    "signals": feats,
                    "score": float(score)
                }
                if args.save_viz and mm[0] is not None:
                    k0, k1, conf, (sx0, sy0), (sx1, sy1) = mm
                    # 축별로 복원: x는 sx로, y는 sy로 나눔
                    if sx0 != 1.0 or sy0 != 1.0:
                        k0 = k0.copy()
                        k0[:,0] /= sx0
                        k0[:,1] /= sy0
                    if sx1 != 1.0 or sy1 != 1.0:
                        k1 = k1.copy()
                        k1[:,0] /= sx1
                        k1[:,1] /= sy1

                    # 경계 클리핑(시각화 안전장치)
                    H0, W0 = img_o.shape[:2]
                    H1, W1 = img_r.shape[:2]
                    k0[:,0] = np.clip(k0[:,0], 0, W0-1); k0[:,1] = np.clip(k0[:,1], 0, H0-1)
                    k1[:,0] = np.clip(k1[:,0], 0, W1-1); k1[:,1] = np.clip(k1[:,1], 0, H1-1)

                    k0_rgb = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
                    k1_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
                    viz_dir = os.path.join(args.out, "viz"); os.makedirs(viz_dir, exist_ok=True)
                    viz_path = os.path.join(viz_dir, f"{os.path.basename(p)}__vs__{os.path.basename(cand_path)}.png")
                    try:
                        save_debug_matches(viz_path, k0_rgb, k1_rgb, k0, k1, conf)
                        entry["viz"] = viz_path
                    except Exception as e:
                        entry["viz_error"] = str(e)

                entries.append(entry)

            entries.sort(key=lambda x: x["score"], reverse=True)
            best = entries[0]
            best["decision"] = "copied" if best["score"]>=cfg["thresholds"]["copied"] else "uncertain"
            fout.write(json.dumps(best, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
