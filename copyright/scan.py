import time
import os.path as  osp
import argparse, os, json, numpy as np, tqdm, cv2, yaml, math
from copyright.src import (VisualEmbedder, EmbeddingIndex)
from copyright.utils.functionals import load_image_bgr, save_debug_matches, compute_starts, cosine

from pathlib import Path
FILE = Path(__file__)
ROOT = FILE.parent.resolve()


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    return ap.parse_args()

def main():
    args = get_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    cfg["faiss"]["output_dir"] = osp.join(cfg["output_dir"], cfg["faiss"]["output_dir"])
    os.makedirs(cfg["faiss"]["output_dir"], exist_ok=True)
    
    patch_cfg = cfg.get("patch", {})
    patch_use = bool(patch_cfg.get("use", False))
    patch_w = int(patch_cfg.get("width", cfg["embedding"]["input_size"]))
    patch_h = int(patch_cfg.get("height", cfg["embedding"]["input_size"]))

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

    out_path = os.path.join(cfg["faiss"]["output_dir"], 'scan.json')
    with open(out_path, "w", encoding="utf-8") as fout:
        for p in tqdm.tqdm(res_paths, desc="Scanning"):
            img_r = load_image_bgr(p)
            if img_r is None:
                print("Warning: failed to load reseller image:", p)
                continue
            h, w = img_r.shape[:2]

            # build list of patches for this reseller image
            patches = []
            if not patch_use:
                # treat whole image as single "patch" (coords cover full image)
                patches.append({"x":0,"y":0,"w":w,"h":h, "img": img_r})
            else:
                x_starts = compute_starts(w, patch_w)
                y_starts = compute_starts(h, patch_h)
                for y in y_starts:
                    for x in x_starts:
                        x2 = min(w, x + patch_w)
                        y2 = min(h, y + patch_h)
                        pw = x2 - x
                        ph = y2 - y
                        patch_img = img_r[y:y2, x:x2].copy()
                        patches.append({"x":int(x),"y":int(y),"w":int(pw),"h":int(ph),"img":patch_img})

            per_patch_results = []
            best_scores = []

            for patch in patches:
                patch_img = patch["img"]
                try:
                    e_r = ve.embed_np_bgr(patch_img).astype("float32")[None,:]  # shape (1, dim)
                except Exception as e:
                    print("Warning: failed to embed reseller patch:", p, patch.get("x"), patch.get("y"), e)
                    continue

                # search index (index holds original patches' embeddings)
                D, I, metas = idx.search(e_r, topk=cfg["faiss"]["topk"])
                D0 = D[0]  # topk distances/scores
                metas0 = metas[0]

                # build topk list (keep original meta + score)
                topk_list = []
                for rank, (d, m) in enumerate(zip(D0, metas0)):
                    # some indices could be -1 if index smaller than topk; metas should still be defined in EmbeddingIndex.search
                    topk_list.append({
                        "rank": int(rank),
                        "score": float(d),
                        "original": m
                    })

                # pick best (max score)
                best_idx = int(np.argmax(D0))
                best_score = float(D0[best_idx])
                best_meta = metas0[best_idx]

                # per-patch decision
                patch_decision = "copied" if best_score >= cfg["thresholds"]["copied"] else "uncertain"

                per_patch_results.append({
                    "patch": {"x": int(patch["x"]), "y": int(patch["y"]), "w": int(patch["w"]), "h": int(patch["h"])},
                    "best_score": best_score,
                    "best_original": best_meta,
                    "topk": topk_list,
                    "decision": patch_decision
                })

                best_scores.append(best_score)

            # image-level aggregation: average of per-patch best scores
            if len(best_scores) == 0:
                avg_score = 0.0
            else:
                avg_score = float(np.mean(best_scores))

            image_decision = "copied" if avg_score >= cfg["thresholds"]["copied"] else "uncertain"

            # compose final record
            record = {
                "reseller": p,
                "n_patches": len(per_patch_results),
                "patch_use": bool(patch_use),
                "patch_width": patch_w,
                "patch_height": patch_h,
                "patches": per_patch_results,
                "avg_patch_best_score": avg_score,
                "decision": image_decision,
                "timestamp": time.time()
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Scan finished. Results saved to", out_path)

if __name__ == "__main__":
    main()
