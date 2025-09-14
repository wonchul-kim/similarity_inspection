import os.path as osp
import argparse, os, numpy as np, tqdm
from copyright.src.embedder import VisualEmbedder, EmbeddingIndex
from copyright.utils.functionals import load_image_bgr, compute_starts
import yaml

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
    cfg['embedding']['output_dir'] = osp.join(cfg['output_dir'], cfg['embedding']['output_dir'])

    patch_cfg = cfg.get("patch", {})
    patch_use = bool(patch_cfg.get("use", False))
    patch_w = int(patch_cfg.get("width", cfg["embedding"]["input_size"]))
    patch_h = int(patch_cfg.get("height", cfg["embedding"]["input_size"]))

    ve = VisualEmbedder(name=cfg["embedding"]["name"],
                        device=cfg["device"],
                        input_size=cfg["embedding"]["input_size"],
                        normalize=cfg["faiss"]["normalize"])

    paths = []
    for root,_,files in os.walk(cfg['original_image_dir']):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp")):
                paths.append(os.path.join(root,f))
    paths.sort()

    embs = []
    metas = []

    total_images = len(paths)
    print(f"Found {total_images} images. patch_use={patch_use}, patch_size=({patch_w},{patch_h})")

    for p in tqdm.tqdm(paths, desc="Embedding originals"):
        try:
            img = load_image_bgr(p)
            if img is None:
                print("Warning: failed to load", p)
                continue
        except Exception as e:
            print("Warning: exception loading", p, e)
            continue

        if not patch_use:
            try:
                emb = ve.embed_np_bgr(img)
                embs.append(emb)
                metas.append({"path": p})
            except Exception as e:
                print("Warning: failed to embed", p, e)
            continue

        h, w = img.shape[:2]
        x_starts = compute_starts(w, patch_w)
        y_starts = compute_starts(h, patch_h)

        for y in y_starts:
            for x in x_starts:
                x2 = min(w, x + patch_w)
                y2 = min(h, y + patch_h)
                pw = x2 - x
                ph = y2 - y
                patch = img[y:y2, x:x2].copy()

                try:
                    emb = ve.embed_np_bgr(patch)
                    embs.append(emb)
                    metas.append({"path": p, "patch": {"x": int(x), "y": int(y), "w": int(pw), "h": int(ph)}})
                except Exception as e:
                    print(f"Warning: failed to embed patch of {p} at ({x},{y})", e)

    if len(embs) == 0:
        raise RuntimeError("No embeddings were produced. Check your image paths and patch settings.")

    X = np.stack(embs).astype("float32")
    idx = EmbeddingIndex(dim=X.shape[1], nlist=cfg["faiss"]["faiss_nlist"],
                         nprobe=cfg["faiss"]["faiss_nprobe"],
                         normalize=cfg["faiss"]["normalize"])
    idx.add(X, metas)
    os.makedirs(cfg['embedding']['output_dir'], exist_ok=True)
    idx.save(os.path.join(cfg['embedding']['output_dir'],"originals.faiss"),
             os.path.join(cfg['embedding']['output_dir'],"originals.jsonl"))
    print("Saved index to", cfg['embedding']['output_dir'])


if __name__ == "__main__":
    main()
