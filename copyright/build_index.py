import argparse, os, numpy as np, tqdm
from copyright.src.embedder import VisualEmbedder, EmbeddingIndex
from copyright.utils.functionals import load_image_bgr
import yaml

from pathlib import Path
FILE = Path(__file__)
ROOT = FILE.parent.resolve()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--originals", default=str(ROOT / '../assets/original'))
    ap.add_argument("--out", default=str(ROOT / '../outputs/index'))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    ve = VisualEmbedder(name=cfg["embedding"]["name"],
                        device=cfg["device"],
                        input_size=cfg["embedding"]["input_size"],
                        normalize=cfg["faiss"]["normalize"])

    paths = []
    for root,_,files in os.walk(args.originals):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp")):
                paths.append(os.path.join(root,f))
    paths.sort()
    embs = []
    metas = []
    for p in tqdm.tqdm(paths, desc="Embedding originals"):
        img = load_image_bgr(p)
        emb = ve.embed_np_bgr(img)
        embs.append(emb)
        metas.append({"path": p})
    X = np.stack(embs).astype("float32")
    idx = EmbeddingIndex(dim=X.shape[1], nlist=cfg["faiss"]["faiss_nlist"],
                         nprobe=cfg["faiss"]["faiss_nprobe"],
                         normalize=cfg["faiss"]["normalize"])
    idx.add(X, metas)
    os.makedirs(args.out, exist_ok=True)
    idx.save(os.path.join(args.out,"originals.faiss"), os.path.join(args.out,"originals.jsonl"))
    print("Saved index to", args.out)

if __name__ == "__main__":
    main()
