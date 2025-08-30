import argparse, os, json, numpy as np, tqdm
from PIL import Image
from copyleft_detector import VisualEmbedder, EmbeddingIndex
from copyleft_detector.utils import load_image_bgr
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--originals", default='/HDD/_projects/github/similarity_inspection/assets/original')
    ap.add_argument("--config", default="/HDD/_projects/github/similarity_inspection/pdp_copyleft_detector/configs/default.yaml")
    ap.add_argument("--out", default='/HDD/_projects/github/similarity_inspection/outputs/pdp/index')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    ve = VisualEmbedder(name=cfg["visual_backbone"]["name"],
                        device=cfg["device"],
                        input_size=cfg["visual_backbone"]["input_size"],
                        normalize=cfg["retrieval"]["normalize"])

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
    idx = EmbeddingIndex(dim=X.shape[1], nlist=cfg["retrieval"]["faiss_nlist"],
                         nprobe=cfg["retrieval"]["faiss_nprobe"],
                         normalize=cfg["retrieval"]["normalize"])
    idx.add(X, metas)
    os.makedirs(args.out, exist_ok=True)
    idx.save(os.path.join(args.out,"originals.faiss"), os.path.join(args.out,"originals.jsonl"))
    print("Saved index to", args.out)

if __name__ == "__main__":
    main()
