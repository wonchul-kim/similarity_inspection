import argparse, os, json, yaml, random, numpy as np, pandas as pd, tqdm, torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from copyleft_detector import VisualEmbedder, LoFTRMatcher, OCRTextEncoder, FusionMLP

def cosine(a,b): return float((a*b).sum())

class PairDataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.ve = VisualEmbedder(name=cfg["visual_backbone"]["name"],
                                 device=cfg["device"],
                                 input_size=cfg["visual_backbone"]["input_size"],
                                 normalize=cfg["retrieval"]["normalize"])
        self.matcher = LoFTRMatcher(device=cfg["device"]) if cfg["local_match"]["enable"] else None
        self.ocr = OCRTextEncoder(langs=cfg["ocr"]["lang"], sentence_model=cfg["ocr"]["sentence_model"], device=cfg["device"]) if cfg["ocr"]["enable"] else None

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        p0, p1, y = r["original"], r["reseller"], float(r["label"])
        import cv2, numpy as np
        img0 = cv2.imread(p0); img1 = cv2.imread(p1)
        if img0 is None or img1 is None:
            raise FileNotFoundError(p0 if img0 is None else p1)
        e0 = self.ve.embed_np_bgr(img0); e1 = self.ve.embed_np_bgr(img1)
        gcos = cosine(e0, e1)
        lratio = 0.0
        if self.matcher is not None:
            lratio, _ = self.matcher.match_ratio(img0, img1, max_size=self.cfg["local_match"]["max_size"])
        osim = 0.0
        if self.ocr is not None:
            osim, _ = self.ocr.semantic_sim(img0, img1, max_words=self.cfg["ocr"]["max_words"])
        x = np.array([gcos, lratio, osim], dtype=np.float32)
        return x, np.array([y], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True, help="CSV with columns: original,reseller,label (1=copied,0=not)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_path", default="artifacts/fusion_mlp.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    df = pd.read_csv(args.pairs_csv)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_train = int(len(df)*0.8)
    tr, va = df.iloc[:n_train], df.iloc[n_train:]

    train_ds = PairDataset(tr, cfg); val_ds = PairDataset(va, cfg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FusionMLP(hidden=cfg["fusion"]["hidden"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"]=="cuda" else "cpu")
    model.to(device)

    best_auc = -1.0
    for ep in range(args.epochs):
        model.train()
        for X, y in tqdm.tqdm(train_loader, desc=f"Train ep{ep+1}"):
            X,y = X.to(device), y.to(device)
            logits = model(X)
            loss = bce(logits, y.squeeze(1))
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        ys = []; ps = []
        with torch.no_grad():
            for X, y in val_loader:
                X,y = X.to(device), y.to(device)
                logit = model(X)
                prob = torch.sigmoid(logit)
                ys.append(y.squeeze(1).cpu().numpy())
                ps.append(prob.cpu().numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(ys, ps)
        except Exception:
            auc = float("nan")
        print(f"[val] AUC={auc:.4f}")
        if auc==auc and auc>best_auc:
            best_auc = auc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print("Saved:", args.save_path)

if __name__ == "__main__":
    main()
