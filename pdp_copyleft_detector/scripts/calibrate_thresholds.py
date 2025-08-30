import argparse, json, numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_jsonl", required=True, help="scan_dir.py 결과(JSONL)")
    ap.add_argument("--target_fpr", type=float, default=0.01)
    args = ap.parse_args()

    ys, ps = [], []
    with open(args.scan_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if "score" in o:
                ps.append(float(o["score"]))
            if "label" in o:
                ys.append(int(o["label"]))
    ps = np.array(ps)
    print(f"#samples={len(ps)}")
    if len(ys)==len(ps) and len(ys)>0:
        ys = np.array(ys)
        fpr, tpr, thr = roc_curve(ys, ps)
        i = np.argmin(np.abs(fpr-args.target_fpr))
        print(f"Threshold @ FPR≈{args.target_fpr:.3f} => {thr[i]:.4f}, TPR={tpr[i]:.4f}, FPR={fpr[i]:.4f}")
    else:
        print("라벨이 없어 분포 통계만 출력합니다.")
        for p in [0.5,0.7,0.8,0.9,0.95,0.98]:
            print(f"score p{int(p*100)} = {np.quantile(ps, p):.4f}")

if __name__ == "__main__":
    main()
