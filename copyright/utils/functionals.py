import os, re, io, sys, time, json, math, base64, hashlib
import cv2, numpy as np, torch, random, os
from PIL import Image

def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_image_bgr(path: str):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

# def imresize_max_dim(img_bgr, max_dim=1200):
#     h,w = img_bgr.shape[:2]
#     scale = max_dim / max(h,w)
#     if scale >= 1.0:
#         return img_bgr, 1.0
#     nh, nw = int(h*scale), int(w*scale)
#     return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA), scale
def imresize_max_dim(img_bgr, max_dim=1200):
    h, w = img_bgr.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        # 축소 안 했음
        return img_bgr, (1.0, 1.0)   # (sx, sy)
    nh, nw = int(round(h*scale)), int(round(w*scale))
    img = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    sx = nw / float(w)   # x축(가로) 스케일
    sy = nh / float(h)   # y축(세로) 스케일
    return img, (sx, sy)
    

def save_debug_matches(path, img0_rgb, img1_rgb, kpts0, kpts1, matches):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig = plt.figure(figsize=(10,5))
    H0,W0 = img0_rgb.shape[:2]
    H1,W1 = img1_rgb.shape[:2]
    canvas = np.zeros((max(H0,H1), W0+W1, 3), dtype=np.uint8)
    canvas[:H0,:W0] = img0_rgb
    canvas[:H1,W0:W0+W1] = img1_rgb
    # shift kpts1 x by W0
    k1 = kpts1.copy(); k1[:,0] += W0
    lines = np.stack([kpts0, k1], axis=1)
    lc = LineCollection(lines, linewidths=0.5)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(canvas)
    ax.add_collection(lc)
    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
