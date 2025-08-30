import cv2 
from PIL import Image 
import numpy as np
import math
from typing import Tuple, Dict, Any
from skimage.feature import hog

MSER_DELTA = 5


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def ensure_3ch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return arr[:,:,:3]
    return arr


# ---------------- 1) 여백 자동 트림 ----------------
def auto_trim_by_corner_color(im_cv: np.ndarray, tol: float = 8.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = im_cv.shape[:2]
    im_lab = cv2.cvtColor(ensure_3ch(im_cv), cv2.COLOR_BGR2LAB).astype(np.float32)
    corners = np.vstack([
        im_lab[0:20, 0:20].reshape(-1, 3),
        im_lab[0:20, w-20:w].reshape(-1, 3),
        im_lab[h-20:h, 0:20].reshape(-1, 3),
        im_lab[h-20:h, w-20:w].reshape(-1, 3),
    ])
    ref = corners.mean(axis=0)[None, None, :]
    dist = np.linalg.norm(im_lab - ref, axis=2)
    fg = (dist > tol).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    ys, xs = np.where(fg > 0)
    if len(xs) == 0 or len(ys) == 0:
        return im_cv, {"trimmed": False, "bbox": (0,0,w,h)}
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    m = 2
    x0 = max(0, x0-m); y0 = max(0, y0-m); x1 = min(w-1, x1+m); y1 = min(h-1, y1+m)
    return im_cv[y0:y1+1, x0:x1+1], {"trimmed": True, "bbox": (int(x0), int(y0), int(x1), int(y1))}

# ---------------- 2) 정규화 ----------------
def normalize_image(im_cv: np.ndarray) -> np.ndarray:
    im = ensure_3ch(im_cv).copy()
    ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.fastNlMeansDenoising(y, h=3, templateWindowSize=7, searchWindowSize=21)
    y = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge([y,cr,cb]), cv2.COLOR_YCrCb2BGR)

# ---------------- 3) 타일링 ----------------
def sliding_tiles(im_cv: np.ndarray, tile_size: int, stride: int):
    h, w = im_cv.shape[:2]
    boxes=[]
    for y in range(0, max(1, h - tile_size + 1), stride):
        for x in range(0, max(1, w - tile_size + 1), stride):
            boxes.append((x,y,tile_size,tile_size))
    if boxes:
        last_row_y = boxes[-1][1]
        if last_row_y + tile_size < h:
            for x in range(0, max(1, w - tile_size + 1), stride):
                boxes.append((x, h - tile_size, tile_size, tile_size))
        rows = math.ceil((h - tile_size)/stride) + 1
        for r in range(rows):
            y = min(r*stride, h - tile_size)
            if (w - tile_size) % stride != 0:
                boxes.append((w - tile_size, y, tile_size, tile_size))
    return list(dict.fromkeys(boxes))

# ---------------- 4) 텍스트 밀도 ----------------
def estimate_text_mask(gray: np.ndarray) -> np.ndarray:
    mser = cv2.MSER_create()
    mser.setDelta(MSER_DELTA); mser.setMinArea(60); mser.setMaxArea(8000)
    regions, _ = mser.detectRegions(gray)
    mask_mser = np.zeros_like(gray, dtype=np.uint8)
    for pts in regions:
        cv2.fillPoly(mask_mser, [pts.reshape(-1,1,2)], 255)
    sobx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    soby = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    mag  = cv2.addWeighted(cv2.convertScaleAbs(sobx), 0.5, cv2.convertScaleAbs(soby), 0.5, 0)
    _, mask_edge = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(mask_mser, mask_edge)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.medianBlur(mask, 3)

# ---------------- 5) 해시 ----------------
def ahash(im_pil: Image.Image, hash_size: int = 8) -> str:
    im = im_pil.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    arr = np.array(im, dtype=np.float32)
    bits = (arr > arr.mean()).astype(np.uint8).flatten()
    return "".join(["%x" % int("".join(map(str, bits[i:i+4])), 2) for i in range(0, bits.size, 4)])

def dhash(im_pil: Image.Image, hash_size: int = 8) -> str:
    im = im_pil.convert("L").resize((hash_size+1, hash_size), Image.Resampling.LANCZOS)
    arr = np.array(im, dtype=np.int16)
    diff = (arr[:,1:] > arr[:,:-1]).astype(np.uint8).flatten()
    return "".join(["%x" % int("".join(map(str, diff[i:i+4])), 2) for i in range(0, diff.size, 4)])

def phash(im_pil: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    img_size = hash_size * highfreq_factor
    im = im_pil.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    arr = np.array(im, dtype=np.float32)
    dct = cv2.dct(arr)
    dctlow = dct[:hash_size, :hash_size]
    med = np.median(dctlow[1:,1:])
    bits = (dctlow > med).flatten()
    return "".join(["%x" % int("".join('1' if b else '0' for b in bits[i:i+4]), 2) for i in range(0, bits.size, 4)])

# ---------------- 6) 임베딩 ----------------
def visual_embedding(tile_cv: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(tile_cv, cv2.COLOR_BGR2GRAY)
    hog_vec = hog(gray, orientations=9, pixels_per_cell=(32,32), cells_per_block=(2,2),
                  block_norm='L2-Hys', transform_sqrt=True, feature_vector=True)
    hsv = cv2.cvtColor(tile_cv, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv],[0],None,[32],[0,180]).flatten()
    s_hist = cv2.calcHist([hsv],[1],None,[32],[0,256]).flatten()
    v_hist = cv2.calcHist([hsv],[2],None,[32],[0,256]).flatten()
    hist = np.concatenate([h_hist, s_hist, v_hist]); hist /= (hist.sum()+1e-6)
    vec = np.concatenate([hog_vec, hist]).astype(np.float32)
    vec /= (np.linalg.norm(vec)+1e-8)
    return vec

# ---------------- 7) ORB ----------------
def orb_features(tile_cv: np.ndarray):
    orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=8, edgeThreshold=31, patchSize=31)
    kp, desc = orb.detectAndCompute(tile_cv, None)
    return kp, (np.zeros((0,32), np.uint8) if desc is None else desc)
