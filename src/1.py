"""
A단계: 공식 상세페이지 인제스트 파이프라인 (단일 스크립트)

기능
- 여백 자동 트림(Lab 색거리 기반)
- 정규화(Y채널 노이즈↓/대비표준화)
- 중첩 타일링(기본 512x512, stride=224)
- 타일별 특징
  • 텍스트 비율(MSER+에지), 에지 밀도
  • pHash/aHash/dHash
  • ORB 키포인트/디스크립터(후속 기하정합용)
  • HOG+HSV 임베딩(후속 후보검색용)
  • 문서내 유니크니스(임베딩 상호유사도 기반)
- 시각화(타일 오버레이, 텍스트 히트맵, 유니크니스 오버레이)
- 메타데이터 저장(official_meta.json, tiles_meta.json)

의존성: opencv-python, numpy, pillow, scikit-image, pandas (선택), matplotlib(없어도 됨)
"""

import os, json, math, hashlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog
from functionals import *

# ---------------- 유틸 ----------------

def image_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def text_density(tile_cv: np.ndarray) -> Tuple[float, float, np.ndarray]:
    gray = cv2.cvtColor(tile_cv, cv2.COLOR_BGR2GRAY)
    mask = estimate_text_mask(gray)
    ratio = (mask > 0).sum() / mask.size
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = edges.mean() / 255.0
    return float(ratio), float(edge_ratio), mask


# ---------------- 8) 유니크니스 ----------clip------
def compute_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) == 0: return np.zeros(0, dtype=np.float32)
    sims  = embeddings @ embeddings.T
    denom = np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims  = sims / (denom @ denom.T + 1e-8)
    np.fill_diagonal(sims, -1.0)
    K = min(8, embeddings.shape[0]-1) if embeddings.shape[0] > 1 else 1
    idx   = np.argsort(-sims, axis=1)[:, :K]
    topk  = np.take_along_axis(sims, idx, axis=1)
    avg   = topk.mean(axis=1) if K > 0 else np.zeros(embeddings.shape[0])
    uniq  = 1.0 - avg
    return (uniq - uniq.min())/(uniq.max()-uniq.min()+1e-8)

# ---------------- 데이터클래스 ----------------
@dataclass
class TileInfo:
    tile_id: str
    x: int; y: int; w: int; h: int
    phash: str; ahash: str; dhash: str
    text_ratio: float; edge_density: float; uniqueness: float
    orb_kp: int
    orb_desc_path: str
    embedding_path: str
    png_path: str

# ---------------- 메인 ----------------
def run(IMAGE_PATH: str, OUT_ROOT: str):
    # 원본 로드
    orig = pil_to_cv(Image.open(IMAGE_PATH))
    meta = {"input_path": IMAGE_PATH, "sha1": image_sha1(IMAGE_PATH),
            "orig_size": (orig.shape[1], orig.shape[0])}
    cv2.imwrite(f"{OUT_ROOT}/visuals/00_original.jpg", orig)

    # 트림
    trimmed, trim_info = auto_trim_by_corner_color(orig, tol=8.0)
    meta["trim_info"] = trim_info
    cv2.imwrite(f"{OUT_ROOT}/visuals/01_trimmed.jpg", trimmed)

    # 정규화
    norm = normalize_image(trimmed)
    cv2.imwrite(f"{OUT_ROOT}/visuals/02_normalized.jpg", norm)

    # 타일링
    boxes = sliding_tiles(norm, TILE_SIZE, STRIDE)
    vis = norm.copy()
    for (x,y,w,h) in boxes:
        cv2.rectangle(vis, (x,y), (x+w-1,y+h-1), (0,255,0), 2)
    # 가시성 위해 다운스케일
    scale = min(1200/vis.shape[1], 2400/vis.shape[0], 1.0)
    vis_small = cv2.resize(vis, (int(vis.shape[1]*scale), int(vis.shape[0]*scale)))
    cv2.imwrite(f"{OUT_ROOT}/visuals/03_tiles_overlay.jpg", vis_small)

    # 타일 특징 추출
    tile_infos: List[TileInfo] = []
    embeddings = []
    text_mask_full = np.zeros(norm.shape[:2], dtype=np.uint8)

    for idx, (x,y,w,h) in enumerate(boxes):
        tile = norm[y:y+h, x:x+w].copy()
        tile_pil = cv_to_pil(tile)
        tid = f"tile_{idx:06d}_y{y:05d}_x{x:05d}"
        png_path = f"{OUT_ROOT}/tiles/{tid}.png"
        cv2.imwrite(png_path, tile)

        # 해시
        a_h = ahash(tile_pil); d_h = dhash(tile_pil); p_h = phash(tile_pil)

        # 텍스트/에지
        t_ratio, e_ratio, tmask = text_density(tile)
        text_mask_full[y:y+h, x:x+w] = np.maximum(text_mask_full[y:y+h, x:x+w], tmask)

        # 임베딩
        emb = visual_embedding(tile)
        emb_path = f"{OUT_ROOT}/features/{tid}.emb.npy"; np.save(emb_path, emb)
        embeddings.append(emb)

        # ORB
        kps, desc = orb_features(tile)
        orb_path = f"{OUT_ROOT}/features/{tid}.orb.npy"; np.save(orb_path, desc)

        tile_infos.append(TileInfo(
            tile_id=tid, x=int(x), y=int(y), w=int(w), h=int(h),
            phash=p_h, ahash=a_h, dhash=d_h,
            text_ratio=float(t_ratio), edge_density=float(e_ratio), uniqueness=0.0,
            orb_kp=len(kps), orb_desc_path=orb_path, embedding_path=emb_path, png_path=png_path
        ))

    # 유니크니스
    embeddings = np.vstack(embeddings) if len(embeddings) else np.zeros((0,1), np.float32)
    uniq = compute_uniqueness(embeddings) if len(embeddings) else np.array([])
    for i, u in enumerate(uniq):
        tile_infos[i].uniqueness = float(u)

    # 메타 저장
    with open(f"{OUT_ROOT}/features/tiles_meta.json", "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in tile_infos], f, ensure_ascii=False, indent=2)
    with open(f"{OUT_ROOT}/official_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 히트맵/오버레이
    heat = cv2.applyColorMap(cv2.normalize(text_mask_full, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(ensure_3ch(norm), 0.6, heat, 0.4, 0)
    cv2.imwrite(f"{OUT_ROOT}/visuals/04_text_mask_heatmap.jpg", overlay)

    uni_vis = ensure_3ch(norm.copy())
    for i, (x,y,w,h) in enumerate(boxes):
        u = float(uniq[i]) if len(uniq) else 0.0
        color = (int(255*(1.0-u)), 50, int(255*u))  # blue->red
        cv2.rectangle(uni_vis, (x,y), (x+w-1,y+h-1), color, 4)
    cv2.imwrite(f"{OUT_ROOT}/visuals/05_uniqueness_overlay.jpg", uni_vis)

    print("A단계 인제스트 완료.")
    print("출력 경로:", OUT_ROOT)

# ---------------- 실행 ----------------
if __name__ == "__main__":
    from pathlib import Path 
    FILE = Path(__file__).resove()
    ROOT = FILE.parent
    
    # ---------------- CONFIG ----------------
    IMAGE_PATH = "/HDD/_projects/github/similarity_inspection/assets/original/3f2b8fa7-3455-42d5-b274-fcd4acc72c36.png"  # 입력: 공식 상세페이지 이미지
    OUT_ROOT   = "/HDD/etc/outputs/sim"                               # 출력 루트
    TILE_SIZE  = 512                                                     # 타일 크기
    STRIDE     = 224                                                     # 타일 stride
    MSER_DELTA = 5                                                       # 텍스트 추정 민감도

    # ---------------- I/O 준비 ----------------
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(f"{OUT_ROOT}/tiles", exist_ok=True)
    os.makedirs(f"{OUT_ROOT}/features", exist_ok=True)
    os.makedirs(f"{OUT_ROOT}/visuals", exist_ok=True)

    
    run(IMAGE_PATH, OUT_ROOT)
