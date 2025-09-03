#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non‑DL Image Plagiarism Detector — tailored for cosmetics product *detail pages*
================================================================================

목표
----
- 딥러닝 없이(= 전통 컴퓨터비전 + 견고한 해시/특징 매칭) 리셀러가 공식 상세페이지 이미지를 **무단 사용**했는지 판별.
- 상세페이지의 특성(세로로 길다, 여백이 많다, 텍스트가 많다)을 반영하여 **레이아웃 중심** 지문과 **부분 이미지(제품컷) 재사용**까지 탐지.

핵심 아이디어(멀티 패스)
-------------------------
1) **전역(whole-page) 유사도**
   - pHash(DCT) + HSV 색상 히스토그램으로 전체적인 톤/구성을 비교.
2) **세로 스트립 시퀀스 해시(LayoutStripHash)**
   - 폭을 정규화한 뒤 세로 스트립마다 pHash → 두 시퀀스를 **DTW**(동적 시간 왜곡)로 정렬해 삽입/삭제/크롭을 허용.
   - 긴 상세페이지(스크롤 페이지)에서도 강건.
3) **레이아웃 바코드(Whitespace Barcode)**
   - 각 행(row)별 **백색 비율**을 벡터화(길이 고정) → 벡터 상관도(정규화 상관계수)로 여백 패턴/텍스트 블록 패턴을 비교.
   - OCR 없이 텍스트 많은 페이지의 **구조**만으로도 유사도 확보.
4) **지역(부분) 이미지 재사용 탐지**
   - ORB 특징 + RANSAC으로 **제품컷/배너** 등 부분 재사용을 기하학적으로 검증(스케일/회전/원근 변형 허용).
   - 추가로 **다중 스케일 템플릿 매칭(NCC)**로 워터마크/여백 변형에도 견고한 보조 신호.
5) **결정 로직**
   - 위 신호들을 보정/정규화하여 결합 점수 산출.
   - “레이아웃 강한 유사 + 부분 이미지 재사용” 조합을 특히 강하게 가중.

사용법
------
python detect_plagiarism.py \
    --originals ./originals \
    --candidates ./resellers \
    --out ./report \
    --jobs 4

필요 패키지: opencv-python, numpy
(선택) pillow는 미사용. skimage 미사용. **딥러닝 프레임워크 전혀 사용하지 않음.**

pip install opencv-python numpy

출력물
------
- CSV: pairwise 결과(최상 매칭 기준), 구성 점수, 최종 판정.
- 시각화: 매칭 근거를 **annotated_*.jpg**로 저장(호모그래피 사각형/템플릿 위치/스트립 정렬 점수 등).

주의/튜닝
---------
- 임계값은 보수적으로 설정(위양성 억제). 조직 내 데이터로 **thresholds.json**을 점차 업데이트 권장.
- 초장문의 상세페이지는 **--strip-height**/--strip-overlap 조정.
"""

from __future__ import annotations
import os
import sys
import glob
import math
import json
import csv
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2


# -------------------------------
# 유틸
# -------------------------------

def imread_rgb(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(path: str, img_rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1].lower()
    # ensure unicode path support
    _, buf = cv2.imencode(
        ".jpg" if ext not in [".png", ".webp"] else ext, bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    )
    buf.tofile(path)


def resize_by_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def near_white_mask(img_rgb: np.ndarray, thr: int = 245) -> np.ndarray:
    # 백색(또는 매우 밝은) 픽셀 마스크
    return (img_rgb > thr).all(axis=2).astype(np.uint8)


# -------------------------------
# pHash / Hamming
# -------------------------------

def dct2(a: np.ndarray) -> np.ndarray:
    return cv2.dct(a.astype(np.float32))


def phash(img: np.ndarray, hash_size: int = 16, highfreq_factor: int = 4) -> np.ndarray:
    """DCT 기반 pHash → 64/256비트 등으로 확장 가능.
    반환: {0,1}의 1D np.ndarray (길이 = hash_size*hash_size)
    """
    gray = to_gray(img)
    # 크기를 (hash_size*highfreq_factor) 로 리사이즈 후 DCT
    size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = dct2(resized)
    # 좌상단 저주파 블록 취득(DC 제외)
    dct_low = dct[:hash_size, :hash_size]
    dct_flat = dct_low.flatten()
    # 중앙값 기준 이진화(DC 성분 영향을 낮추기 위해 median)
    med = np.median(dct_flat[1:]) if dct_flat.size > 1 else np.median(dct_flat)
    bits = (dct_flat > med).astype(np.uint8)
    return bits


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    assert a.shape == b.shape
    return int(np.count_nonzero(a != b))


# -------------------------------
# HSV Color Histogram (MPEG‑7풍)
# -------------------------------

def hsv_hist(img: np.ndarray, bins: Tuple[int, int, int] = (8, 3, 3)) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def bhattacharyya(hist1: np.ndarray, hist2: np.ndarray) -> float:
    # 0(동일) ~ 1(매우 다름)에 가까운 거리로 변환
    bc = np.sum(np.sqrt(hist1 * hist2))
    bc = np.clip(bc, 1e-8, 1.0)
    return float(np.sqrt(1.0 - bc))


# -------------------------------
# 레이아웃 바코드(Whitespace Barcode)
# -------------------------------

def whitespace_barcode(img: np.ndarray, target_len: int = 256, white_thr: int = 245) -> np.ndarray:
    """각 행(row)마다 백색 비율(0~1)을 계산하고 target_len으로 리샘플.
    세로로 긴 상세페이지의 여백/텍스트 블록 패턴을 반영.
    """
    rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mask = near_white_mask(rgb, thr=white_thr)
    row_white_ratio = mask.mean(axis=1)  # (H,)
    # 보정: 모폴로지로 소소한 잡음 완화
    row_white_ratio = cv2.GaussianBlur(row_white_ratio.reshape(-1, 1), (0, 0), 1.5).ravel()
    # 길이 정규화
    x = np.linspace(0, 1, num=row_white_ratio.size)
    xp = np.linspace(0, 1, num=target_len)
    resampled = np.interp(xp, x, row_white_ratio)
    # 표준화
    mu, sigma = float(np.mean(resampled)), float(np.std(resampled) + 1e-6)
    return (resampled - mu) / sigma


def ncc_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float(np.clip(np.mean(a * b), -1.0, 1.0))


# -------------------------------
# 세로 스트립 시퀀스 해시 + DTW
# -------------------------------

def strip_hash_sequence(img: np.ndarray, norm_width: int = 512, strip_h: int = 128, overlap: float = 0.5,
                        hash_size: int = 16) -> List[np.ndarray]:
    imgw = resize_by_width(img, norm_width)
    H, W = imgw.shape[:2]
    step = max(1, int(strip_h * (1.0 - overlap)))
    seq = []
    for y in range(0, H - strip_h + 1, step):
        crop = imgw[y:y + strip_h, :]
        seq.append(phash(crop, hash_size=hash_size))
    if not seq:
        seq = [phash(imgw, hash_size=hash_size)]
    return seq


def dtw_distance_hash(seqA: List[np.ndarray], seqB: List[np.ndarray], band: Optional[int] = None) -> float:
    """해시 시퀀스 간 DTW 거리 (비용=정규화 해밍거리). 반환: 총 비용.
    band: 사코에-치바 밴드 폭(선택) → 연산량/왜곡 제어
    """
    n, m = len(seqA), len(seqB)
    if n == 0 or m == 0:
        return float("inf")
    max_h = float(seqA[0].size)
    band = band or max(5, abs(n - m) + 5)
    INF = 1e12
    dp = np.full((n + 1, m + 1), INF, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        for j in range(j_start, j_end + 1):
            cost = hamming(seqA[i - 1], seqB[j - 1]) / max_h
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


# -------------------------------
# ORB 지역 특징 + RANSAC
# -------------------------------

def keypoint_inlier_score(img1: np.ndarray, img2: np.ndarray, max_side: int = 1400,
                          nfeatures: int = 5000) -> Tuple[int, int, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """ORB 특징 매칭 후 RANSAC 호모그래피로 공간 검증.
    반환: (inliers, total_good_matches, inlier_ratio, H, corners_in_img2)
    """
    def downscale(img):
        h, w = img.shape[:2]
        scale = max(h, w) / float(max_side)
        return cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA) if scale > 1 else img

    a = downscale(img1)
    b = downscale(img2)
    gray1, gray2 = to_gray(a), to_gray(b)

    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
    k1, d1 = orb.detectAndCompute(gray1, None)
    k2, d2 = orb.detectAndCompute(gray2, None)

    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0, 0, 0.0, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return 0, len(good), 0.0, None, None

    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return 0, len(good), 0.0, None, None

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / max(1, len(good))

    # 원본 a 이미지의 코너를 b로 사상해서 표시용 좌표 반환
    h, w = a.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H)

    return inliers, len(good), float(inlier_ratio), H, proj


# -------------------------------
# 템플릿 매칭 (다중 스케일, NCC)
# -------------------------------

def local_entropy(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    k = np.ones((ksize, ksize), np.float32)
    mean = cv2.filter2D(gray.astype(np.float32), -1, k / (ksize * ksize))
    mean_sq = cv2.filter2D((gray.astype(np.float32) ** 2), -1, k / (ksize * ksize))
    var = np.clip(mean_sq - mean ** 2, 0, None)
    # 엔트로피 근사: 분산 기반(정규분포 가정)
    return 0.5 * np.log(2 * math.pi * math.e * (var + 1e-6))


def high_entropy_patches(img: np.ndarray, size: int = 256, stride: int = 128, topk: int = 6) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    gray = to_gray(img)
    H, W = gray.shape
    ent = local_entropy(gray, ksize=9)
    patches = []
    for y in range(0, max(1, H - size + 1), stride):
        for x in range(0, max(1, W - size + 1), stride):
            e = float(ent[y:y + size, x:x + size].mean())
            patches.append((e, (x, y, size, size)))
    patches.sort(key=lambda t: t[0], reverse=True)
    sel = patches[:topk]
    out = []
    for _, (x, y, w, h) in sel:
        out.append((img[y:y + h, x:x + w], (x, y, w, h)))
    return out


def multiscale_ncc(scene: np.ndarray, templ: np.ndarray, scales: List[float] | None = None) -> Tuple[float, Tuple[int, int, int, int]]:
    if scales is None:
        scales = [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4]
    gray_s = to_gray(scene)
    gray_t = to_gray(templ)
    best_val = -1.0
    best_box = (0, 0, 0, 0)
    for s in scales:
        th = max(8, int(gray_t.shape[0] * s))
        tw = max(8, int(gray_t.shape[1] * s))
        if th >= gray_s.shape[0] or tw >= gray_s.shape[1]:
            continue
        t_resized = cv2.resize(gray_t, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gray_s, t_resized, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = float(max_val)
            x, y = max_loc
            best_box = (x, y, tw, th)
    return best_val, best_box


# -------------------------------
# 점수 결합 및 판정
# -------------------------------
@dataclass
class Scores:
    ham64: float
    color_bhat: float
    layout_corr: float
    strip_dtw_cost: float
    key_inliers: int
    key_inlier_ratio: float
    templ_ncc: float

    def composite(self) -> float:
        # 0~1 사이 점수. 높을수록 도용 가능성 ↑
        # 해밍/바타차리야/DTW는 거리를 유사도로 변환.
        s_global = max(0.0, 1.0 - self.ham64) * 0.35 + max(0.0, 1.0 - self.color_bhat) * 0.15
        s_layout = max(0.0, (self.layout_corr + 1) / 2) * 0.25  # -1~1 → 0~1
        # DTW 비용은 평균 해밍(0~1) 수준으로 정규화되어 있음
        s_strip = max(0.0, 1.0 - min(1.0, self.strip_dtw_cost / 1.0)) * 0.15
        s_local = min(1.0, self.key_inliers / 80.0) * 0.07 + min(1.0, self.key_inlier_ratio) * 0.03
        s_templ = max(0.0, (self.templ_ncc - 0.5) / 0.5) * 0.05  # NCC 0.5↑ 가산
        return float(s_global + s_layout + s_strip + s_local + s_templ)


@dataclass
class Decision:
    verdict: str
    reason: str


def decide(scores: Scores) -> Decision:
    comp = scores.composite()
    # 보수적 임계값. 내부 데이터로 재튜닝 권장.
    # 강한 증거 우선 규칙들:
    strong_layout = scores.layout_corr > 0.85 and scores.strip_dtw_cost < 0.25
    strong_local = (scores.key_inliers >= 30 and scores.key_inlier_ratio >= 0.35) or scores.templ_ncc >= 0.75
    very_high_global = (1.0 - scores.ham64) > 0.92 and (1.0 - scores.color_bhat) > 0.85

    if strong_layout and strong_local:
        return Decision("PLAGIARISM-LIKELY", "레이아웃 고유사 + 부분 이미지 재사용 강한 증거")
    if very_high_global and (strong_layout or strong_local):
        return Decision("PLAGIARISM-LIKELY", "전역 유사 + 추가 증거")
    if comp >= 0.78:
        return Decision("PLAGIARISM-LIKELY", f"종합 점수 {comp:.2f} (보수 임계 초과)")
    if comp <= 0.45 and scores.key_inliers < 10 and scores.templ_ncc < 0.6:
        return Decision("UNLIKELY", f"종합 점수 {comp:.2f} + 지역 증거 미약")
    return Decision("UNCERTAIN", f"종합 점수 {comp:.2f} — 수동 검토 권장")


# -------------------------------
# 페어 평가
# -------------------------------
@dataclass
class PairResult:
    original: str
    candidate: str
    scores: Scores
    decision: Decision


def evaluate_pair(img_o: np.ndarray, img_c: np.ndarray,
                  strip_w: int = 512, strip_h: int = 128, strip_overlap: float = 0.5,
                  hash_size: int = 16) -> Tuple[Scores, Dict[str, np.ndarray]]:
    # 전역 해시/색상
    h_o = phash(img_o, hash_size=hash_size)
    h_c = phash(img_c, hash_size=hash_size)
    ham = hamming(h_o, h_c) / float(h_o.size)  # 0~1
    hist_o, hist_c = hsv_hist(img_o), hsv_hist(img_c)
    bhat = bhattacharyya(hist_o, hist_c)      # 0~1 (작을수록 유사)

    # 레이아웃 바코드
    bc_o = whitespace_barcode(img_o, target_len=256)
    bc_c = whitespace_barcode(img_c, target_len=256)
    layout_corr = ncc_1d(bc_o, bc_c)          # -1~1

    # 스트립 해시 시퀀스 + DTW
    seq_o = strip_hash_sequence(img_o, norm_width=strip_w, strip_h=strip_h, overlap=strip_overlap, hash_size=hash_size)
    seq_c = strip_hash_sequence(img_c, norm_width=strip_w, strip_h=strip_h, overlap=strip_overlap, hash_size=hash_size)
    dtw_cost = dtw_distance_hash(seq_o, seq_c, band=None) / max(1, max(len(seq_o), len(seq_c)))  # 평균 비용(0~1 근사)

    # 지역 증거: ORB + RANSAC
    inliers, total_good, inlier_ratio, H, proj = keypoint_inlier_score(img_o, img_c)

    # 템플릿 보조: 원본에서 상위 엔트로피 패치들
    best_ncc = -1.0
    best_box = (0, 0, 0, 0)
    for patch, _ in high_entropy_patches(img_o, size=256, stride=160, topk=5):
        ncc, box = multiscale_ncc(img_c, patch)
        if ncc > best_ncc:
            best_ncc, best_box = ncc, box

    scores = Scores(
        ham64=ham,
        color_bhat=bhat,
        layout_corr=layout_corr,
        strip_dtw_cost=dtw_cost,
        key_inliers=inliers,
        key_inlier_ratio=inlier_ratio,
        templ_ncc=best_ncc,
    )

    # 디버그/리포트 이미지 생성
    vis = visualize_report(img_o, img_c, scores, H, proj, best_box)

    return scores, vis


# -------------------------------
# 시각화
# -------------------------------

def draw_polygon(img: np.ndarray, poly: np.ndarray, color=(0, 255, 0), thickness=3) -> np.ndarray:
    out = img.copy()
    poly = poly.reshape(-1, 2).astype(int)
    for i in range(4):
        p1 = tuple(poly[i % 4])
        p2 = tuple(poly[(i + 1) % 4])
        cv2.line(out, p1, p2, color, thickness)
    return out


def visualize_report(img_o: np.ndarray, img_c: np.ndarray, scores: Scores,
                     H: Optional[np.ndarray], proj_corners: Optional[np.ndarray], templ_box: Tuple[int, int, int, int]) -> Dict[str, np.ndarray]:
    # 사이드바이사이드 + 결과 텍스트
    oh, ow = img_o.shape[:2]
    ch, cw = img_c.shape[:2]
    scale = 800 / max(oh, ch)
    o2 = cv2.resize(img_o, (int(ow * scale), int(oh * scale)))
    c2 = cv2.resize(img_c, (int(cw * scale), int(ch * scale)))

    canvas_h = max(o2.shape[0], c2.shape[0])
    canvas = np.full((canvas_h, o2.shape[1] + c2.shape[1] + 20, 3), 255, np.uint8)
    canvas[:o2.shape[0], :o2.shape[1]] = o2
    canvas[:c2.shape[0], o2.shape[1] + 20:o2.shape[1] + 20 + c2.shape[1]] = c2

    # 호모그래피 결과 폴리곤(후반부 위치)
    if proj_corners is not None:
        poly = proj_corners.copy()
        poly[:, :, 0] += (o2.shape[1] + 20) / scale  # 오른쪽 이미지 오프셋 역보정 → 그리기 전 다시 스케일
        # 원래 c 이미지 스케일에서 그려진 뒤, 최종 캔버스로 리사이즈가 쉬워서 아래처럼 직접 그리지 않음.

    # 오른쪽 원본 c2에 바로 그리기
    c_draw = c2.copy()
    if proj_corners is not None:
        # proj_corners는 a(축소 전)→b(축소 전). 시각화를 위해 비율 맞춤.
        s = c2.shape[1] / img_c.shape[1]
        poly = proj_corners.copy()
        poly[:, :, 0] *= s
        poly[:, :, 1] *= s
        c_draw = draw_polygon(c_draw, poly, color=(0, 200, 0), thickness=3)

    # 템플릿 박스
    x, y, w, h = templ_box
    if w > 0 and h > 0:
        s = c2.shape[1] / img_c.shape[1]
        x2, y2, w2, h2 = int(x * s), int(y * s), int(w * s), int(h * s)
        cv2.rectangle(c_draw, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 3)

    # 다시 합치기
    canvas[:c2.shape[0], o2.shape[1] + 20:o2.shape[1] + 20 + c2.shape[1]] = c_draw

    # 텍스트 블록
    txt = [
        f"Global pHash sim = {1.0 - scores.ham64:.3f} (1-ham)",
        f"Color similarity = {1.0 - scores.color_bhat:.3f} (1-bhat)",
        f"Layout corr (whitespace) = {scores.layout_corr:.3f}",
        f"Strip-DTW similarity = {1.0 - scores.strip_dtw_cost:.3f}",
        f"ORB inliers = {scores.key_inliers} (ratio={scores.key_inlier_ratio:.2f})",
        f"Template NCC = {scores.templ_ncc:.3f}",
        f"Composite = {scores.composite():.3f}",
    ]

    y0 = 20
    for t in txt:
        cv2.putText(canvas, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y0 += 20

    return {"report": canvas}


# -------------------------------
# 배치 처리 & CLI
# -------------------------------

def collect_images(folder: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)


def process_all(originals_dir: str, candidates_dir: str, out_dir: str, jobs: int = 1) -> None:
    os.makedirs(out_dir, exist_ok=True)
    originals = collect_images(originals_dir)
    candidates = collect_images(candidates_dir)

    if not originals or not candidates:
        print("[!] 원본 또는 후보 이미지가 없습니다.")
        return

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        header = [
            "candidate", "best_original", "verdict", "reason",
            "sim_phash", "sim_color", "layout_corr", "sim_strip", "key_inliers", "key_inlier_ratio", "templ_ncc", "composite"
        ]
        writer.writerow(header)

        for cand in candidates:
            img_c = imread_rgb(cand)
            best: Optional[PairResult] = None

            for org in originals:
                img_o = imread_rgb(org)
                scores, vis = evaluate_pair(img_o, img_c)
                dec = decide(scores)
                pair = PairResult(original=org, candidate=cand, scores=scores, decision=dec)

                # report 이미지 저장(상위만 남길 예정이라 임시 경로에 저장)
                base = os.path.splitext(os.path.basename(cand))[0]
                baseo = os.path.splitext(os.path.basename(org))[0]
                rep_path = os.path.join(out_dir, f"_tmp_{base}__{baseo}.jpg")
                imwrite(rep_path, vis["report"])  # 임시 저장

                if (best is None) or (pair.scores.composite() > best.scores.composite()):
                    # 이전 임시 파일 삭제
                    if best is not None:
                        prev = os.path.join(out_dir, f"_tmp_{os.path.splitext(os.path.basename(best.candidate))[0]}__{os.path.splitext(os.path.basename(best.original))[0]}.jpg")
                        try:
                            os.remove(prev)
                        except OSError:
                            pass
                    best = pair

            assert best is not None
            # 최종 베스트의 임시 파일명을 정식 파일명으로 변경
            base = os.path.splitext(os.path.basename(best.candidate))[0]
            baseo = os.path.splitext(os.path.basename(best.original))[0]
            final_rep = os.path.join(out_dir, f"annotated_{base}__vs__{baseo}.jpg")
            tmp_rep = os.path.join(out_dir, f"_tmp_{base}__{baseo}.jpg")
            if os.path.exists(tmp_rep):
                os.replace(tmp_rep, final_rep)

            writer.writerow([
                os.path.basename(best.candidate), os.path.basename(best.original), best.decision.verdict, best.decision.reason,
                f"{1.0 - best.scores.ham64:.4f}", f"{1.0 - best.scores.color_bhat:.4f}", f"{best.scores.layout_corr:.4f}",
                f"{1.0 - best.scores.strip_dtw_cost:.4f}", best.scores.key_inliers, f"{best.scores.key_inlier_ratio:.4f}",
                f"{best.scores.templ_ncc:.4f}", f"{best.scores.composite():.4f}",
            ])

    print(f"[✓] 완료 — CSV: {csv_path}")
    print(f"[✓] 시각화 이미지는 {out_dir}/annotated_*.jpg 로 저장되었습니다.")


def parse_args():
    p = argparse.ArgumentParser(description="Non‑DL image plagiarism detector for cosmetics detail pages")
    p.add_argument("--originals", default="/HDD/github/similarity_inspection/assets/original")
    p.add_argument("--candidates", default='/HDD/github/similarity_inspection/assets/reseller')
    p.add_argument("--out", default='/HDD/github/similarity_inspection/outputs')
    p.add_argument("--jobs", type=int, default=1, help="병렬 처리(미사용: 순차)")
    # 고급 튜닝 옵션(필요시 사용)
    p.add_argument("--strip-width", type=int, default=512)
    p.add_argument("--strip-height", type=int, default=128)
    p.add_argument("--strip-overlap", type=float, default=0.5)
    p.add_argument("--hash-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    process_all(args.originals, args.candidates, args.out, jobs=args.jobs)


if __name__ == "__main__":
    main()
