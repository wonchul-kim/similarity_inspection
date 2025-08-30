"""
B단계: 리셀러 페이지 인제스트 파이프라인 (단일 스크립트)

기능
- Playwright로 리셀러 URL 렌더링(봇 차단 완화용 헤더/UA/뷰포트 설정)
- 전체 페이지 스크린샷(풀페이지) 저장
- 모든 이미지 수집
  • <img src/srcset>, <picture> 소스
  • CSS background-image(url(...)) 전부
- 이미지 다운로드(상대경로→절대경로 해석, data:URL 처리)
- 각 이미지(+풀페이지 스크린샷)에 대해:
  • 트림(여백 제거) + 정규화
  • 멀티스케일 피라미드(0.5×, 1×, 1.5×, 2×; 최솟값 ≥ TILE_SIZE)
  • 타일링(512×512, stride=224)
  • 타일 특징: pHash/aHash/dHash, HOG+HSV 임베딩, ORB, 텍스트/단색띠 마스크
- 산출물: /reseller_pack/{job_id}/raw|screens|tiles|features|visuals + 메타 JSON

필요 패키지: playwright, requests, opencv-python, pillow, numpy, scikit-image
사전 준비: `playwright install` (크로미움 런타임 설치)
"""

import os, re, io, sys, time, json, math, base64, hashlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Iterable
from urllib.parse import urljoin, urlparse

import numpy as np
import cv2
import requests
from PIL import Image
from skimage.feature import hog
from playwright.sync_api import sync_playwright

from functionals import *
from tqdm import tqdm



# ---------------- I/O ROOT ----------------
def make_job_root(base="/mnt/data/reseller_pack", tag=None):
    import datetime, uuid
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    job_id = f"{ts}_{uuid.uuid4().hex[:8]}" if not tag else f"{ts}_{tag}"
    root = os.path.join(base, job_id)
    for d in ["raw", "screens", "tiles", "features", "visuals"]:
        os.makedirs(os.path.join(root, d), exist_ok=True)
    return root

# ---------------- UTIL ----------------
def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()



def detect_solid_band_mask(bgr: np.ndarray, win=25, var_thr=8.0) -> np.ndarray:
    """흰색/단색 띠(상하/좌우 패딩) 마스크: 로우/컬럼 분산이 매우 작은 넓은 구간"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h,w = gray.shape
    row_var = cv2.boxFilter(gray**2, -1, (win,1)) - cv2.boxFilter(gray, -1, (win,1))**2
    col_var = cv2.boxFilter(gray**2, -1, (1,win)) - cv2.boxFilter(gray, -1, (1,win))**2
    mask = ((row_var < var_thr) | (col_var < var_thr)).astype(np.uint8)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), 1)
    return mask

def text_and_band_masks(tile_cv: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(tile_cv, cv2.COLOR_BGR2GRAY)
    tmask = estimate_text_mask(gray)
    bmask = detect_solid_band_mask(tile_cv)
    return cv2.bitwise_or(tmask, bmask)

# ---------------- 다운로드/수집 ----------------
def resolve_and_filter_urls(base_url: str, urls: Iterable[str]) -> List[str]:
    outs = []
    for u in urls:
        if not u: continue
        u = u.strip().strip('"').strip("'")
        if u.startswith("data:image/"): outs.append(u); continue
        absu = urljoin(base_url, u)
        # 일반적으로 tracking gif는 제외(1x1)
        outs.append(absu)
    # dedup preserving order
    seen=set(); res=[]
    for u in outs:
        if u not in seen:
            seen.add(u); res.append(u)
    return res

def gather_page_image_urls(page) -> List[str]:
    js = r"""
(() => {
  const set = new Set();
  // <img> + srcset
  document.querySelectorAll('img').forEach(img => {
    if (img.currentSrc) set.add(img.currentSrc);
    if (img.src) set.add(img.src);
    if (img.srcset) {
      // pick largest descriptor
      const parts = img.srcset.split(',').map(s=>s.trim());
      let best=null, bestW=0;
      for (const p of parts) {
        const m = p.match(/(\S+)\s+(\d+)(w|x)/);
        if (m) {
          const url=m[1], val=parseInt(m[2],10);
          const w = m[3]==='x' ? val*1000 : val;
          if (w>bestW) {best=url; bestW=w;}
        } else {
          const tok=p.split(/\s+/)[0];
          if (tok) best = tok;
        }
      }
      if (best) set.add(best);
    }
  });
  // CSS background-image
  const all = document.querySelectorAll('*');
  for (const el of all) {
    const bg = getComputedStyle(el).getPropertyValue('background-image');
    if (bg && bg !== 'none') {
      const matches = bg.match(/url\((.*?)\)/g);
      if (matches) {
        for (const m of matches) {
          let url = m.replace(/^url\((.*)\)$/,'$1').trim();
          url = url.replace(/^["']|["']$/g, '');
          if (url) set.add(url);
        }
      }
    }
  }
  return Array.from(set);
})()
"""
    raw_urls = page.evaluate(js)
    return resolve_and_filter_urls(page.url, raw_urls)

def download_image(url: str, out_dir: str, session: requests.Session, timeout=20) -> Tuple[str, Dict[str,Any]]:
    meta = {"source": url, "ok": False}
    try:
        if url.startswith("data:image/"):
            header, b64 = url.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
            ext = ".png" if "png" in mime else (".jpg" if "jpeg" in mime or "jpg" in mime else ".bin")
            data = base64.b64decode(b64)
            sha = sha1_bytes(data)
            path = os.path.join(out_dir, f"{sha}{ext}")
            with open(path, "wb") as f: f.write(data)
            meta.update({"ok": True, "path": path, "sha1": sha, "bytes": len(data), "status": "data-url"})
            return path, meta
        resp = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.content
        sha = sha1_bytes(data)
        # guess extension
        ctype = resp.headers.get("Content-Type","").lower()
        ext = ".jpg"
        if "png" in ctype: ext = ".png"
        elif "webp" in ctype: ext = ".webp"
        elif "gif" in ctype: ext = ".gif"
        path = os.path.join(out_dir, f"{sha}{ext}")
        with open(path, "wb") as f: f.write(data)
        meta.update({"ok": True, "path": path, "sha1": sha, "bytes": len(data), "status": resp.status_code})
        return path, meta
    except Exception as e:
        meta.update({"error": str(e)})
        return "", meta

# ---------------- 타일 처리 ----------------
@dataclass
class TileRecord:
    url: str           # 리셀러 페이지 URL
    parent_img: str    # 원본 이미지 파일 경로(또는 fullpage)
    parent_sha1: str
    parent_wh: Tuple[int,int]
    scale: float
    x: int; y: int; w: int; h: int
    phash: str; ahash: str; dhash: str
    orb_kp: int
    text_ratio: float
    band_ratio: float
    embedding_path: str
    orb_path: str
    tile_path: str

def process_image_to_tiles(img_path: str, base_out: str, page_url: str, label: str) -> Tuple[List[TileRecord], Dict[str,Any]]:
    recs: List[TileRecord] = []
    meta = {"input": img_path, "ok": False}
    try:
        # load with PIL to handle webp/gif
        pil = Image.open(img_path).convert("RGB")
        cv = pil_to_cv(pil)
        trim, trim_info = auto_trim_by_corner_color(cv, tol=8.0)
        norm = normalize_image(trim)
        h, w = norm.shape[:2]
        meta.update({"trim": trim_info, "norm_wh": (w,h)})
        # multiscale
        scales = [0.5, 1.0, 1.5, 2.0]
        for s in scales:
            sw, sh = int(w*s), int(h*s)
            if sw < TILE_SIZE or sh < TILE_SIZE: continue
            resized = cv2.resize(norm, (sw, sh), interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC)
            boxes = sliding_tiles(resized, TILE_SIZE, STRIDE)
            for i,(x,y,tw,th) in enumerate(boxes):
                tile = resized[y:y+th, x:x+tw].copy()
                tile_pil = cv_to_pil(tile)

                # masks
                tm = estimate_text_mask(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY))
                bm = detect_solid_band_mask(tile)
                masked = cv2.bitwise_or(tm, bm)
                text_ratio = float((tm>0).sum()/tm.size)
                band_ratio = float((bm>0).sum()/bm.size)

                # features
                a_h = ahash(tile_pil); d_h = dhash(tile_pil); p_h = phash(tile_pil)
                emb  = visual_embedding(tile)
                kps, desc = orb_features(tile)

                # save
                base = f"{hashlib.sha1((img_path+label+str(s)+str(i)).encode()).hexdigest()}"
                tpath = os.path.join(base_out, "tiles", f"{base}.png")
                cv2.imwrite(tpath, tile)
                epath = os.path.join(base_out, "features", f"{base}.emb.npy"); np.save(epath, emb)
                opath = os.path.join(base_out, "features", f"{base}.orb.npy"); np.save(opath, desc)

                recs.append(TileRecord(
                    url=page_url, parent_img=img_path, parent_sha1=sha1_file(img_path),
                    parent_wh=(w,h), scale=float(s),
                    x=int(x), y=int(y), w=int(tw), h=int(th),
                    phash=p_h, ahash=a_h, dhash=d_h,
                    orb_kp=int(len(kps)), text_ratio=text_ratio, band_ratio=band_ratio,
                    embedding_path=epath, orb_path=opath, tile_path=tpath
                ))
        meta["ok"] = True
    except Exception as e:
        meta["error"] = str(e)
    return recs, meta

# ---------------- Playwright 렌더링 ----------------
def render_and_capture(url: str, job_root: str) -> Dict[str,Any]:
    out = {"url": url, "ok": False}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport=VIEWPORT,
            locale="ko-KR",
            device_scale_factor=1.0,
            java_script_enabled=True
        )
        page = context.new_page()
        page.set_default_timeout(TIMEOUT_MS)
        page.goto(url, wait_until="domcontentloaded")
        # 스크롤 다운 & lazy-load 유도
        page.wait_for_timeout(800)
        last_height = 0
        for _ in range(50):  # 최대 50 스텝
            page.mouse.wheel(0, 1200)
            page.wait_for_timeout(350)
            curr = page.evaluate("document.scrollingElement ? document.scrollingElement.scrollHeight : document.body.scrollHeight")
            if curr == last_height: break
            last_height = curr
        # 네트워크 안정화
        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass

        # 풀페이지 스크린샷
        screen_path = os.path.join(job_root, "screens", f"{sha1_bytes(url.encode())}_fullpage.png")
        page.screenshot(path=screen_path, full_page=True)
        out["screenshot_path"] = screen_path

        # 이미지 URL 수집
        urls = gather_page_image_urls(page)
        out["found_image_urls"] = urls

        # 다운로드
        raw_dir = os.path.join(job_root, "raw")
        sess = requests.Session()
        downloaded = []
        for u in urls:
            pth, meta = download_image(u, raw_dir, sess)
            if meta.get("ok"): downloaded.append(meta)
        out["downloaded"] = downloaded
        out["ok"] = True

        context.close(); browser.close()
    return out

# ---------------- 메인 파이프라인 ----------------
def run_b_stage(urls: List[str], output_dir, job_tag=None) -> str:
    root = make_job_root(base=output_dir, tag=job_tag)
    session_meta = {"job_root": root, "urls": urls, "pages": []}

    for url in urls:
        print(f"[B] Processing: {url}")
        page_info = render_and_capture(url, root)
        session_meta["pages"].append(page_info)

        # 풀페이지 스크린샷도 이미지처럼 처리
        tiles_all: List[Dict[str,Any]] = []
        screenshot_path = page_info.get("screenshot_path")
        if screenshot_path and os.path.exists(screenshot_path):
            recs, meta = process_image_to_tiles(screenshot_path, root, url, label="__fullpage__")
            tiles_all.extend([asdict(r) for r in recs])
            page_info["screenshot_process"] = meta

        # 각 다운로드 이미지 처리
        for d in page_info.get("downloaded", []):
            ipath = d.get("path")
            if not ipath or not os.path.exists(ipath): continue
            recs, meta = process_image_to_tiles(ipath, root, url, label="__asset__")
            tiles_all.extend([asdict(r) for r in recs])

        # 페이지 단위 메타/타일 저장
        page_hash = sha1_bytes(url.encode())[:12]
        with open(os.path.join(root, "features", f"tiles_{page_hash}.json"), "w", encoding="utf-8") as f:
            json.dump(tiles_all, f, ensure_ascii=False, indent=2)

    # 세션 메타 저장
    with open(os.path.join(root, "session_meta.json"), "w", encoding="utf-8") as f:
        json.dump(session_meta, f, ensure_ascii=False, indent=2)

    print("[B] Done. Output root:", root)
    return root

# ---------------- CLI ----------------
if __name__ == "__main__":
    """
    사용 예:
    $ python b_stage_ingest.py https://www.11st.co.kr/products/5662267421 https://item.gmarket.co.kr/Item?goodscode=4426684995

    결과물:
      reseller_pack/{job_id}/
        raw/           # 다운로드 이미지
        screens/       # 풀페이지 스크린샷
        tiles/         # 타일 PNG
        features/
          tiles_*.json # 타일 메타(해시/좌표/스케일/마스크 비율/특징 경로)
          *.emb.npy    # 임베딩
          *.orb.npy    # ORB 디스크립터
        visuals/       # (필요시 추가 시각화)
        session_meta.json
    """
    
    # ---------------- CONFIG ----------------
    TILE_SIZE  = 512
    STRIDE     = 224
    MSER_DELTA = 5
    TIMEOUT_MS = 45_000
    VIEWPORT   = {"width": 1365, "height": 900}
    USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
    HEADLESS   = True

    urls = ['https://www.11st.co.kr/products/5662267421',
            'https://item.gmarket.co.kr/Item?goodscode=4426684995']
    run_b_stage(urls, output_dir='/HDD/_projects/github/similarity_inspection/outputs')
