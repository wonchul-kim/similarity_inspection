"""
Image Plagiarism Detector (Non-deep approach)

- 목적: 길게 스크롤된 상세페이지(매우 긴 이미지)에서 원본 이미지의 일부가
  리셀러 이미지에 무단 사용되었는지 고정밀(법적 증거 수준)에 가깝게 판단
  하기 위한 도구입니다.

- 핵심 아이디어 (요약):
  1) coarse-to-fine 전략: 먼저 축소판에서 후보영역을 찾고, 후보영역만 원본
     해상도로 정밀검사합니다.
  2) 특징점(주로 SIFT; 없으면 AKAZE/ORB) 기반 매칭 + Lowe ratio test
  3) DBSCAN으로 리셀러 상의 매칭 점군을 군집화하여 후보 영역 추출
  4) RANSAC으로 호모그래피 추정 -> inlier 기반 정량검증
  5) 호모그래피로 원본을 리셀러 좌표계로 warp한 뒤 SSIM / NCC로 정밀
     유사도(픽셀 레벨) 확인
  6) 여러 지표(인라이어 개수, 인라이어 비율, 재투영 오차, SSIM/NCC)를 조합
     해서 도용 여부 판정

- 주요 장점: 딥러닝을 사용하지 않으면서도 크기/잘림/부분 포함에 강건하고,
  정량 · 시각적 증거(매칭 선, 바운딩 박스, 유사도 점수)를 남길 수 있어
  법적 제출 자료로 적합합니다.

Dependencies:
  - OpenCV (cv2) (권장: opencv-contrib-python)
  - numpy
  - scikit-learn (DBSCAN)
  - scikit-image (optional: structural_similarity)
  - matplotlib (optional: 시각화 저장)

사용 예:
  python image_plagiarism_detector.py --original original.jpg --reseller reseller.jpg --out_dir out

참고: 긴 이미지(예: 높이 10000px 이상)를 다루는 경우 코드는 자동으로 축소판
(crude)에서 후보를 찾은 뒤 풀 해상도로 정밀검사합니다. 필요하면 tile 기반
검사로 더 확장할 수 있습니다.
"""

import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import json
import argparse

# Optional imports
try:
    from skimage.metrics import structural_similarity as compare_ssim
    _HAS_SSIM = True
except Exception:
    _HAS_SSIM = False

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def resize_to_max_dim(img, max_dim=2000):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0
    scale = max_dim / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def create_feature_detector(prefer='SIFT'):
    # Try SIFT -> AKAZE -> ORB
    if prefer == 'SIFT':
        try:
            return cv2.SIFT_create()
        except Exception:
            pass
    try:
        return cv2.AKAZE_create()
    except Exception:
        pass
    # fallback
    return cv2.ORB_create(nfeatures=5000)


def detect_and_compute(detector, img):
    kp, des = detector.detectAndCompute(img, None)
    if kp is None:
        kp = []
    return kp, des


def match_descriptors(des1, des2, ratio_thresh=0.75):
    # des1, des2 can be None
    if des1 is None or des2 is None:
        return []

    # Choose matcher depending on descriptor type
    # AKAZE/ORB -> binary (uint8); SIFT -> float32
    use_flann = False
    if des1.dtype == np.float32 or des2.dtype == np.float32:
        use_flann = True

    matches = []
    if use_flann:
        # FLANN params for SIFT-like descriptors
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = flann.knnMatch(des1, des2, k=2)
        for m_n in raw_matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                matches.append(m)
    else:
        # BFMatcher with Hamming for binary descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = bf.knnMatch(des1, des2, k=2)
        for m_n in raw_matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                matches.append(m)
    return matches


def cluster_matches_by_reseller_kp(kp_reseller, matches, reseller_h, eps_px=None, min_samples=4):
    if len(matches) == 0:
        return []

    pts = np.array([kp_reseller[m.trainIdx].pt for m in matches])  # Nx2

    # eps default: proportion of reseller height
    if eps_px is None:
        eps_px = max(30, int(reseller_h * 0.01))

    clustering = DBSCAN(eps=eps_px, min_samples=min_samples).fit(pts)
    labels = clustering.labels_

    clusters = []
    for lab in set(labels):
        if lab == -1:
            continue
        idxs = np.where(labels == lab)[0]
        cluster_matches = [matches[i] for i in idxs]
        clusters.append(cluster_matches)
    # sort by size desc
    clusters.sort(key=lambda x: len(x), reverse=True)
    return clusters


def estimate_and_verify_homography(kp_orig, kp_res, matches, min_inliers=20, reproj_thresh=5.0):
    if len(matches) < 4:
        return None

    src_pts = np.float32([kp_orig[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_res[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    if H is None or mask is None:
        return None

    mask = mask.ravel().astype(bool)
    inliers = np.sum(mask)
    matches_mask = mask.tolist()

    return {
        'H': H,
        'inliers': int(inliers),
        'mask': matches_mask,
        'total_matches': len(matches),
        'src_pts': src_pts,
        'dst_pts': dst_pts,
        'inlier_idx': np.where(mask)[0].tolist()
    }


def crop_bbox_from_points(pts):
    # pts: Nx2 array
    xs = pts[:, 0]
    ys = pts[:, 1]
    xmin = int(np.floor(xs.min()))
    xmax = int(np.ceil(xs.max()))
    ymin = int(np.floor(ys.min()))
    ymax = int(np.ceil(ys.max()))
    return xmin, ymin, xmax, ymax


def compare_region_ssim_or_ncc(orig_full, reseller_full, H, bbox, use_ssim=_HAS_SSIM):
    h2, w2 = reseller_full.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(w2, xmax); ymax = min(h2, ymax)
    if xmin >= xmax or ymin >= ymax:
        return {'ssim': 0.0, 'ncc': 0.0}

    # warp whole original to reseller coordinate system (full res)
    warped = cv2.warpPerspective(orig_full, H, (w2, h2), flags=cv2.INTER_LINEAR)

    roi_res = reseller_full[ymin:ymax, xmin:xmax]
    roi_warp = warped[ymin:ymax, xmin:xmax]

    if roi_res.size == 0 or roi_warp.size == 0:
        return {'ssim': 0.0, 'ncc': 0.0}

    # Ensure uint8
    roi_res_u = cv2.normalize(roi_res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    roi_warp_u = cv2.normalize(roi_warp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    res = {'ssim': 0.0, 'ncc': 0.0}
    # SSIM if available
    if use_ssim:
        try:
            s = compare_ssim(roi_res_u, roi_warp_u)
            res['ssim'] = float(s)
        except Exception:
            res['ssim'] = 0.0

    # NCC via matchTemplate (both images same size -> single value)
    try:
        # convert to BGR if needed
        if roi_warp_u.shape != roi_res_u.shape:
            # fallback to resize
            roi_warp_u = cv2.resize(roi_warp_u, (roi_res_u.shape[1], roi_res_u.shape[0]))
        ncc_val = cv2.matchTemplate(roi_res_u, roi_warp_u, cv2.TM_CCOEFF_NORMED)
        # result is single value if template == image
        if ncc_val is None:
            res['ncc'] = 0.0
        else:
            if isinstance(ncc_val, np.ndarray):
                res['ncc'] = float(ncc_val.max())
            else:
                res['ncc'] = float(ncc_val)
    except Exception:
        res['ncc'] = 0.0

    return res


def draw_evidence_visualization(orig_full, reseller_full, kp1, kp2, matches, homography_res, out_path):
    # Draw only inlier matches
    if homography_res is None:
        img = cv2.drawMatches(orig_full, kp1, reseller_full, kp2, matches[:50], None, flags=2)
        cv2.imwrite(out_path, img)
        return out_path

    inlier_idx = homography_res['inlier_idx']
    inlier_matches = [matches[i] for i in inlier_idx]
    img = cv2.drawMatches(orig_full, kp1, reseller_full, kp2, inlier_matches, None, flags=2)

    # draw projected corners
    h1, w1 = orig_full.shape[:2]
    corners = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    H = homography_res['H']
    h2, w2 = reseller_full.shape[:2]
    try:
        projected = cv2.perspectiveTransform(corners, H)
        # projected in reseller coords, draw polygon on second half of concatenated image
        # draw on the right-half offset (since drawMatches concatenates images horizontally)
        offset = orig_full.shape[1]
        img_color = img.copy()
        pts = projected.reshape(-1,2)
        pts_off = np.array([[int(x+offset),int(y)] for (x,y) in pts])
        cv2.polylines(img_color, [pts_off], isClosed=True, color=(0,255,0), thickness=3)
        cv2.imwrite(out_path, img_color)
    except Exception:
        cv2.imwrite(out_path, img)
    return out_path


def analyze_images(original_path, reseller_path, params=None):
    if params is None:
        params = {}
    # default params
    max_coarse_dim = params.get('max_coarse_dim', 1600)
    ratio_thresh = params.get('ratio_thresh', 0.75)
    eps_px = params.get('eps_px', None)
    min_cluster_matches = params.get('min_cluster_matches', 6)
    min_inliers = params.get('min_inliers', 25)
    min_inlier_ratio = params.get('min_inlier_ratio', 0.20)
    reproj_thresh = params.get('reproj_thresh', 5.0)
    min_ssim = params.get('min_ssim', 0.55)
    min_ncc = params.get('min_ncc', 0.60)

    # load full-res images
    orig_full = load_gray(original_path)
    res_full = load_gray(reseller_path)

    h_res_full, w_res_full = res_full.shape[:2]

    # create coarse (downscaled) images for initial candidate search
    orig_coarse, scale_o = resize_to_max_dim(orig_full, max_dim=max_coarse_dim)
    res_coarse, scale_r = resize_to_max_dim(res_full, max_dim=max_coarse_dim)

    detector = create_feature_detector('SIFT')

    kp1_c, des1_c = detect_and_compute(detector, orig_coarse)
    kp2_c, des2_c = detect_and_compute(detector, res_coarse)

    print(f"coarse kp orig: {len(kp1_c)}, reseller: {len(kp2_c)}")

    matches_c = match_descriptors(des1_c, des2_c, ratio_thresh=ratio_thresh)
    print(f"coarse raw good matches: {len(matches_c)}")

    if len(matches_c) < min_cluster_matches:
        return {
            'plagiarism': False,
            'reason': 'not_enough_coarse_matches',
            'coarse_matches': len(matches_c)
        }

    # Map coarse matches back to full-res keypoints
    # But kp coordinates are in coarse scale; convert to full-res coords
    # For kp1: multiply by (1/scale_o); for kp2: multiply by (1/scale_r)
    kp1_full_est = [
                    cv2.KeyPoint(x=kp.pt[0]/scale_o, y=kp.pt[1]/scale_o, size=kp.size/scale_o)
                    for kp in kp1_c
                ]

    # Use cluster on reseller keypoints (coarse) to get candidate regions (coarse coords -> adapt to full-res later)
    clusters = cluster_matches_by_reseller_kp(kp2_c, matches_c, reseller_h=res_coarse.shape[0], eps_px=eps_px, min_samples=4)
    if len(clusters) == 0:
        return {
            'plagiarism': False,
            'reason': 'no_clusters_found',
            'coarse_matches': len(matches_c)
        }

    results = []
    # For each cluster, perform full-res precise matching and homography
    for cluster_matches in clusters:
        print(f"cluster size (coarse): {len(cluster_matches)}")
        # Build descriptor match lists at full-res: need to compute keypoints/descriptors at full-res for the region
        # Simpler approach: compute features on full-res images globally (can be slower) but more accurate.

        # compute full-res keypoints/descriptors if not yet computed
        detector_full = create_feature_detector('SIFT')
        kp1_full, des1_full = detect_and_compute(detector_full, orig_full)
        kp2_full, des2_full = detect_and_compute(detector_full, res_full)

        # match full-res descriptors
        matches_full = match_descriptors(des1_full, des2_full, ratio_thresh=ratio_thresh)
        print(f"full-res good matches: {len(matches_full)}")

        # cluster matches on reseller full-res KP
        clusters_full = cluster_matches_by_reseller_kp(kp2_full, matches_full, reseller_h=res_full.shape[0], eps_px=None, min_samples=4)
        if len(clusters_full) == 0:
            continue

        # examine the largest cluster(s)
        for i, cl in enumerate(clusters_full[:3]):
            print(f"Examining cluster {i} size: {len(cl)}")
            if len(cl) < min_cluster_matches:
                continue
            hom_res = estimate_and_verify_homography(kp1_full, kp2_full, cl, min_inliers=min_inliers, reproj_thresh=reproj_thresh)
            if hom_res is None:
                continue

            # scale checks and compute bbox
            h1, w1 = orig_full.shape[:2]
            corners = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
            projected = cv2.perspectiveTransform(corners, hom_res['H']).reshape(-1,2)
            xmin, ymin, xmax, ymax = crop_bbox_from_points(projected)

            # compute SSIM/NCC on the projected region
            sim = compare_region_ssim_or_ncc(orig_full, res_full, hom_res['H'], (xmin, ymin, xmax, ymax), use_ssim=_HAS_SSIM)

            # reprojection error for inliers (mean)
            inlier_idxs = hom_res['inlier_idx']
            src_in = np.squeeze(hom_res['src_pts'][inlier_idxs], axis=1)
            dst_in = np.squeeze(hom_res['dst_pts'][inlier_idxs], axis=1)
            # project src_in by H
            src_in_h = cv2.perspectiveTransform(src_in.reshape(-1,1,2), hom_res['H']).reshape(-1,2)
            reproj_errs = np.linalg.norm(src_in_h - dst_in, axis=1)
            mean_reproj = float(np.mean(reproj_errs)) if len(reproj_errs)>0 else float('inf')

            inlier_count = hom_res['inliers']
            inlier_ratio = inlier_count / float(hom_res['total_matches']) if hom_res['total_matches']>0 else 0.0

            result = {
                'cluster_size': len(cl),
                'inlier_count': inlier_count,
                'inlier_ratio': inlier_ratio,
                'mean_reproj_error': mean_reproj,
                'ssim': sim.get('ssim', 0.0),
                'ncc': sim.get('ncc', 0.0),
                'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)],
                'H': hom_res['H'].tolist()
            }

            # decision rule: conservative defaults (tunable)
            passed = (
                (inlier_count >= min_inliers) and
                (inlier_ratio >= min_inlier_ratio) and
                (mean_reproj <= max(8.0, reproj_thresh*2)) and
                (result['ncc'] >= min_ncc or result['ssim'] >= min_ssim)
            )
            result['plagiarism_decision'] = bool(passed)

            results.append(result)

            if passed:
                # prepare visualization file path
                vis_path = os.path.splitext(reseller_path)[0] + '_evidence.png'
                draw_evidence_visualization(orig_full, res_full, kp1_full, kp2_full, matches_full, hom_res, vis_path)
                result['evidence_image'] = vis_path
                # stop after first positive
                return {
                    'plagiarism': True,
                    'reason': 'match_found',
                    'details': result,
                    'all_candidates': results
                }

    # if no candidate passed
    return {
        'plagiarism': False,
        'reason': 'no_candidate_passed',
        'candidates': results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', default='/HDD/github/similarity_inspection/assets/original/PDP_(EC) 111174852_1.jpg')
    parser.add_argument('--reseller', default='/HDD/github/similarity_inspection/assets/reseller/f622788b0a42dcd2f5f57069e7bb6b9fbf5e1c1f_fullpage.png')
    parser.add_argument('--out_dir', default='outputs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    res = analyze_images(args.original, args.reseller)
    out_json = os.path.join(args.out_dir, 'analysis_result.json')
    with open(out_json, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Saved result -> {out_json}")
    if 'details' in res and 'evidence_image' in res['details']:
        print(f"Evidence image: {res['details']['evidence_image']}")


if __name__ == '__main__':
    main()