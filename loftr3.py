import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches

# -----------------------
# 1. 이미지 로드
# -----------------------
# fname1 = "/HDD/github/similarity_inspection/assets/original/PDP_(EC) 111174852_1.jpg"
fname1 = "/HDD/github/similarity_inspection/assets/original/PDP_(EC) 111174852_2.jpg"
# fname1 = "/HDD/github/similarity_inspection/assets/original/PDP_(EC) 111174855_1.jpg"
# fname1 = "/HDD/github/similarity_inspection/assets/original/PDP_(EC) 111174855_2.jpg"
# fname2 = "/HDD/github/similarity_inspection/assets/reseller/1b3ed56a3f2f0ab42ab910c76143e95151062427_fullpage.png"
fname2 = "/HDD/github/similarity_inspection/assets/reseller/f622788b0a42dcd2f5f57069e7bb6b9fbf5e1c1f_fullpage.png"
# fname1 = "/HDD/github/similarity_inspection/assets/etc/3.png"
# fname2 = "/HDD/github/similarity_inspection/assets/etc/4.png"


img1 = K.io.load_image(fname1, K.io.ImageLoadType.RGB32)[None, ...]
img2 = K.io.load_image(fname2, K.io.ImageLoadType.RGB32)[None, ...]

# -----------------------
# 2. 비율 유지 리사이즈
# -----------------------
def resize_keep_aspect(img, target_height=600):
    """높이를 기준으로 비율 유지 리사이즈"""
    _, _, h, w = img.shape
    scale = target_height / h
    new_w = int(w * scale)
    resized = K.geometry.resize(img, (target_height, new_w), antialias=True)
    return resized, scale

img1_resized, scale1 = resize_keep_aspect(img1, target_height=4096)
img2_resized, scale2 = resize_keep_aspect(img2, target_height=4096)

# -----------------------
# 3. LoFTR 매칭
# -----------------------
matcher = KF.LoFTR(pretrained="outdoor")
input_dict = {
    "image0": K.color.rgb_to_grayscale(img1_resized),
    "image1": K.color.rgb_to_grayscale(img2_resized),
}

with torch.inference_mode():
    correspondences = matcher(input_dict)

for k, v in correspondences.items():
    print(k)


# matches = {k: v.cpu().numpy() for k, v in data.items()}
# conf = matches['confidence']  # shape: (N, )
# good_idx = conf > 0.7  # 0.7 이상인 매칭만 필터링
# kpts0 = matches['keypoints0'][good_idx]
# kpts1 = matches['keypoints1'][good_idx]
# -----------------------
# 4. 좌표 원본 비율로 복원
# -----------------------
mkpts0 = correspondences["keypoints0"].cpu().numpy() / scale1
mkpts1 = correspondences["keypoints1"].cpu().numpy() / scale2

# -----------------------
# 5. RANSAC으로 인라이어 필터링
# -----------------------
Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0
# inliers = np.array([1]*len(mkpts0))

# -----------------------
# 6. 시각화
# -----------------------
fig = draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts0).view(1, -1, 2),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts1).view(1, -1, 2),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
    ),
    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    inliers,
    draw_dict={
        "inlier_color": (0.2, 1, 0.2),
        "tentative_color": None,
        "feature_color": (0.2, 0.5, 1),
        "vertical": False,
        'thickness': 0.01,
        'radius': 0.01
    },
)

# -----------------------
# 7. 이미지 저장
# -----------------------
fig = plt.gcf()
fig.savefig("loftr_laf_matches_fixed.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Result saved as loftr_laf_matches_fixed.png")
