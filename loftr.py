import cv2
import torch
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. Load and preprocess images
# -------------------------------
def load_and_preprocess(path, target_size=(640, 1024)):
    """이미지를 불러와서 비율 유지 리사이즈 + 패딩 후 tensor 변환"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = img.shape[:2]

    target_h, target_w = target_size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    padded = cv2.copyMakeBorder(
        resized,
        pad_h // 2, pad_h - pad_h // 2,
        pad_w // 2, pad_w - pad_w // 2,
        cv2.BORDER_CONSTANT, value=0
    )

    tensor = K.image_to_tensor(padded, False).float() / 255.0
    meta = {
        "orig_size": (orig_h, orig_w),
        "scale": scale,
        "pad": (pad_w // 2, pad_h // 2)
    }
    return tensor, meta

# 경로 지정
img1_tensor, meta1 = load_and_preprocess("/HDD/github/similarity_inspection/assets/etc/1.jpg")
img2_tensor, meta2 = load_and_preprocess("/HDD/github/similarity_inspection/assets/etc/2.jpg")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
img1_tensor, img2_tensor = img1_tensor.to(device), img2_tensor.to(device)

# -------------------------------
# 2. Load LoFTR model
# -------------------------------
matcher = KF.LoFTR(pretrained="outdoor").to(device)
matcher.eval()

# -------------------------------
# 3. Run matching
# -------------------------------
with torch.no_grad():
    input_dict = {"image0": img1_tensor, "image1": img2_tensor}
    correspondences = matcher(input_dict)

mkpts1 = correspondences["keypoints0"].cpu().numpy()
mkpts2 = correspondences["keypoints1"].cpu().numpy()

# -------------------------------
# 4. Map matched points back to original coordinates
# -------------------------------
def recover_original_coords(points, meta):
    """패딩과 스케일을 고려하여 원본 좌표로 복원"""
    pad_x, pad_y = meta["pad"]
    scale = meta["scale"]
    orig_h, orig_w = meta["orig_size"]

    points[:, 0] = (points[:, 0] - pad_x) / scale
    points[:, 1] = (points[:, 1] - pad_y) / scale

    # 범위를 벗어나는 좌표는 클리핑
    points[:, 0] = np.clip(points[:, 0], 0, orig_w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, orig_h - 1)
    return points

mkpts1_orig = recover_original_coords(mkpts1.copy(), meta1)
mkpts2_orig = recover_original_coords(mkpts2.copy(), meta2)

print(f"Number of matches: {len(mkpts1_orig)}")

# -------------------------------
# 5. Visualization
# -------------------------------
img1_vis = cv2.imread("/HDD/github/similarity_inspection/assets/etc/1.jpg")
img2_vis = cv2.imread("/HDD/github/similarity_inspection/assets/etc/2.jpg")

def draw_matches(img1, img2, pts1, pts2):
    """Draw matched keypoints between two images"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1+w2] = img2
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, (int(x1), int(y1)), 3, color, -1)
        cv2.circle(canvas, (int(x2) + w1, int(y2)), 3, color, -1)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2) + w1, int(y2)), color, 1)
    return canvas

match_vis = draw_matches(img1_vis, img2_vis, mkpts1_orig, mkpts2_orig)
cv2.imwrite("loftr_matches_fixed.jpg", match_vis)
plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
