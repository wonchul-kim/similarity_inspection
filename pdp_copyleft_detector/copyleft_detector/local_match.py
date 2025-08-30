import torch, cv2, numpy as np
from .utils import imresize_max_dim

class LoFTRMatcher:
    """
    LoFTR feature matcher via kornia. Returns inlier ratio (matches per keypoints proxy).
    """
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() or device=="cpu" else "cpu")
        try:
            from kornia.feature import LoFTR
            self.matcher = LoFTR(pretrained="outdoor").to(self.device).eval()
            self.ok = True
        except Exception as e:
            print("[WARN] LoFTR not available:", e)
            self.matcher = None; self.ok = False

    @torch.no_grad()
    def match_ratio(self, img0_bgr, img1_bgr, max_size=1200):
        if not self.ok:
            return 0.0, (None, None, None)
        # img0, s0 = imresize_max_dim(img0_bgr, max_dim=max_size)
        # img1, s1 = imresize_max_dim(img1_bgr, max_dim=max_size)
        img0, (sx0, sy0) = imresize_max_dim(img0_bgr, max_dim=max_size)
        img1, (sx1, sy1) = imresize_max_dim(img1_bgr, max_dim=max_size)

        g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        import torch
        def to_tensor(gray):
            t = torch.from_numpy(gray)[None,None].float()/255.0
            return t.to(self.device)
        d = {"image0": to_tensor(g0), "image1": to_tensor(g1)}
        out = self.matcher(d)
        m0 = out["keypoints0"].cpu().numpy() if "keypoints0" in out else np.zeros((0,2))
        m1 = out["keypoints1"].cpu().numpy() if "keypoints1" in out else np.zeros((0,2))
        conf = out["confidence"].cpu().numpy() if "confidence" in out else np.zeros((0,))
        n_match = (conf>0.5).sum() if conf.size>0 else m0.shape[0]
        denom = max((g0.size+g1.size)/2.0/500.0, 1.0)
        ratio = float(n_match/denom)
        return ratio, (m0, m1, conf, (sx0, sy0), (sx1, sy1))

