import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import os.path as osp
from kornia_moons.viz import draw_LAF_matches


class LoFTR:
    def __init__(self, output_dir, pretrained='outdoor', target_height=1024):
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        self._pretrained = pretrained
        self._target_height = target_height
        self._matcher = KF.LoFTR(pretrained=self._pretrained)
        self._outputs = {'mkpts0': None, 'mkpts1': None, 'inliers': None}
    
    def _resize_keep_aspect(self, img, target_height=1024):
        """높이를 기준으로 비율 유지 리사이즈"""
        _, _, h, w = img.shape
        scale = target_height / h
        new_w = int(w * scale)
        resized = K.geometry.resize(img, (target_height, new_w), antialias=True)
        return resized, scale
        
    def run(self, fname1, fname2, threshold=0.3, ransac=True, verbose=True):

        self._img1 = K.io.load_image(fname1, K.io.ImageLoadType.RGB32)[None, ...]
        self._img2 = K.io.load_image(fname2, K.io.ImageLoadType.RGB32)[None, ...]
        
        img1_resized, self._scale1 = self._resize_keep_aspect(self._img1, target_height=self._target_height)
        img2_resized, self._scale2 = self._resize_keep_aspect(self._img2, target_height=self._target_height)

        input_dict = {
            "image0": K.color.rgb_to_grayscale(img1_resized),
            "image1": K.color.rgb_to_grayscale(img2_resized),
        }

        with torch.inference_mode():
            results = self._matcher(input_dict)

        if verbose:
            for k, v in results.items():
                print(k)
                
        matches = {k: v.cpu().numpy() for k, v in results.items()}
        conf = matches['confidence']  # shape: (N, )
        good_idx = conf > threshold  # 0.7 이상인 매칭만 필터링
        kpts0 = matches['keypoints0'][good_idx]
        kpts1 = matches['keypoints1'][good_idx]
        # -----------------------
        # 4. 좌표 원본 비율로 복원
        # -----------------------
        self._outputs['mkpts0'] = kpts0 / self._scale1
        self._outputs['mkpts1'] = kpts1 / self._scale2

        if ransac:
            Fm, inliers = cv2.findFundamentalMat(self._outputs['mkpts0'], self._outputs['mkpts1'], cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
            self._outputs['inliers'] = inliers > 0
        else:
            self._outputs['inliers'] = np.array([1]*len(mkpts0))
        
        return self._outputs
                
    def visualize(self):

        fig = draw_LAF_matches(
            KF.laf_from_center_scale_ori(
                torch.from_numpy(self._outputs['mkpts0']).view(1, -1, 2),
                torch.ones(self._outputs['mkpts0'].shape[0]).view(1, -1, 1, 1),
                torch.ones(self._outputs['mkpts0'].shape[0]).view(1, -1, 1),
            ),
            KF.laf_from_center_scale_ori(
                torch.from_numpy(self._outputs['mkpts1']).view(1, -1, 2),
                torch.ones(self._outputs['mkpts1'].shape[0]).view(1, -1, 1, 1),
                torch.ones(self._outputs['mkpts1'].shape[0]).view(1, -1, 1),
            ),
            torch.arange(self._outputs['mkpts0'].shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(self._img1),
            K.tensor_to_image(self._img2),
            self._outputs['inliers'],
            draw_dict={
                "inlier_color": (0.2, 1, 0.2),
                "tentative_color": None,
                "feature_color": (0.2, 0.5, 1),
                "vertical": False,
                'thickness': 0.01,
                'radius': 0.01
            },
        )

        fig = plt.gcf()
        fig.savefig(osp.join(self._output_dir, "loftr_laf_matches_fixed.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Result saved as loftr_laf_matches_fixed.png")


