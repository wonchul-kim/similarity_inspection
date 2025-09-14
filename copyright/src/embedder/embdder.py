import torch, torch.nn as nn
import timm
import open_clip
import torchvision.transforms as T
from PIL import Image
import numpy as np

class VisualEmbedder(nn.Module):
    """
    Visual embedder using DINOv2 ViT-L/14 or CLIP ViT-L/14.
    Returns L2-normalized embeddings by default.
    """
    def __init__(self, name="dino_vitl14", device="cuda", input_size=518, normalize=True):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() or device=="cpu" else "cpu")
        self.normalize = normalize
        self.input_size = input_size
        if "dino2" in name:
            if 'vigl14' in name:
                self.model = timm.create_model("vit_giant_patch14_reg4_dinov2.lvd142m", pretrained=True)
            elif 'vitl14' in name:
                self.model = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True)
            elif 'vitb14':
                self.model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True)
            elif 'vits14':
                self.model = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", pretrained=True)
            else:
                raise NotImplementedError(f"There is no such embedder: {name}")
            
            self.out_dim = self.model.num_features
            self.forward_fn = self._forward_timm
            mean = (0.485,0.456,0.406); std=(0.229,0.224,0.225)
            self.transform = T.Compose([
                T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean, std)
            ])
        elif "dinov3" in name:
            if 'vigl14' in name:
                model_id = "facebook/dinov3-vitgiant14-pretrain-lvd1689m"
            elif 'vitl14' in name:
                model_id = "facebook/dinov3-vitlarge14-pretrain-lvd1689m"
            elif 'vitb16' in name:
                model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
            elif 'vits14' in name:
                model_id = "facebook/dinov3-vitsmall14-pretrain-lvd1689m"
            else:
                raise NotImplementedError(f"There is no such DINOv3 embedder: {name}")

            from transformers import AutoImageProcessor, Dinov3Model

            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = Dinov3Model.from_pretrained(model_id)
            self.out_dim = self.model.config.hidden_size
            self.forward_fn = self._forward_hf

            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            self.transform = T.Compose([
                T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
            
        elif name == "clip_vitl14":
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
            self.model = model
            self.out_dim = model.visual.output_dim
            self.forward_fn = self._forward_clip
            self.transform = preprocess
        else:
            raise ValueError("Unknown backbone")
        self.eval().to(self.device)

    @torch.no_grad()
    def embed_pil(self, pil: Image.Image):
        x = self.transform(pil).unsqueeze(0).to(self.device)
        feat = self.forward_fn(x)
        if self.normalize:
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed_np_bgr(self, np_bgr):
        pil = Image.fromarray(np_bgr[:,:,::-1])
        return self.embed_pil(pil)

    def _forward_timm(self, x):
        return self.model.forward_features(x).mean(dim=1)

    def _forward_clip(self, x):
        return self.model.encode_image(x)
