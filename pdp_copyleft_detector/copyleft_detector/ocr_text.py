import easyocr, numpy as np
from sentence_transformers import SentenceTransformer
import torch

class OCRTextEncoder:
    def __init__(self, langs=["ko","en"], sentence_model="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        self.reader = easyocr.Reader(langs, gpu=(torch.cuda.is_available() and device=="cuda"))
        self.sent_model = SentenceTransformer(sentence_model, device=device)

    def extract_text(self, img_bgr, max_words=256):
        result = self.reader.readtext(img_bgr, detail=1, paragraph=False)
        texts = [r[1] for r in result if isinstance(r, (list,tuple)) and len(r)>=2]
        if len(texts)>max_words:
            texts = texts[:max_words]
        joined = " ".join(texts).strip()
        return joined

    def embed_text(self, text: str):
        if not text:
            return None
        vec = self.sent_model.encode([text], convert_to_tensor=True, normalize_embeddings=True)
        return vec[0].detach().cpu().numpy()

    def semantic_sim(self, img0_bgr, img1_bgr, max_words=256):
        t0 = self.extract_text(img0_bgr, max_words=max_words)
        t1 = self.extract_text(img1_bgr, max_words=max_words)
        if not t0 or not t1:
            return 0.0, (t0, t1)
        e0 = self.embed_text(t0)
        e1 = self.embed_text(t1)
        sim = float((e0*e1).sum())
        return sim, (t0, t1)
