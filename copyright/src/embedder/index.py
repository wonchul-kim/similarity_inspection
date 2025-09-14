import faiss, os, json, numpy as np, tqdm

class EmbeddingIndex:
    def __init__(self, dim, nlist=256, nprobe=16, normalize=True):
        self.dim = dim
        self.normalize = normalize
        self.quantizer = faiss.IndexFlatIP(dim) if normalize else faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(self.quantizer, dim, nlist,
                                        faiss.METRIC_INNER_PRODUCT if normalize else faiss.METRIC_L2)
        self.trained = False
        self.meta = []

    def _norm(self, X):
        if not self.normalize: return X
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norms

    def train(self, X_train):
        X = self._norm(X_train.astype("float32"))
        self.index.train(X); self.trained = True

    def add(self, X, metas):
        X = self._norm(X.astype("float32"))
        if not self.trained:
            self.train(X)
        self.index.add(X); self.meta.extend(metas)

    def search(self, Q, topk=5):
        Q = self._norm(Q.astype("float32"))
        D,I = self.index.search(Q, topk) # D는 1/유사도
        metas = [[self.meta[i] for i in row] for row in I]
        return D, I, metas

    def save(self, path_index, path_meta):
        faiss.write_index(self.index, path_index)
        with open(path_meta, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False)+"\n")

    @staticmethod
    def load(path_index, path_meta, normalize=True):
        idx = EmbeddingIndex(1)  # dummy
        idx.index = faiss.read_index(path_index)
        idx.dim = idx.index.d
        idx.normalize = normalize
        idx.meta = []
        with open(path_meta, "r", encoding="utf-8") as f:
            for line in f:
                idx.meta.append(json.loads(line))
        idx.trained = True
        return idx
