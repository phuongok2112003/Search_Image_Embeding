import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image

class BaseModelEmbedder(ABC):
    def __init__(self, dataset_dir="./images", device=None):
        self.dataset_dir = dataset_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.gallery_paths = []
        self.gallery_embs = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def get_image_embedding(self, image_path: str):
        pass

    @abstractmethod
    def get_text_embedding(self, text: str):
        pass

    def build_gallery_embeddings(self):
        """Encode toàn bộ ảnh trong thư mục images/"""
        paths = [os.path.join(self.dataset_dir, f)
                 for f in os.listdir(self.dataset_dir)
                 if f.lower().endswith((".jpg", ".png"))]

        embs = []
        for path in paths:
            embs.append(self.get_image_embedding(path))

        self.gallery_paths = paths
        self.gallery_embs = np.vstack(embs)
        return self.gallery_paths, self.gallery_embs

    def search(self, query_emb, top_k=5):
        """Tính cosine similarity và trả về top ảnh"""
        self.gallery_embs = self.gallery_embs / np.linalg.norm(self.gallery_embs, axis=1, keepdims=True)

        sims = np.dot(self.gallery_embs, query_emb)
        topk_idx = np.argsort(sims)[::-1][:top_k]
        results = [{"path": self.gallery_paths[i], "similarity": float(sims[i])} for i in topk_idx]
        return results
