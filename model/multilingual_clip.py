from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
from PIL import Image
import torch
from .base_models import BaseModelEmbedder


class MultilingualClipModel(BaseModelEmbedder):
    """
    Model: sentence-transformers/clip-ViT-B-32-multilingual-v1
    - Ảnh encode bằng CLIP gốc (ViT-B-32)
    - Text encode bằng bản multilingual map vào cùng không gian
    """

    def load_model(self):
        try:
            self.model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
            self.img_model = SentenceTransformer("clip-ViT-B-32")
            self.text_model = SentenceTransformer(self.model_name)
            print(f"✅ Loaded model: {self.model_name}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def build_gallery_embeddings(self):
        """Encode toàn bộ ảnh trong thư mục dataset_dir"""
        paths = [
            os.path.join(self.dataset_dir, f)
            for f in os.listdir(self.dataset_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        if not paths:
            raise ValueError("Không có ảnh nào trong thư mục dataset_dir")

        images = [Image.open(p).convert("RGB") for p in paths]
        embs = self.img_model.encode(images, convert_to_numpy=True, show_progress_bar=True)
        print("Gallery embeddings shape:", embs.shape)
        self.gallery_paths = paths
        self.gallery_embs = np.vstack(embs)
        print("Gallery embeddings shape:",  self.gallery_embs.shape)
        print(f"📸 Encoded {len(paths)} ảnh trong gallery.")
        return self.gallery_paths, self.gallery_embs

    def get_image_embedding(self, image_input):
        """Encode 1 ảnh (path hoặc PIL.Image)"""
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            else:
                image = image_input.convert("RGB")
            emb = self.img_model.encode([image], convert_to_numpy=True)
            return emb
        except Exception as ex:
            print(f"❌ Lỗi khi encode ảnh: {ex}")
            return None

    def get_text_embedding(self, text: str):
        """Encode 1 câu mô tả"""
        try:
            emb = self.text_model.encode([text], convert_to_numpy=True)
            return emb
        except Exception as ex:
            print(f"❌ Lỗi khi encode text: {ex}")
            return None

    def search(self, query_emb, top_k=5):
        if query_emb is None:
            return []
        print("model ",self.model_name)
        cos_sim = util.cos_sim(query_emb, self.gallery_embs)  # shape (1, N)
        cos_sim = cos_sim.squeeze(0).numpy()                            # shape (N,)

        topk_idx = np.argsort(cos_sim)[::-1][:top_k]
        results = [{"path": self.gallery_paths[i], "similarity": float(cos_sim[i])} for i in topk_idx]
        return results
