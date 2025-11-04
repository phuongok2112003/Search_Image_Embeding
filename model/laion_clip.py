import open_clip
import torch
import numpy as np
from PIL import Image
from .base_models import BaseModelEmbedder

class LaionClipModel(BaseModelEmbedder):
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.pretrained = pretrained

    def load_model(self):
        self.model, _, self.processor = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        print(f"✅ Loaded open_clip {self.model_name} ({self.pretrained})")

    def get_image_embedding(self, image_input):
        """
        image_input: có thể là đường dẫn (str) hoặc PIL.Image
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input

        image_tensor = self.processor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(image_tensor)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

    def get_text_embedding(self, text: str):
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()
