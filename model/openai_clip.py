from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from PIL import Image
from .base_models import BaseModelEmbedder


class OpenAIClipModel(BaseModelEmbedder):
    def load_model(self):
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"✅ Loaded {model_name}")

    def get_image_embedding(self, image_input):
        """
        image_input: có thể là đường dẫn (str) hoặc PIL.Image
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input  # PIL.Image

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

    def get_text_embedding(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()
