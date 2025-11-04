import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# 1. Load model CLIP
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. Hàm encode ảnh
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2)
    return emb.cpu().numpy().flatten()

# 3. Hàm encode văn bản
def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2)
    return emb.cpu().numpy().flatten()

# 4. Tạo dataset ảnh
dataset_dir = "images"
image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith((".jpg", ".png"))]

gallery_embs = []
for path in tqdm(image_paths, desc="Encoding gallery"):
    gallery_embs.append(get_image_embedding(path))
gallery_embs = np.vstack(gallery_embs)

while True:
    # 5. Query có thể là text hoặc ảnh
    query = input("Nhập mô tả hoặc đường dẫn ảnh: ").strip()
    if query=='q':
        break
    if os.path.exists(query):  # nếu là đường dẫn file ảnh
        print(f"Tìm ảnh giống: {query}")
        query_emb = get_image_embedding(query)
    else:  # nếu là văn bản
        print(f"Tìm ảnh phù hợp với mô tả: '{query}'")
        query_emb = get_text_embedding(query)

    # 6. Tính cosine similarity
    similarities = np.dot(gallery_embs, query_emb)
    topk_idx = np.argsort(similarities)[::-1][:5]

    # 7. In kết quả
    print("\nẢnh phù hợp nhất:")
    for i in topk_idx:
        print(f"{image_paths[i]} - similarity: {similarities[i]:.4f}")
