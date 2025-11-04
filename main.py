from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import JSONResponse
from model import MODEL_CLASS_MAP, ModelType
from PIL import Image
import numpy as np

app = FastAPI(title="Multi-Model Image Search API")

loaded_models = {}

@app.post("/load_model")
def load_model(model_type: ModelType):
    try:
        if model_type not in loaded_models:
            ModelClass = MODEL_CLASS_MAP[model_type]
            model_instance = ModelClass()
            model_instance.load_model()
            model_instance.build_gallery_embeddings()
            loaded_models[model_type] = model_instance
        return {"message": f"Loaded {model_type.value} thành công!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/search")
async def search(
    model_type: ModelType = Form(...),
    query_text: str = Form(None),
    query_image: UploadFile = None,
    top_k: int = Form(5)
):
    try:
        # 1️⃣ Load model nếu chưa có
        if model_type not in loaded_models:
            ModelClass = MODEL_CLASS_MAP[model_type]
            model_instance = ModelClass()
            model_instance.load_model()
            model_instance.build_gallery_embeddings()
            loaded_models[model_type] = model_instance
        else:
            model_instance = loaded_models[model_type]

        # 2️⃣ Lấy embedding
        if query_image:
            img = Image.open(query_image.file).convert("RGB")
            query_emb = model_instance.get_image_embedding(img)
        elif query_text:
            query_emb = model_instance.get_text_embedding(query_text)
        else:
            return JSONResponse(status_code=400, content={"error": "Phải gửi query_text hoặc query_image."})

        # 3️⃣ Tìm ảnh tương tự
        results = model_instance.search(query_emb, top_k=top_k)

        return {
            "model": model_type.value,
            "query_text": query_text,
            "top_k": top_k,
            "results": results,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
