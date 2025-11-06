# import base64
# import asyncio
# from langchain_ollama import ChatOllama
# from langchain_core.messages import HumanMessage

# async def main():
#     # 1️⃣ Khởi tạo model
#     llm = ChatOllama(
#         model="qwen3-vl",
#         base_url="http://10.0.13.13:11434",  # Địa chỉ Ollama server
#         temperature=0,
#     )

#     # 2️⃣ Đọc ảnh & mã hóa base64
#     image_path = "Media (1).jpg"
#     with open(image_path, "rb") as f:
#         image_base64 = base64.b64encode(f.read()).decode()

#     # 3️⃣ Tạo message đúng chuẩn cho Ollama
#     message = HumanMessage(content=[
#         {"type": "text", "text": "Mô tả chi tiết nội dung của bức ảnh này."},
#         {"type": "image_url", "image_url": image_base64}  # ❗ KHÔNG có prefix
#     ])

#     # 4️⃣ Gọi model async
#     response = await llm.ainvoke([message])

#     # 5️⃣ In kết quả
#     print(response.content)

# # 6️⃣ Chạy async main
# if __name__ == "__main__":
#     asyncio.run(main())


import requests
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

url = "https://g8home.vn/hinh-san-pham/medium/DEN-AM-TRAN-DE-MONG-12W-ANH-SANG-DOI-MAU-26517.jpg"
r = requests.get(url, headers=headers, timeout=10)
print(r.status_code)