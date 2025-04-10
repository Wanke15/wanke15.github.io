# paddleocr                     2.7.2
# paddlepaddle                  2.5.2

import hashlib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
import cv2
import numpy as np
from paddleocr import PaddleOCR

import memcache
memcache_client = memcache.Client(['127.0.0.1:11211'], debug=True)

cache_ttl = 3600 * 24 * 30
BASE_KEY = "qwoppqxxczkcpslqmljhgsas"

def generate_hash(input_string):
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode('utf-8'))
    return sha256.hexdigest()

# 初始化PaddleOCR
ocr = PaddleOCR(cpu_threads=30)  # 可以根据需要选择语言

app = FastAPI()

class ImageBase64Request(BaseModel):
    image_base64_list: List[str]

class OCRResponse(BaseModel):
    texts: List[str]

def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(request: ImageBase64Request):
    texts = []
    for image_base64 in request.image_base64_list:
        try:
            all_identifiers = str(BASE_KEY) + "paddleocr" + str(image_base64)
            unique_key = generate_hash(all_identifiers)
            _cache = memcache_client.get(unique_key, None)
            if _cache is not None:
                texts.append(_cache)
                print("命中缓存！")
                continue

            all_text = ''
            img = base64_to_image(image_base64)
            result = ocr.ocr(img, cls=True)
            if result[0] is None:
                result = [[]]
            for idx, res in enumerate(result[0]):
                all_text += res[-1][0] + '\n'
            texts.append(all_text)
            memcache_client.set(unique_key, all_text, cache_ttl)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    return OCRResponse(texts=texts)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1038)
