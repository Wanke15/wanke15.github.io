# 利用paddleocr进行图片文字识别

## 1. 环境安装配置
```bash
# python==3.7.6
python -V

pip install paddlepaddle==2.5.2
pip install paddleocr==2.7.0.2
```

## 2. 模型下载
下载好的模型放在以下目录：
~/.paddleocr/whl

## 3. demo
```python
import logging
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)

from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(cpu_threads=12, gpu_id=1)

def ocr_analyze(image_path):
  image = cv2.imread(image_path)
  result = ocr.ocr(image)
  area_square_sum = 0
  image_square = image.shape[0] * image.shape[1]
  for idx, res in enumerate(result):
    bbox = res[0][0]
    pt1, pt2 = [[int(b) for b in bbox[0]], [int(b) for b in bbox[2]]]
    area_square = (pt2[0] - pt1[0]) * (pt2[1] - pt1[1])
    area_square_sum += area_square
  print(image_path, '文字区块数量：', len(result[0]), '文字区域面积：', area_square_sum, '图片面积：', image_square, '文字区域比例：', area_square_sum / image_square * 1.0)


```
