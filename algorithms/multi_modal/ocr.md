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

from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import math
import numpy as np

ocr = PaddleOCR(cpu_threads=12, gpu_id=1)

def ocr_analyze(image_path):
  image = cv2.imread(image_path)
  detections = ocr.ocr(image)
  result = detections[0]
  if not result:
    print('完全非文字图片')
    return
  area_square_sum = 0
  image_square = image.shape[0] * image.shape[1]
  for idx, res in enumerate(result):
    bbox = res[0]
    pt1, pt2 = [[int(b) for b in bbox[0]], [int(b) for b in bbox[2]]]
    area_square = (pt2[0] - pt1[0]) * (pt2[1] - pt1[1])

    # pt1, pt2 = [[int(b) for b in bbox[0]], [int(b) for b in bbox[2]]]
    # cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)
    # area_square = math.fabs((pt2[0] - pt1[0]) * (pt2[1] - pt1[1]))
    
    pts = np.array(bbox, np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], True, (0, 0, 255), 5)
    area_square = Polygon(bbox).area

    area_square_sum += area_square
  text_block_num = len(result[0])
  text_block_ratio = area_square_sum / image_square * 1.0
  if text_block_num > 5 or text_block_ratio > 0.05:
    print('大概率是文字图片')
  else:
    print('非文字图片（部分）')
  print(image_path, '文字区块数量：', text_block_num, '文字区域面积：', area_square_sum, '图片面积：', image_square, '文字区域比例：', text_block_ratio)

  plt.figure(figsize=(8, 12))
  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


```
