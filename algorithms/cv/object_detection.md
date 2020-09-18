1. mmdetection, tensorflow object detection api, detectron2

2. 样本不均衡
 - [理解Focal Loss与GHM——解决样本不平衡利器](https://zhuanlan.zhihu.com/p/80594704) - 涉及论文：[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) 和 [Gradient Harmonized Single-stage Detector](https://arxiv.org/pdf/1811.05181)。
 需要注意的是这两种方法都**是针对One-stage的**，对于像faster-rcnn这样的模型效果可能并不好
 - [OHEM论文解读](https://zhuanlan.zhihu.com/p/58162337)，论文：[Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/abs/1604.03540)
 - 数据增强 [imgaug](https://github.com/aleju/imgaug)

3. RCNN 相关
 - 说一下faster-rcnn的流程
 - 具体说说roi pooling

4. [目标检测比赛中的tricks](https://zhuanlan.zhihu.com/p/102817180)

主要结合mmdetection讲一些目标检测相关的技巧和优化
