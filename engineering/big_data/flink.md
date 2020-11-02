### 道路交通实时速度处理项目
1. 简述

模拟道路测速器实时上报传感器数据到kafka，flink实时消费计算每条道路的在15分钟内的平均速度，可视化展示

2. 数据

百度开放数据: [traffic_speed_sub-data](https://ai.baidu.com/broad/download?dataset=traffic)

3. 模块拆分
 - 模拟数据生产者
 - flink 消费 kafka 数据。具体包括：时间窗口计算；watermark机制处理乱序(迟到)数据；sink结果数据到Redis
 - 前端实时可视化 Redis 数据
 
