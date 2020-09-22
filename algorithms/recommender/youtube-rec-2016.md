##### 1. candidate generation 是如何在线serving的？
 - softmax之前隐含层的输出作为user_embedding。这部分因为用户的观看历史和搜索会变化，因此是需要在线实时计算的
 - 这里的softmax层是dense+softmax。softmax的输出是所有视频的weighted probalitity, 对于每一个视频来说，其对应的输入为user_embedding，这样的话对于某一个视频来说，
   可以将其和前一个隐层每个节点的权重作为video_embedding此时user_embedding和video_embedding维度是一致的
 - video_embedding作为权重来说是固定的；user_embedding是实时的，在线serving得到user_embedding后和video_embedding做NN近邻查询，召回candidate
 
