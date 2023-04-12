#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops.init_ops import TruncatedNormal

batch_size = 10
feat_num = 2
feat_embed_size = 4

embedding_size = 4
att_embedding_size = 8
head_num = 2
seed = 101

# 1. 权重初始化
weight_shape = [embedding_size, att_embedding_size * head_num]

W_Query = tf.Variable(name='query', shape=weight_shape,
                      dtype=tf.float32,
                      initial_value=TruncatedNormal(seed=seed)(weight_shape))
W_key = tf.Variable(name='key', shape=weight_shape,
                    dtype=tf.float32,
                    initial_value=TruncatedNormal(seed=seed + 1)(weight_shape))
W_Value = tf.Variable(name='value', shape=weight_shape,
                      dtype=tf.float32,
                      initial_value=TruncatedNormal(seed=seed + 2)(weight_shape))
W_Res = tf.Variable(name='res', shape=weight_shape,
                    dtype=tf.float32,
                    initial_value=TruncatedNormal(seed=seed)(weight_shape))

# 2. 模拟输入
inputs = tf.random.uniform(shape=(batch_size, feat_num, feat_embed_size))

# 3. Q, K, V 计算
# None, feat_d, atten_d * head_num
querys = tf.tensordot(inputs, W_Query, axes=(-1, 0))
keys = tf.tensordot(inputs, W_key, axes=(-1, 0))
values = tf.tensordot(inputs, W_Value, axes=(-1, 0))

# 4. 变形，为了使用矩阵乘法来实现 multi_head
# head_num, None, feat_d, atten_d
querys = tf.stack(tf.split(querys, head_num, axis=2))
keys = tf.stack(tf.split(keys, head_num, axis=2))
values = tf.stack(tf.split(values, head_num, axis=2))

# 5. 注意力得分
# head_num, None, feat_d, feat_d
inner_product = tf.matmul(querys, keys, transpose_b=True)
inner_product /= att_embedding_size ** 0.5
normalized_atten_score = tf.nn.softmax(inner_product)

# 6. 基于注意力得分的输出
# head_num, None, feat_d, atten_d
result = tf.matmul(normalized_atten_score, values)

# 1, None, feat_d, atten_d * head_num
result = tf.concat(tf.split(result, head_num), axis=-1)

# None, feat_d, atten_d * head_num
result = tf.squeeze(result, axis=0)

# 7. 残差连接
# None, feat_d, atten_d * head_num
result += tf.tensordot(inputs, W_Res, axes=(-1, 0))

# 8. 最后的非线性计算
# None, feat_d, atten_d * head_num
result = tf.nn.relu(result)

print(result.shape)

assert result.shape == (batch_size, feat_num, att_embedding_size * head_num)

print(result.shape)
