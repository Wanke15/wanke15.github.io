### 推荐系统实践中的Q&A(暂时未总结，随想随记)
目前的架构
<img src=./assets/architect.png>
#### 1. 如何形成推荐系统的闭环？
如何形成闭环其实说白了就是如何搜集和利用来自于他人或环境的反馈从而不断完善和迭代某种行为或决策。
大体的想法是通过埋点日志解析用户对推荐的反馈，这些反馈进一步构成推荐召回算法和排序算法的训练数据，有了源源不断的训练数据，就可以不断的迭代更新推荐模型，从而形成闭环。
这里的关键是合理地设置用户埋点，合理的意思是考虑分析问题与算法建模需要什么样的数据。

#### 2. 离线推荐列表计算需要考虑的问题(持续更新)：
 （1）离线存储每个用户或物品的推荐列表要多大才够用？10？100？200？1000？
 
 人活着需要多少钱才够用？不同的人、人生的不同阶段可能都会有不同的答案。离线存储的”够用“也必须结合具体的业务才能做出判断，比如短视频信息流推荐，我觉得在保证推荐质量的前提下，多多益善。
 如果是像出国旅游选酒店这种场景，用户行为间隔比较久，总体上对于推荐的量可能要求不高。因此，需要结合现有的业务需求做些尝试，从而做到既不冗余，又能满足场景需要。
 
#### 3. 在线接口某些场景下的考虑(持续更新)：
 （1）新用户的推荐策略
  - 随机
  - 新品推荐
  - 热门推荐
  - 特定业务规则推荐
 
 （2）过滤规则
   
  为了避免重复推荐，一般会在Redis中维护一个下发历史列表，通过这个列表过滤召回或排序的结果。但是不可能进入该用户下发历史列表的物料以后就不推荐了，这样肯定是不合理的。
  那么什么时候该恢复该物料或重置该列表呢？目前的有两个思路：a) 只过滤最近比如说10次***接口请求***的推荐物料 b) 设置每个进入下发历史列表的物料的***过期时间***
  
#### 4. 排序模型可以考虑的特征有哪些？
 - User 特征：用户年龄，性别，婚否，是否价格敏感，有无孩子等
 - Item 特征：价格，折扣，品类和品牌相关特征，短期和长期统计特征等
 - Context 特征：天气，时间，地理位置，温度等
 - 行为 特征：用户点击Item序列，下单的Item序列
 
#### 5. 用户兴趣的向量化表达：

为了生成用户兴趣向量，可以先根据用户行为日志(点击、下单等)训练word2vec模型形成静态的向量表示，然后对行为序列采用AvgPooling、MaxPooling、WeightedPooling等方式融合得到固定维度的用户兴趣向量表示。
或者利用类似BERT的模型，形成固定维度的动态的用户兴趣向量。

#### 6. 提高推荐结果多样性可以尝试的方法：
 - 召回阶段采用多路召回：协同召回、Word2vec召回、长期兴趣召回、实时短期兴趣召回
 - 构建排序模型时，引入或构造更丰富的用户、物品等的交叉特征，可以显著提升推荐的多样性（携程人工智能实践）
 - 对于排序的结果，根据得分进行加权随机采样。在系统初期采用标签推荐算法时可以有效避免某一种类型标签权重过大导致的多样性缺失
 
#### 7. 用户实时兴趣计算：
 - （1）实时行为数据流计算清洗：客户端 -> nginx日志+Lua解析 -> Flume push -> Kafka -> SparkStreaming -> Redis
 - （2）兴趣标签权重计算：行为类型的业务权重及行为时间衰减加权
 
#### 8. 物品Embedding
 - （1）简单的方法可以利用word2vec算法训练物品的向量表示。数据清洗每个用户每个推荐session的点击序列，作为词向量模型训练中的"句子"，每个物品即为"单词"，从而构建物品的embedding
 - （2）类似BERT，通过完成某种预训练任务，每个物品的embedding作为中间产物来表征物品
 - （3）计算物品Embedding之间的相似度作为离线Embedding推荐的依据，可以直接用于线上类似"猜你喜欢"的推荐模块或者作为多路召回的一个源，提供给后续的rank或rerank模块
 
#### 9. 向量召回近邻搜索
 - (1）直接根据word2vec或GE构建完物品向量后，可以根据用户历史浏览物品或用户自身的向量表示进行向量召回。召回模块应该满足的一个特点是速度快，在内容池很大时，直接计算物品之间的余弦距离或欧氏距离开销较大，因此在工程实践中一般采用**近似近邻检索**来满足召回模块的时延要求。近似近邻检索的总结后边可以单独写一下，在这里记一下其相关的工具和利用nmslib和直接Word2vec精确计算的时间效率对比：
    - [Non-metric space library (nmslib)](https://github.com/nmslib/nmslib)
    - [Faiss libary by facebook](https://github.com/facebookresearch/faiss)
    - [Annoy by Spotify](https://github.com/spotify/annoy)
 - (2) 评测
 <img src="https://github.com/erikbern/ann-benchmarks/raw/master/results/glove-100-angular.png">
 
 - (3) Glove词向量实验对比
 
   ```python
   # 1. nmslib
   %timeit -n 10000 glove_index.knnQuery(data[word2id["hello"]], k=5+1)
   # 29.4 µs ± 1.05 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
   
   # 2. gensim word2vec
   %timeit -n 10000 glove_model.most_similar("hello", topn=5)
   # 7.55 ms ± 310 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
   
   # 3. faiss-cpu
   %timeit -n 100 glove_index.search(vecs[:1], 5+1)
   # 18.7 ms ± 204 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
   ```

#### 9. 如何评价Embedding的效果
目前信息流的数据是有各个业务类别属性的，可以将Embedding作为每个Item的特征表示，从而通过类别预测监督学习的准确率、召回率等指标来评估

#### 10. Tensorflow wide and deep 模型推理速度优化

Nvidia官方博文[Accelerating Wide & Deep Recommender Inference on GPUs](https://developer.nvidia.com/blog/accelerating-wide-deep-recommender-inference-on-gpus/)

#### 11. Keras 实现 FM 非常有意思的参考
```python
#############################################################################################################
# Created by qqgeogor
# https://www.kaggle.com/qqgeogor
#############################################################################################################

import numpy as np
from sklearn.base import BaseEstimator
from keras.layers import Input, Embedding, Dense,Flatten, merge,Activation
from keras.models import Model
from keras.regularizers import l2 as l2_reg
from keras import initializations
import itertools


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def batch_generator(X,y,batch_size=128,shuffle=True):
    sample_size = X[0].shape[0]
    index_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch = [X[i][batch_ids] for i in range(len(X))]
            y_batch = y[batch_ids]
            yield X_batch,y_batch


def test_batch_generator(X,y,batch_size=128):
    sample_size = X[0].shape[0]
    index_array = np.arange(sample_size)
    batches = make_batches(sample_size, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        X_batch = [X[i][batch_ids] for i in range(len(X))]
        y_batch = y[batch_ids]
        yield X_batch,y_batch


def predict_batch(model,X_t,batch_size=128):
    outcome = []
    for X_batch,y_batch in test_batch_generator(X_t,np.zeros(X_t[0].shape[0]),batch_size=batch_size):
        outcome.append(model.predict(X_batch,batch_size=batch_size))
    outcome = np.concatenate(outcome).ravel()
    return outcome



def build_model(max_features,K=8,solver='adam',l2=0.0,l2_fm = 0.0):

    inputs = []
    flatten_layers=[]
    columns = range(len(max_features))
    for c in columns:
        inputs_c = Input(shape=(1,), dtype='int32',name = 'input_%s'%c)
        num_c = max_features[c]

        embed_c = Embedding(
                        num_c,
                        K,
                        input_length=1,
                        name = 'embed_%s'%c,
                        W_regularizer=l2_reg(l2_fm)
                        )(inputs_c)

        flatten_c = Flatten()(embed_c)

        inputs.append(inputs_c)
        flatten_layers.append(flatten_c)

    fm_layers = []
    for emb1,emb2 in itertools.combinations(flatten_layers, 2):
        dot_layer = merge([emb1,emb2],mode='dot',dot_axes=1)
        fm_layers.append(dot_layer)

    for c in columns:
        num_c = max_features[c]
        embed_c = Embedding(
                        num_c,
                        1,
                        input_length=1,
                        name = 'linear_%s'%c,
                        W_regularizer=l2_reg(l2)
                        )(inputs[c])

        flatten_c = Flatten()(embed_c)

        fm_layers.append(flatten_c)
        
        
    flatten = merge(fm_layers,mode='sum')
    outputs = Activation('sigmoid',name='outputs')(flatten)
    
    model = Model(input=inputs, output=outputs)

    model.compile(
                optimizer=solver,
                loss= 'binary_crossentropy'
              )

    return model


class KerasFM(BaseEstimator):
    def __init__(self,max_features=[],K=8,solver='adam',l2=0.0,l2_fm = 0.0):
        self.model = build_model(max_features,K,solver,l2=l2,l2_fm = l2_fm)

    def fit(self,X,y,batch_size=128,nb_epoch=10,shuffle=True,verbose=1,validation_data=None):
        self.model.fit(X,y,batch_size=batch_size,nb_epoch=nb_epoch,shuffle=shuffle,verbose=verbose,validation_data=None)

    def fit_generator(self,X,y,batch_size=128,nb_epoch=10,shuffle=True,verbose=1,validation_data=None,callbacks=None):
        tr_gen = batch_generator(X,y,batch_size=batch_size,shuffle=shuffle)
        if validation_data:
            X_test,y_test = validation_data
            te_gen = batch_generator(X_test,y_test,batch_size=batch_size,shuffle=False)
            nb_val_samples = X_test[-1].shape[0]
        else:
            te_gen = None
            nb_val_samples = None

        self.model.fit_generator(
                tr_gen, 
                samples_per_epoch=X[-1].shape[0], 
                nb_epoch=nb_epoch, 
                verbose=verbose, 
                callbacks=callbacks, 
                validation_data=te_gen, 
                nb_val_samples=nb_val_samples, 
                max_q_size=10
                )

    def predict(self,X,batch_size=128):
        y_preds = predict_batch(self.model,X,batch_size=batch_size)
        return y_preds
```

#### 12. [汽车之家推荐系统排序算法迭代之路](https://www.infoq.cn/article/87gOLIaqWZW4moL0G9Ke)

1. 模型演进。LR -> XGB(同阶段：XGB+LR) -> FM -> DeepFM(同阶段：WideAndDeep) -> Online DeepFM
2. 样本生成。对于 Label 及特征的实时获取是通过每次请求的唯一标识 id 使用服务端 dump 的特征和客户端的 Label ( 曝光、点击 ) 进行 join 生成，这里要注意的是 Label 必须和当次请求的特征 join，如果特征数据在 Label 之后有更新，则会产生特征穿越的问题。
3. 特征介绍。
<img src="https://static001.infoq.cn/resource/image/2d/aa/2d3f840ecf63fd2aaf450b92cf5327aa.png">

