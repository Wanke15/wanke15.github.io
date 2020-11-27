1. Dataset 实用干货。包括dataset创建优化、spark dataframe 生成 tfrecord等

   [Tensorflow之dataset介绍](https://zhuanlan.zhihu.com/p/138099468)

2. tensorflow serving 模型热更新毛刺问题优化

参考： [爱奇艺文章](https://mp.weixin.qq.com/s/DkCGusznH8F8p39oRLuNBQ)

（1）配置模型Warmup文件来预热模型

（2）使用Jemalloc做内存分配优化

（3）TF模型参数分配和RPC请求内存分配分离

用tensorflow serving 做模型服务的时候，接口的第一次请求会较慢，20ms，接下来的请求就会稳定在 5ms 左右，原因在于 tensorflow 第一次计算时才会去做计算图初始化。
这个问题之前在自己用flask做接口时就遇到过，当时的解决方法就是加载模型后，给模型喂一两条数据，跑完之后才算模型初始化正式完成。在用tensorflow serving的时候，看日志有这么一条信息：
```bash
2020-09-05 09:34:32.442529: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:105] No warmup data file found at /models/boston/1/assets.extra/tf_serving_warmup_requests
```
也就是说，如果在模型目录的子目录***assets.extra***下创建一条接口所需的数据示例***tf_serving_warmup_requests***，那么tensorflow serving就会在加载模型后读取该数据做warmup。
那么，这条数据该怎么创建呢，直接上代码：
```python
import tensorflow as tf
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2


def main():
    with tf.io.TFRecordWriter("../fm_keras_saved_model/1/assets.extra/tf_serving_warmup_requests") as writer:
        request = predict_pb2.PredictRequest(
            model_spec=model_pb2.ModelSpec(name="boston"),
            inputs={"input_1": tf.make_tensor_proto([[12.49, 16.85, 79.19, 481.6, 0.08511, 0.03834, 0.004473, 0.006423, 0.1215, 0.05673, 0.1716, 0.7151, 1.047, 12.69, 0.004928, 0.003012, 0.00262, 0.00339, 0.01393, 0.001344, 13.34, 19.71, 84.48, 544.2, 0.1104, 0.04953, 0.01938, 0.02784, 0.1917, 0.06174]])}

        )
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())


if __name__ == "__main__":
    main()
    
```
这样创建之后，第一次接口请求的速度就变为了 5ms 左右，解决了线上请求的毛刺问题。
美团技术团队有一篇[基于TensorFlow Serving的深度学习在线预估](https://tech.meituan.com/2018/10/11/tfserving-improve.html)的文章有更多相关的优化可以参考

3. tf2 crf 实现
[tf2CRF](https://github.com/xuxingya/tf2crf)
```python
from tf2crf import CRF
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(None,), dtype='int32')
output = Embedding(100, 40, trainable=True, mask_zero=True)(inputs)
output = Bidirectional(GRU(64, return_sequences=True))(output)
output = Dense(9, activation=None)(output)
crf = CRF(dtype='float32')
output = crf(output)
model = Model(inputs, output)
model.compile(loss=crf.loss, optimizer='adam', metrics=[crf.accuracy])

x = [[5, 2, 3] * 3] * 10
y = [[1, 2, 3] * 3] * 10

model.fit(x=x, y=y, epochs=2, batch_size=2)
model.save('model')
```

4. 单机多卡分布式训练
```python
import tensorflow_datasets as tfds
import tensorflow as tf

print(tf.__version__)

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']

strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label


with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))


callbacks = [
    # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]


train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

model.fit(train_dataset, epochs=12, callbacks=callbacks)

```
