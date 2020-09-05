### FM算法相关 

1. 论文：[Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

2. 博客: 

   - [Introductory Guide – Factorization Machines & their application on huge datasets (with codes in Python)](https://www.analyticsvidhya.com/blog/2018/01/factorization-machines/)

3. 工具：

   - [xLearn](https://xlearn-doc.readthedocs.io/en/latest/index.html)

     <img height=400 width=300 src=https://github.com/aksnzhy/xLearn/raw/master/img/code.png>

4. 大佬关于FM模型的知乎链接 - [推荐系统召回四模型之：全能的FM模型](https://zhuanlan.zhihu.com/p/58160982)

5. FM keras 实现
```python
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

K = tf.keras.backend


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim=30, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(FMLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(FMLayer, self).build(input_shape)

    def call(self, x):
        a = K.pow(K.dot(x, self.kernel), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))
        return K.mean(a - b, 1, keepdims=True) * 0.5

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def build_model(feature_dim):
    inputs = tf.keras.Input((feature_dim,))
    liner = tf.keras.layers.Dense(units=1,
                                  bias_regularizer=tf.keras.regularizers.l2(0.01),
                                  kernel_regularizer=tf.keras.regularizers.l1(0.02),
                                  )(inputs)
    cross = FMLayer(feature_dim)(inputs)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.layers.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.optimizers.Adam(0.001),
                  metrics=['binary_accuracy'])
    return model


if __name__ == '__main__':
    fm = build_model(30)
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                        random_state=27, stratify=data.target)
    fm.fit(X_train, y_train, epochs=1, batch_size=16, validation_data=(X_test, y_test))
    
    tf.keras.models.save_model(
        fm,
        './fm_keras_saved_model/1',
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
```
