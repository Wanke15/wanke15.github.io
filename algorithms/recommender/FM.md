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
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
# assert tf.__version__ == "2.3.0"
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

K = tf.keras.backend


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, embedding_dim, **kwargs):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = 1
        super(FMLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config

    def build(self, input_shape):
        self.embedding = self.add_weight(name='embedding',
                                         shape=(self.input_dim, self.embedding_dim),
                                         initializer='glorot_uniform',
                                         trainable=True)
        super(FMLayer, self).build(input_shape)

    @tf.function
    def call(self, x):
        a = K.pow(K.dot(x, self.embedding), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.embedding, 2))
        return K.mean(a - b, 1, keepdims=True) * 0.5

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def build_model(feature_dim, embedding_dim=8):
    inputs = tf.keras.Input((feature_dim,))
    liner = tf.keras.layers.Dense(units=1,
                                  bias_regularizer=tf.keras.regularizers.l2(0.01),
                                  kernel_regularizer=tf.keras.regularizers.l1(0.02),
                                  )(inputs)
    cross = FMLayer(feature_dim, embedding_dim)(inputs)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.layers.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['binary_accuracy'])
    return model


if __name__ == '__main__':
    fm = build_model(30)
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                        random_state=27, stratify=data.target)
    fm.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

    tf.saved_model.simple_save(
        tf.keras.backend.get_session(),
        './fm_keras_saved_model/1',
        inputs={t.name: t for t in fm.inputs},
        outputs={t.name: t for t in fm.outputs}
    )
     
    # 下面的步骤是保存为单一的pb模型。该方法在2.3.0版本下亲测可用
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    
    full_model = tf.function(lambda x: fm(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(fm.inputs[0].shape, fm.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models2",
                      name="model.pb",
                      as_text=False)
```

6. Keras 实现 FM 非常有意思的参考
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
