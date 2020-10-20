```python
import numpy as np

import pandas as pd
from attention import Attention
from sklearn.metrics import classification_report


def load_hand_select():
    with open('./data/xxx.txt', 'r') as f:
        func_sents = [line.strip() for line in f]

    with open('./data/xxx.txt', 'r') as f:
        rent_sents = [line.strip() for line in f]

    with open('./data/xxx.txt', 'r') as f:
        other_sents = [line.strip() for line in f]

    with open('./data/xxx.txt', 'r') as f:
        poi_sents = [line.strip() for line in f]

    data = [{"text": rs, "label": 0} for rs in rent_sents]
    data.extend([{"text": fs, "label": 1} for fs in func_sents])
    data.extend([{"text": pois, "label": 2} for pois in poi_sents])
    data.extend([{"text": ots, "label": 3} for ots in other_sents])

    return pd.DataFrame(data)


df = load_hand_select()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], random_state=0, test_size=0.3)


char_to_id = {c: i + 1 for i, c in enumerate(set([_ for t in X_train.values for _ in t]))}
char_to_id.update({"*": 0})
max_len = 10
unk_id = char_to_id.get('*')

import json
with open("./char_dict.json", 'w') as f:
    json.dump(char_to_id, f, indent=4, ensure_ascii=False)


import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model


def build_bigru_model():
    inputs = Input(shape=(None,), dtype='int32')
    output = Embedding(len(char_to_id), 32, trainable=True, mask_zero=True)(inputs)
    output = Bidirectional(GRU(8))(output)
    # output = GRU(8)(output)
    output = Dense(4, activation='softmax')(output)
    model = Model(inputs, output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def build_bigru_atten_model():
    inputs = Input(shape=(None,), dtype='int32')
    output = Embedding(len(char_to_id), 32, trainable=True, mask_zero=True)(inputs)

    # output = Bidirectional(GRU(8, return_sequences=True))(output)
    output = GRU(8, return_sequences=True)(output)
    output = Attention()(output)
    output = Dense(4, activation='softmax')(output)
    model = Model(inputs, output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def build_cnn_model():
    inputs = Input(shape=(10,), dtype='int32')
    embed = Embedding(len(char_to_id), 32, trainable=True, mask_zero=True)(inputs)

    cnn1 = tf.keras.layers.Conv1D(8, 2, padding='same', strides=1, activation='relu')(embed)
    cnn1 = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn1)
    cnn2 = tf.keras.layers.Conv1D(8, 3, padding='same', strides=1, activation='relu')(embed)
    cnn2 = tf.keras.layers.MaxPooling1D(pool_size=3)(cnn2)
    cnn3 = tf.keras.layers.Conv1D(8, 4, padding='same', strides=1, activation='relu')(embed)
    cnn3 = tf.keras.layers.MaxPooling1D(pool_size=4)(cnn3)
    # 合并三个模型的输出向量
    cnn = tf.keras.layers.concatenate([cnn1, cnn2, cnn3], axis=1)
    flat = tf.keras.layers.Flatten()(cnn)

    output = Dense(4, activation='softmax')(flat)
    model = Model(inputs, output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def build_cnn_model_v2():
    inputs = Input(shape=(10,), dtype='int32')
    embed = Embedding(len(char_to_id), 32, trainable=True, mask_zero=True)(inputs)

    cnn1 = tf.keras.layers.Conv1D(16, 2, padding='same', strides=1, activation='relu')(embed)
    cnn1 = tf.keras.layers.GlobalAveragePooling1D()(cnn1)
    cnn2 = tf.keras.layers.Conv1D(8, 3, padding='same', strides=1, activation='relu')(embed)
    cnn2 = tf.keras.layers.GlobalAveragePooling1D()(cnn2)
    cnn3 = tf.keras.layers.Conv1D(4, 4, padding='same', strides=1, activation='relu')(embed)
    cnn3 = tf.keras.layers.GlobalAveragePooling1D()(cnn3)
    # 合并三个模型的输出向量
    cnn = tf.keras.layers.concatenate([cnn1, cnn2, cnn3], axis=1)
    flat = tf.keras.layers.Flatten()(cnn)

    output = Dense(4, activation='softmax')(flat)
    model = Model(inputs, output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def transform_input(data):
    id_data = [[char_to_id.get(c, unk_id) for c in d] for d in data]
    id_data = tf.keras.preprocessing.sequence.pad_sequences(id_data, maxlen=max_len, dtype='int32', padding='post',
                                                            truncating='post', value=unk_id)
    return np.array(id_data)


X_train = transform_input(X_train.values)
X_test = transform_input(X_test.values)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.2)

# model = build_bigru_model()
# model = build_bigru_atten_model()
# model = build_cnn_model()
model = build_cnn_model_v2()

# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)
call_backs = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_acc')]
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=call_backs)

# id_to_cat = {0: 'rent_car', 1: 'function', 2: 'poi'}
id_to_cat = {0: 'xxx', 1: 'xxx', 2: 'xxx', 3: 'xxx'}
# target_names = ["xxx", "xxx", "xxx", "xxx"]
target_names = ["xxx", "xxx", "xxx", "xxx"]
# target_names = ["xxx", "xxx", "xxx"]

print(df['label'].value_counts())
print(classification_report(y_test, np.argmax(model.predict(X_test), axis=1), target_names=target_names))

# model.save("bigru.h5")
# model.save("bigru_atten.h5")
# model.save("textcnn_v1.h5")
# model.save("textcnn_v2.h5")


```
