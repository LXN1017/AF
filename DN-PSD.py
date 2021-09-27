## Code for the DN-PSD network
## Author： Xuenan Liu
## 2021/09/20

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import regularizers


def feature_normalize(data):
    for i in range(data.shape[0]):
        mu = np.mean(data[i, :])
        sigma = np.std(data[i, :])
        data[i, :] = (data[i, :] - mu) / sigma
    return data


x_train_VPPG = np.load('x_train_VPPG.npy') # VPPG pulse signals from face videos
x_train_PPG = np.load('x_train_PPG.npy') # PPG pulse signals from fingertips
y_train = np.load('y_train.npy') # lables for AF classification


# -----------------------Sparse representation in DN-PSD-------------------------------
inputs = tf.keras.Input(shape=(1, 600), name='raw_bvp')
d1 = tf.keras.layers.Dense(750, activation='tanh', )(inputs)   # encoder
d2 = tf.keras.layers.Dense(900, activation='tanh', activity_regularizer=regularizers.l1(10e-3))(d1) # encoder
d3 = tf.keras.layers.Dense(750, use_bias=False)(d2)          # decoder
outputs = tf.keras.layers.Dense(600, use_bias=False)(d3)          # decoder
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='PSD')
model.summary()


model.compile(optimizer='adam',
              loss='mse',
              metrics='mse',
              )

history_PSD = model.fit(x_train_VPPG, x_train_PPG, batch_size=16, epochs=50)


# extract feature layers from the trained DN-PSD
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
successive_feature_maps = visualization_model.predict(x_train_VPPG)

# -----------------------AF classifier in DN-PSD-------------------------------
inputs = tf.keras.Input(shape=(1, 900), name='sparse_codes')
c1 = tf.keras.layers.Dense(450, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(2, activation='sigmoid')(c1)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='AF_classify')
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['AUC'],
              )

history_class = model.fit(successive_feature_maps[0], y_train, batch_size=16, epochs=50)
