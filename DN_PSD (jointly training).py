## Code for the DN-PSD network
## Author： Xuenan Liu
## 2021/09/20

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import regularizers
from scipy.io import loadmat as load
from matplotlib import pyplot as plt


def feature_normalize(data):
    for i in range(data.shape[0]):
        min_val = np.min(data[i, :])
        max_val = np.max(data[i, :])
        data[i, :] = (data[i, :] - min_val) / (max_val - min_val)
    return data


def loss_DN_PSD(lab, pre):
    l1 = tf.keras.losses.mse(lab[:,  0:600], pre[:, 0:600])
    l2 = tf.keras.losses.categorical_crossentropy(lab[:, 600:602], pre[:, 600:602])
    l_sum = l1 + l2
    return l_sum


x_train_VPPG = np.load('x_train_VPPG.npy') # VPPG pulse signals from face videos
x_train_PPG = np.load('x_train_PPG.npy') # PPG pulse signals from fingertips
y_train = np.load('y_train.npy') # lables for AF classification


inputs = tf.keras.Input(shape=600, name='raw_bvp')
d1 = tf.keras.layers.Dense(750, activation='tanh', )(inputs)   # encoder
d2 = tf.keras.layers.Dense(900, activation='tanh', activity_regularizer=regularizers.l1(10e-3))(d1) # encoder
d3 = tf.keras.layers.Dense(750, use_bias=False)(d2)
d4 = tf.keras.layers.Dense(450, use_bias=False)(d2)
output1 = tf.keras.layers.Dense(600, use_bias=False)(d3)
output2 = tf.keras.layers.Dense(2, use_bias=False)(d4)
model = tf.keras.Model(inputs=inputs, outputs=[output1, output2], name='PSD')
model.summary()

model.compile(optimizer='adam',
              loss=loss_DN_PSD,
              )

history_PSD = model.fit(x_train_VPPG, [x_train_PPG, y_train], batch_size=16, epochs=50, validation_split=0.2)


loss = history_PSD.history['loss']  # 训练集loss曲线
val_loss = history_PSD.history['val_loss']  # 验证集loss曲线
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
