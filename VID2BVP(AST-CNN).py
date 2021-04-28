## updated by authors of ICCV manuscript 6416.
## 2021/03/29
import tensorflow as tf
import h5py
from scipy.io import loadmat as load
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.layers import Conv3D, BatchNormalization, Activation, MaxPool3D, Dropout, Flatten, Dense
import scipy.io as io
from scipy.io import loadmat as load


def feature_normalize(data):
    for i in range(data.shape[0]):
        mu = np.mean(data[i, :])
        sigma = np.std(data[i, :])
        data[i, :] = (data[i, :] - mu) / sigma
    return data


#-----------------load videos used as inputs of AST-CNN-----------------------
feature = h5py.File('vid_set.mat')
x_train = feature['vid_set']
tf.bitcast(x_train, float)
x_train = np.expand_dims(x_train, axis=0)
x_train = np.expand_dims(x_train, axis=4)
#-----------------load phase changes of videos--------------------------------
mask = load('vid_phase.mat')
x_train_mask = mask['vid_phase']
tf.bitcast(x_train_mask, float)
x_train_mask = np.expand_dims(x_train_mask, axis=0)
x_train_mask = np.expand_dims(x_train_mask, axis=4)
#-----------------load reference pulse signals used as labels-----------------
data = load('bvp_set.mat')
y_train = data['bvp_set']
tf.bitcast(y_train, float)
# -----------------------AST-CNN architecture-------------------------------
input1 = tf.keras.Input(shape=(100, 200, 600, 1), name='bvp')
c11 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 1), strides=[3, 3, 1], padding='valid', activation='tanh')(input1)
c12 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 3), strides=[1, 1, 1], padding='valid',  activation='tanh')(c11)
input2 = tf.keras.Input(shape=(100, 200, 600, 1), name='phase')
c12_mask = tf.keras.layers.AvgPool3D(pool_size=(3, 3, 3), strides=[3, 3, 1], padding='valid')(input2)
c12_mask_mul = tf.tile(c12_mask, [1, 1, 1, 1, 64])
c12_att = tf.multiply(c12, c12_mask_mul)
c21 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 1), strides=[3, 3, 1], padding='valid', activation='tanh')(c12_att)
c22 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 3), strides=[1, 1, 1], padding='valid',  activation='tanh')(c21)
c22_mask = tf.keras.layers.AvgPool3D(pool_size=(3, 3, 3), strides=[3, 3, 1], padding='valid')(c12_mask)
c22_mask_mul = tf.tile(c22_mask, [1, 1, 1, 1, 32])
c22_att = tf.multiply(c22, c22_mask_mul)
c31 = tf.keras.layers.Conv3D(filters=1,  kernel_size=(1, 1, 1), strides=[1, 1, 1], padding='valid',  activation='tanh')(c22_att)
b1 = BatchNormalization()(c31)
p1 = tf.keras.layers.AvgPool3D(pool_size=(11, 22, 1), strides=[1, 1, 1], padding='valid')(b1)  # 池化层
d1 = tf.keras.layers.Dropout(0.2)(p1) # dropout层
fc1 = tf.keras.layers.Flatten()(d1)
fc2 = tf.keras.layers.Dense(326, activation='sigmoid')(fc1)
fc3 = tf.keras.layers.Dense(326, activation='sigmoid')(fc2)
model = tf.keras.Model(inputs=[input1, input2], outputs=fc3, name='BVP_detector')
model.summary()

# ---------------------loss function----------------------------------
def custom_loss(lab, pre):
    lab_ = lab-tf.reduce_mean(lab, axis=1)
    pre_ = pre-tf.reduce_mean(pre, axis=1)
    Pearson = tf.multiply(lab_, pre_)/(tf.norm(lab_)*tf.norm(pre_))
    custom_loss = 1 - Pearson
    return custom_loss

model.compile(optimizer='adam',
               loss=custom_loss,
               metrics=['MSE', 'CosineSimilarity']
              )

history = model.fit([x_train, x_train_mask], [y_train[:, 2:-2]], batch_size=16, epochs=30, )

# -----------------------feature map learned by AST-CNN -------------------------------
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
successive_feature_maps = visualization_model.predict([x_train, x_train_mask])

