##Code for the AST-CNN network
## Author： Xuenan Liu
## 2021/09/20

import tensorflow as tf
import h5py
import numpy as np
from tensorflow.python.keras.layers import Conv3D, BatchNormalization, Activation, MaxPool3D, Dropout, Flatten, Dense
from scipy.io import loadmat as load


#-----------------signal normalization------------------------------
def feature_normalize(data):
    for i in range(data.shape[0]):
        min_val = np.min(data[i, :])
        max_val = np.max(data[i, :])
        data[i, :] = (data[i, :] - min_val) / (max_val - min_val)
    return data


#-----------------Pearson loss function------------------------------
def custom_loss(lab, pre):
    # lab_ = lab-tf.reduce_mean(lab)
    # pre_ = pre-tf.reduce_mean(pre)
    # Pearson = tf.multiply(lab_, pre_)/(tf.norm(lab_)*tf.norm(pre_))
    # neg_p = 1 - Pearson
    neg_p = tf.zeros([1, lab.shape[1]])
    p_sum = 0
    for i in range(0, lab.shape[0]):
        lab_nor = lab[i, :] - tf.reduce_mean(lab[i, :])
        pre_nor = pre[i, :] - tf.reduce_mean(pre[i, :])
        pearson = 1 - tf.multiply(lab_nor, pre_nor) / (tf.norm(lab_nor) * tf.norm(pre_nor))
        neg_p = p_sum + pearson
    neg_p = neg_p/32
    return neg_p


#-----------------load videos used as inputs of AST-CNN-----------------------
feature = h5py.File('vid_set.mat')
x_train = feature['vid_set']
tf.bitcast(x_train, float)
#-----------------load phase changes of videos--------------------------------
mask = h5py.File('vid_phase.mat')
x_train_mask = mask['vid_phase']
tf.bitcast(x_train_mask, float)
#-----------------load reference pulse signals used as labels-----------------
data = load('bvp_set.mat')
y_train = data['bvp_set']
tf.bitcast(y_train, float)

# -----------------------AST-CNN architecture-------------------------------
input1 = tf.keras.Input(shape=(300, 200, 600, 3), name='video_color')  #video color frames
input2 = tf.keras.Input(shape=(300, 200, 600, 1), name='video_phase')  #video phase frames
c1_3_1 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 1), strides=[3, 3, 1], padding='same', activation='tanh', name='c1_3_1')(input1)
c1_1_3 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 3), strides=[1, 1, 1], padding='same',  activation='tanh', name='c1_1_3')(c1_3_1)
c1_5_1 = tf.keras.layers.Conv3D(filters=64, kernel_size=(5, 5, 1), strides=[3, 3, 1], padding='same', activation='tanh', name='c1_5_1')(input1)
c1_1_5 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 5), strides=[1, 1, 1], padding='same',  activation='tanh', name='c1_1_5')(c1_5_1)
c1_7_1 = tf.keras.layers.Conv3D(filters=64, kernel_size=(7, 7, 1), strides=[3, 3, 1], padding='same', activation='tanh', name='c1_7_1')(input1)
c1_1_7 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 7), strides=[1, 1, 1], padding='same',  activation='tanh', name='c1_1_7')(c1_7_1)
c1_com = c1_1_3 + c1_1_5 + c1_1_7  # 64 feature maps
att_mask_1 = tf.keras.layers.AvgPool3D(pool_size=(3, 3, 3), strides=[3, 3, 1], padding='same')(input2)
att_mask_stack_1 = tf.tile(att_mask_1, [1, 1, 1, 1, 64])  # attention mask for 64 feature maps
c1_com_attention = tf.multiply(c1_com, att_mask_stack_1)  # masked feature maps
b1 = BatchNormalization()(c1_com_attention)
p1 = tf.keras.layers.AvgPool3D(pool_size=(3, 3, 3), strides=[1, 1, 1], padding='same')(b1)
d1 = tf.keras.layers.Dropout(0.2)(p1)

c2_3_1 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 1), strides=[3, 3, 1], padding='same', activation='tanh', name='c2_3_1')(d1)
c2_1_3 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 3), strides=[1, 1, 1], padding='same',  activation='tanh',name='c2_1_3')(c2_3_1)
c2_5_1 = tf.keras.layers.Conv3D(filters=64, kernel_size=(5, 5, 1), strides=[3, 3, 1], padding='same', activation='tanh', name='c2_5_1')(d1)
c2_1_5 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 5), strides=[1, 1, 1], padding='same',  activation='tanh', name='c2_1_5')(c2_5_1)
c2_7_1 = tf.keras.layers.Conv3D(filters=64, kernel_size=(7, 7, 1), strides=[3, 3, 1], padding='same', activation='tanh', name='c2_7_1')(d1)
c2_1_7 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 7), strides=[1, 1, 1], padding='same',  activation='tanh', name='c2_1_7')(c2_7_1)
c2_com = c2_1_3 + c2_1_5 + c2_1_7
att_mask_2 = tf.keras.layers.AvgPool3D(pool_size=(3, 3, 3), strides=[3, 3, 1], padding='same')(att_mask_1)
att_mask_stack_2 = tf.tile(att_mask_2, [1, 1, 1, 1, 64])
c2_com_attention = tf.multiply(c2_com, att_mask_stack_2)
b1 = BatchNormalization()(c1_com_attention)
p2 = tf.keras.layers.AvgPool3D(pool_size=(3, 3, 3), strides=[1, 1, 1], padding='same')(c2_com_attention)
d2 = tf.keras.layers.Dropout(0.2)(p2)
p3 = tf.keras.layers.AvgPool3D(pool_size=(34, 23, 1), strides=[1, 1, 1], padding='valid')(d2)
c31 = tf.keras.layers.Conv3D(filters=1,  kernel_size=(1, 1, 1), strides=[1, 1, 1], padding='valid',  activation='tanh')(p3)
fc1 = tf.keras.layers.Flatten()(c31)
fc2 = tf.keras.layers.Dense(600, activation='sigmoid')(fc1)
fc3 = tf.keras.layers.Dense(600, activation='sigmoid')(fc2)
model = tf.keras.Model(inputs=[input1, input2], outputs=fc3, name='AF_classify')
model.summary()


model.compile(optimizer='adam',
               loss=custom_loss,
               metrics=['MSE']
              )

history = model.fit([x_train, x_train_mask], [y_train], batch_size=16, epochs=50)
