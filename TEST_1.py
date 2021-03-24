import tensorflow as tf
import h5py
from scipy.io import loadmat as load
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.layers import Conv3D, BatchNormalization, Activation, MaxPool3D, Dropout, Flatten, Dense
import scipy.io as io
from scipy.io import loadmat as load

def feature_normalize(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


# feature = h5py.File('vid_set_1.mat')
feature = load('vid_set_1.mat')
x_train =feature['vid_set']
# x_train = x_train[0:1, :, :, :]
tf.bitcast(x_train, float)
data = load('bvp_set_nor_fil_1.mat')
y_train = data['bvp_set']
# y_train = y_train[0:1, :]
y_train = np.transpose(y_train)
tf.bitcast(y_train, float)
# y_train = np.random.rand(1, 330);
x_train = np.expand_dims(x_train, axis=0)
x_train = np.expand_dims(x_train, axis=4)

# -----------------------搭建网络-------------------------------
inputs = tf.keras.Input(shape=(200, 200, 330, 1), name='bvp')
c1 = tf.keras.layers.Conv3D(filters=64, kernel_size=(10, 10, 1), strides=[3, 3, 1], padding='valid')(inputs)
b1 = BatchNormalization()(c1)
a1 = Activation('tanh')(b1)
p1 = tf.keras.layers.AvgPool3D(pool_size=(10, 10, 1), strides=[3, 3, 1], padding='valid')(a1)  # 池化层
d1 = tf.keras.layers.Dropout(0.2)(p1) # dropout层

c2 = tf.keras.layers.Conv3D(filters=1, kernel_size=(5, 5, 1), strides=[3, 3, 1], padding='valid')(d1)
b2 = BatchNormalization()(c2)
a2 = Activation('tanh')(b2)
p2 = tf.keras.layers.AvgPool3D(pool_size=(10, 9, 1), strides=[3, 3, 1], padding='valid')(a2)  # 池化层
d2 = tf.keras.layers.Dropout(0.2)(p2) # dropout层

fc1 = tf.keras.layers.Flatten()(d2)
# fc2 = tf.keras.layers.Dense(300, activation='tanh')(fc1)
outputs = tf.keras.layers.Dense(330, activation='tanh')(fc1)

model = tf.keras.Model(inputs=inputs, outputs=fc1, name='AF_classify')
model.summary()

# -----------------------------------------------------ICCV2019-------------------------------------------------------
# inputs = tf.keras.Input(shape=(200, 200, 330, 1), name='bvp')
# c1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(10, 10, 9), strides=[3, 3, 1], padding='valid', activation='relu')(inputs)
#
# c21 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 1), strides=[2, 2, 1], padding='valid', activation='relu')(c1)
# c22 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 3), strides=[1, 1, 1], padding='valid',  activation='relu')(c21)
# c23 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 1), strides=[2, 2, 1], padding='valid', activation='relu')(c22)
# c24 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 3), strides=[1, 1, 1], padding='valid',  activation='relu')(c23)
# c25 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 1), strides=[2, 2, 1], padding='valid', activation='relu')(c24)
# c26 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 3), strides=[1, 1, 1], padding='valid',  activation='relu')(c25)
#
# p1 = tf.keras.layers.AvgPool3D(pool_size=(7, 7, 1), strides=[1, 1, 1], padding='valid')(c26)  # 池化层
# # p1 = tf.keras.layers.AvgPool3D(pool_size=(15, 15, 1), strides=[1, 1, 1], padding='valid')(c26)  # 池化层
# c3 = tf.keras.layers.Conv3D(filters=1, kernel_size=(1, 1, 1), strides=[1, 1, 1], padding='valid', activation='relu')(p1)  # 池化层
# fc1 = tf.keras.layers.Flatten()(c3)
# fc2 = tf.keras.layers.Dense(632, activation='tanh')(fc1)
# fc3 = tf.keras.layers.Dense(316, activation='tanh')(fc2)
# model = tf.keras.Model(inputs=inputs, outputs=fc3, name='AF_classify')
# model.summary()
# -------------------------------------------------------------------------------------------------------------------

model.compile(optimizer='adam',
               # loss='CosineSimilarity',
               loss='MSE',
               metrics=['MSE']
              )

# history = model.fit(x_train, y_train, batch_size=1, epochs=20, validation_split=0.2, validation_freq=1,
#                     )  # 看样子这是断点续训
history = model.fit(x_train, y_train[:, 7:-7], batch_size=1, epochs=15,
                     )

successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
successive_feature_maps = visualization_model.predict(x_train)

# io.savemat('mapData.mat', {'L1': successive_feature_maps[8]})

out = model.predict(x_train); plt.plot(out[0, :]); plt.show()
# out1 = model.predict(np.random.rand(1, 200, 200, 224, 1));plt.plot(out1[0, :])


# plt.subplot(4, 1, 1);plt.plot(out[0,:]);plt.subplot(4, 1, 2);plt.plot(out[1,:]);plt.subplot(4, 1, 3);plt.plot(out[2,:]);plt.subplot(4, 1, 4);plt.plot(out[3,:]);

# 显示训练集和验证集的acc和loss曲线
acc = history.history['ce']  # 训练集分类精度
val_acc = history.history['val_ce']  # 验证集分类精度
loss = history.history['loss']  # 训练集loss曲线
val_loss = history.history['val_loss']  # 验证集loss曲线

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()