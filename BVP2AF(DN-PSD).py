import tensorflow as tf
from scipy.io import loadmat as load
import numpy as np
from tensorflow.python.keras import regularizers

## updated by authors of ICCV manuscript 6416.
## 2021/03/31

def feature_normalize(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


x_train_face = load('training_samples_face.mat') # VPPG pulse signals from face videos
x_train_finger = load('training_samples_finger.mat') # PPG pulse signals from fingertips
y_train = load('training_labels.mat') # lables for AF classification


# -----------------------Sparse representation in DN-PSD-------------------------------
inputs = tf.keras.Input(shape=(1, 600), name='raw_bvp')
d1 = tf.keras.layers.Dense(900, activation='tanh', activity_regularizer=regularizers.l1(10e-2))(inputs) # encoder
outputs = tf.keras.layers.Dense(600, use_bias=False)(d1)          # decoder
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='PSD')
model.summary()

model.compile(optimizer='adam',
              loss='mse',
              metrics='mse',
              )

history_PSD = model.fit(x_train_face, x_train_finger, batch_size=16, epochs=30, validation_split=0.2)
prediction = model.predict(x_train)

# extract feature layers from the trained DN-PSD
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
successive_feature_maps = visualization_model.predict(x_train)

# -----------------------AF classifier in DN-PSD-------------------------------
inputs = tf.keras.Input(shape=(1, 900), name='bvp_feature')
c1 = tf.keras.layers.Dense(450, activation='sigmoid')(inputs)
outputs = tf.keras.layers.Dense(2, activation='sigmoid')(c1)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='AF_classify')
model.summary()

METRICS = [
    tf.keras.metrics.categorical_accuracy(name='accuracy'),
    tf.keras.metrics.AUC(name='AUC'),
]

model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy(),
              metrics=METRICS,
              )

history_class = model.fit(successive_feature_maps[0], y_train, batch_size=16, epochs=30, validation_split=0.2)
