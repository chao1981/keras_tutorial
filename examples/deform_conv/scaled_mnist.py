from __future__ import division
# %env CUDA_VISIBLE_DEVICES=0

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from deform_conv.layers import ConvOffset2D
from deform_conv.callbacks import TensorBoard
from deform_conv.cnn import get_cnn, get_deform_cnn
from deform_conv.mnist import get_gen
from keras.callbacks import ModelCheckpoint

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# ---
# Config

batch_size = 32
n_train = 60000
n_test = 10000
steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = int(np.ceil(n_test / batch_size))

train_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=True
)
test_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=False
)
train_scaled_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=True
)
test_scaled_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=False
)
train_horizontal_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=True,
  horizontal_flip=True
)

test_horizontal_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=False,
  horizontal_flip=True
)

# ---
# Normal CNN

inputs, outputs = get_cnn()
model = Model(inputs=inputs, outputs=outputs)
# model.summary()
optim = Adam(1e-3)
loss = categorical_crossentropy
model.compile(optim, loss, metrics=['accuracy'])
checkpoint = ModelCheckpoint('models/horizon_cnn.h5',  # model filename
                             monitor='val_acc', # quantity to monitor
                             verbose=1, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='max') # The decision to overwrite model is m
model.fit_generator(
  train_gen, steps_per_epoch=steps_per_epoch,
    epochs=10, verbose=1,
    validation_data=test_gen,callbacks=[checkpoint],validation_steps=validation_steps
)




###############################################################
# Deformable CNN

inputs, outputs = get_deform_cnn(trainable=True)
model = Model(inputs=inputs, outputs=outputs)
# model.load_weights('models/horizon_cnn.h5',by_name=True)
# model.summary()
optim = Adam(1e-3)
loss = categorical_crossentropy
model.compile(optim, loss, metrics=['accuracy'])
checkpoint = ModelCheckpoint('models/deform_cnn.h5',  # model filename
                             monitor='val_acc', # quantity to monitor
                             verbose=1, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='max') # The decision to overwrite model is m

model.fit_generator(
  train_gen, steps_per_epoch=steps_per_epoch,
    epochs=10, verbose=1,
    validation_data=test_gen,callbacks=[checkpoint], validation_steps=validation_steps
)


deform_conv_layers = [l for l in model.layers if isinstance(l, ConvOffset2D)]

Xb, Yb = next(test_gen)
for l in deform_conv_layers:
    print(l)
    _model = Model(inputs=inputs, outputs=l.output)
    offsets = _model.predict(Xb)
    offsets = offsets.reshape(offsets.shape[0], offsets.shape[1], offsets.shape[2], -1, 2)
    print(offsets.min())
    print(offsets.mean())
    print(offsets.max())
