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

test_horizontal_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=False,
  horizontal_flip=True
)

test_vertical_flip_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=False,vertical_flip=True
)
test_flip_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=False,vertical_flip=True,
  horizontal_flip=True
)



inputs, outputs = get_cnn()
model = Model(inputs=inputs, outputs=outputs)
optim = Adam(1e-3)
loss = categorical_crossentropy
model.compile(optim, loss, metrics=['accuracy'])
model.load_weights("models/horizon_cnn.h5")
print(model.layers[-2].get_weights()[0].shape)


print("standard cnn results:")
print("#####################################################")
val_loss, val_acc = model.evaluate_generator(
    test_gen, steps=validation_steps
)
print('Test accuracy with images', val_acc)

val_loss, val_acc = model.evaluate_generator(
    test_scaled_gen, steps=validation_steps
)
print('Test accuracy with scaled images', val_acc)


val_loss, val_acc = model.evaluate_generator(
    test_horizontal_gen, steps=validation_steps
)
print('Test accuracy with horizontal_filp images', val_acc)

val_loss, val_acc = model.evaluate_generator(
    test_vertical_flip_gen, steps=validation_steps
)
print('Test accuracy with vertical_flip images', val_acc)

val_loss, val_acc = model.evaluate_generator(
    test_flip_gen, steps=validation_steps
)
print('Test accuracy with flip images', val_acc)

print("#####################################################\n")




print("deform cnn result:")
print("#####################################################")
inputs, outputs = get_deform_cnn(trainable=False)
deform_model = Model(inputs=inputs, outputs=outputs)
optim = Adam(1e-3)
loss = categorical_crossentropy
deform_model.compile(optim, loss, metrics=['accuracy'])
deform_model.load_weights('models/deform_cnn.h5')

val_loss, val_acc = deform_model.evaluate_generator(
    test_gen, steps=validation_steps
)
print('Test accuracy with  images', val_acc)

val_loss, val_acc = deform_model.evaluate_generator(
    test_scaled_gen, steps=validation_steps
)
print('Test accuracy with scaled images', val_acc)


val_loss, val_acc = deform_model.evaluate_generator(
    test_horizontal_gen, steps=validation_steps
)
print('Test accuracy with horizontal_filp images', val_acc)

val_loss, val_acc = deform_model.evaluate_generator(
    test_vertical_flip_gen, steps=validation_steps
)
print('Test accuracy with vertical_flip images', val_acc)

val_loss, val_acc = deform_model.evaluate_generator(
    test_flip_gen, steps=validation_steps
)
print('Test accuracy with flip images', val_acc)

print("#####################################################")
exit()


Xb, Yb = next(test_gen)

import keras.backend as K
deform_conv_layers = [l.output for l in deform_model.layers if isinstance(l, ConvOffset2D)]
get_output = K.function([deform_model.layers[0].input], deform_conv_layers)
result_ouput= get_output([Xb])


import matplotlib.pyplot as plt

plt.subplot(141)
plt.imshow(Xb[0].reshape(28,28))

i=0
for offsets in result_ouput:
    offsets=np.array(offsets)
    plt.subplot(142+i)
    i+=1
    map=offsets[0].reshape((offsets.shape[1],offsets.shape[2],offsets.shape[3]))
    map=np.sum(map,axis=2)
    plt.imshow(map)
    print(offsets.min())
    print(offsets.mean())
    print(offsets.max())
plt.show()
