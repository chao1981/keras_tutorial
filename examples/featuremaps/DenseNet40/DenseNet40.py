from __future__ import print_function

import os.path

import densenet
import numpy as np
from keras.optimizers import Adam
from keras import backend as K

batch_size = 64
nb_classes = 10
nb_epoch = 300

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 16
dropout_rate = 0.0 # 0.0 for data augmentation

model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                          growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate)
print("Model created")
optimizer = Adam(lr=1e-3) # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
from keras.utils import plot_model
plot_model(model,"densenet40.svg",show_shapes=True)

# print(model.predict([np.random.rand(1,32,32,3)]).shape)

print("load success!")

import matplotlib.pyplot as plt
import seaborn as sns


weights_heat = []
for layer in model.layers:
  for i in range(2, 14):
    if layer.name == "conv2d_%d"%(i):
      temp_layer = layer.get_weights()[0]
      temp = []

      number = np.sum(np.fabs(temp_layer[:, :, :24, :]), axis=(0,1,2, 3)) / 24/9/12
      temp.append(number)
      if temp_layer.shape[2] / 12 > 2:
        for k in range(1, int(temp_layer.shape[2] / 12) -1):
          number = np.sum(np.fabs(temp_layer[:, :, 12+12*k:12*k+24, :]), axis=(0, 1, 2, 3)) / 12/ 9 / 12
          temp.append(number)
      weights_heat.append(temp)


heat = np.zeros((len(weights_heat[-1]), len(weights_heat[-1])))
for i in range(len(weights_heat)):
  for j in range(len(weights_heat[i])):
    heat[j, i] = weights_heat[i][j]

heat[0,0]=1
for i in range(1,len(heat)):
  min=heat[:i+1, i].min()
  max=heat[:i+1, i].max()
  for j in range(i+1):
    if heat[j,i]!=0:
      heat[j,i]=(heat[j,i]-min)/(max-min)

plt.rcParams['figure.figsize'] = (12, 12)
plt.subplot(222)
plt.imshow(heat, cmap='rainbow')
plt.colorbar()
plt.show()


# weights_heat = []
# for layer in model.layers:
#   for i in range(2, 14):
#     if layer.name == "conv2d_%d"%(i):
#       temp_layer = layer.get_weights()[0]
#       temp = []
#       number = np.sum(np.fabs(temp_layer[:, :,temp_layer.shape[2]-24 :temp_layer.shape[2], :]), axis=(2,3))/24
#       temp.append(np.linalg.norm(number,ord=1))
#       if temp_layer.shape[2] / 12 > 2:
#         for k in range(1, int(temp_layer.shape[2] / 12) -1):
#           number = np.sum(np.fabs(temp_layer[:, :, temp_layer.shape[2]-12*k-24:temp_layer.shape[2]-12-12*k, :]), axis=(2,3))/12
#           temp.append(np.linalg.norm(number, ord=1))
#       weights_heat.append(temp)
#
#
# heat = np.zeros((len(weights_heat[-1]), len(weights_heat[-1])))
# for i in range(len(weights_heat)):
#   for j in range(len(weights_heat[i])):
#     heat[j, i] = weights_heat[i][j]
# print(heat)
# heat[0,0]=1
# for i in range(1,len(heat)):
#   min=heat[:i+1, i].min()
#   max=heat[:i+1, i].max()
#   for j in range(i+1):
#     if heat[j,i]!=0:
#       heat[j,i]=(heat[j,i]-min)/(max-min)
#
# plt.rcParams['figure.figsize'] = (12, 12)
# plt.subplot(221)
# plt.imshow(heat)
# plt.colorbar()
# # plt.show()

# weights_heat = []
# for layer in model.layers:
#   for i in range(2, 14):
#     if layer.name == "conv2d_%d"%(i):
#       temp_layer = layer.get_weights()[0]
#       temp = []
#       number = np.sum(np.fabs(temp_layer[:, :,temp_layer.shape[2]-24 :temp_layer.shape[2], :]), axis=(0,1,2, 3)) / 24/9/12
#       temp.append(number)
#       if temp_layer.shape[2] / 12 > 2:
#         for k in range(1, int(temp_layer.shape[2] / 12) -1):
#           number = np.sum(np.fabs(temp_layer[:, :, temp_layer.shape[2]-24-12*k:temp_layer.shape[2]-12*k-12, :]), axis=(0, 1, 2, 3)) / 12/ 9 / 12
#           temp.append(number)
#       weights_heat.append(temp)
#
#
# heat = np.zeros((len(weights_heat[-1]), len(weights_heat[-1])))
# for i in range(len(weights_heat)):
#   for j in range(len(weights_heat[i])):
#     heat[j, i] = weights_heat[i][j]
#
# heat[0,0]=1
# for i in range(1,len(heat)):
#   min=heat[:i+1, i].min()
#   max=heat[:i+1, i].max()
#   for j in range(i+1):
#     if heat[j,i]!=0:
#       heat[j,i]=(heat[j,i]-min)/(max-min)
#
# plt.rcParams['figure.figsize'] = (12, 12)
# plt.subplot(221)
# plt.imshow(heat, cmap='rainbow')
# plt.colorbar()





