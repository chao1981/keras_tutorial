# ##############################keras###########################################
# from keras.models import Sequential
# from keras.layers import *
# from keras.utils import plot_model
# import numpy as np
#
# model=Sequential()
# model.add(Deconvolution2D(3,3,strides=(1,1),padding='valid',input_shape=(2,2,1)))
# model.summary()
# x=np.random.rand(1,2,2,1)
# y=model.predict(x)
# print(x)
# print(y.shape)
#
# model=Sequential()
# model.add(Conv2DTranspose(3,3,strides=(1,1),padding='valid',input_shape=(2,2,1)))
# model.summary()
#
# y=model.predict(x)
# print(y.shape)
# #############################################################################
#
#
# ################################pytorch########################################
# import torch.nn as nn
# import torch
# conv=nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,padding=0)
# input=torch.randn(1,1,5,5) ##(batch,channel,h,w)
# conv_output=conv(input)
# print(conv_output.size())
# deconv=nn.ConvTranspose2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0,output_padding=0)
# ##Lout=(Lin−1)∗stride−2∗padding+kernel_size+output_padding
# deconv_output=deconv(conv_output)
# print(deconv_output.size())
###############################################################################


###############################################################################
import numpy as np
from keras.models import Sequential
from keras.layers import *

x=np.array([[1, 0, 1, 1, 1],
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1]
            ])
model=Sequential()
x=x.reshape((-1,5,5,1))

model.add(Conv2D(1,3,name='conv1')) ##conv filters shape(h,w,in,out)
conv_filter=model.predict(x)
print(conv_filter.shape)
print(conv_filter)


model1=Sequential()
layer=Conv2DTranspose(1,3,name='deconv')
model1.add(layer)
z=model1.predict(conv_filter)
# print(z)
weights=model.get_layer('conv1').get_weights()
kernel=weights[0]
kernel=kernel.reshape(3,3)
de_kernel=kernel.T
weights[0]=de_kernel.reshape(3,3,1,1)
model1.get_layer('deconv').set_weights(weights)

deconv_filter=model1.predict(conv_filter)
print(deconv_filter.shape)

print(deconv_filter.reshape(5,5))
print(x.reshape(5,5))

import matplotlib.pyplot as plt
plt.subplot(121)
plt.imshow(x.reshape(5,5))
plt.subplot(122)
plt.imshow(deconv_filter.reshape(5,5))
plt.show()
