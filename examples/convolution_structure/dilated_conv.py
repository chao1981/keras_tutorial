##############################keras###########################################
from keras.models import Sequential
from keras.layers import *
from keras.utils import plot_model
import numpy as np
model=Sequential()
model.add(Conv2D(3,3,strides=1,dilation_rate=1,input_shape=(4,4,1)))#(16,3,strides=(1,1),padding='valid',input_shape=(4,4,1)))
model.summary()
x=np.random.rand(1,4,4,1)
y=model.predict(x)

print(y.shape)
#############################################################################


################################pytorch########################################
import torch.nn as nn
import torch
conv=nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,dilation=1)
# input=torch.randn(1,1,5,5) ##(batch,channel,h,w)
x=x.transpose(0,3,1,2)
x=torch.Tensor(x)
conv_output=conv(x)
print(conv_output.size())
y=y.transpose(0,3,1,2)
print(y)
print(conv_output)

#############################################################################