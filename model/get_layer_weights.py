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



