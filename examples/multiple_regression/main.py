from keras.models import Sequential
from keras.layers import *
from time import time
import numpy as np
from keras.utils import plot_model
###################data loading#############################

x_train=np.random.rand(3200,1)
y_train=np.random.rand(3200,10)

#####################model###############################
model=Sequential()
model.add(Dense(256,input_shape=(x_train.shape[1],),name='layer1'))
model.add(Dense(10,name='layer2'))
model.compile(optimizer='sgd',loss='mse')
plot_model(model,"model.png",show_shapes=True)
h=model.fit(x_train,y_train,batch_size=32,epochs=10,validation_split=0.4)

X_test=np.random.rand(1,1)
print(X_test)
print(model.predict(X_test))

print(h.history)
print(model.get_layer(name='layer2').output)
print(model.get_layer(name='layer2').output.shape)
print(model.get_layer(name='layer2').get_weights()[0].shape)
print(model.get_layer(name='layer1').get_weights()[0].shape)


"""
##layer层参数
1. layer.input
2. layer.output
3. layer.input_shape
4. layer.output_shape

##共享层节点
1. layer.get_input_at(node_index)
2. layer.get_output_at(node_index)
3. layer.get_input_shape_at(node_index)
4. layer.get_output_shape_at(node_index)

##examples:

a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))
conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)
assert conv.input_shape == (None, 32, 32, 3)
conved_b = conv(b)
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)
"""

