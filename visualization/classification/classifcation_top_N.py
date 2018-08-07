from keras.models import Sequential,Model
from keras.layers import *

from keras_applications import vgg16
preprocess_input = vgg16.preprocess_input

def decode_predictions(preds, top=5):
  CLASS_INDEX={'0':['n100000','person'],'1':['n100001','banana']}
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results


model=Sequential()
model.add(Conv2D(32,3,strides=2,input_shape=(32,32,3),activation='relu',name='conv1'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(16,3,strides=2,activation='relu',name='conv2'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(2,activation='softmax'))
model.summary()


x=np.random.rand(1,32,32,3)
x = x.astype(np.float32)
##预处理使用imageNet的预处理
x=preprocess_input(x)
y=model.predict(x)

print("predictions :",decode_predictions(y,2))
exit()
#########################################################################################
##获得keras模型中某一层的输出

import keras.backend as K
def get_output_layer(model, layer_name):
  # get the symbolic outputs of each "key" layer (we gave them unique names).
  layer_dict = dict([(layer.name, layer) for layer in model.layers])
  layer = layer_dict[layer_name]
  return layer

final_conv_layer = get_output_layer(model, "conv2")
get_output = K.function([model.layers[0].input], [final_conv_layer.output])
[conv_outputs] = get_output([x])

print(conv_outputs.shape)
exit()
#########################################################################################

#########################################################################################
##获得第一层的输出
model = Model(inputs=model.input, outputs=model.get_layer('conv1').output)
conv1=model.predict(x)
print(conv1.shape)
exit()
#########################################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
conv1=conv1.reshape(conv1.shape[1:])
weights=np.sum(conv1,axis=2)
print(weights.shape)
plt.subplot(111)
sns.heatmap(weights)
plt.show()

