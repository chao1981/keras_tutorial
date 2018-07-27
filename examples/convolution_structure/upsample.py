from keras.models import Sequential
from keras.layers import *
from keras.utils import plot_model
import numpy as np


model=Sequential()
model.add(UpSampling2D(size=(2,2),input_shape=(2,2,1)))
model.summary()
plot_model(model,"up_sample.png",show_shapes=True)
x=np.random.rand(1,2,2,1)
y=model.predict(x)
print(x)
print(y)