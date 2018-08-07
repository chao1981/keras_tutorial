import numpy as np
import tensorflow as tf

from keras import Sequential
from keras.layers import Dense

print("Tensorflow version", tf.__version__)

model1 = Sequential()
model1.add(Dense(10, input_shape=(1000,)))

model1.add(Dense(3, activation='relu'))
model1.compile('sgd', 'mse')


def gen():
    while True:
        yield np.zeros([10, 1000]), np.ones([10, 3])


import os
import psutil

process = psutil.Process(os.getpid())
g = gen()
while True:
    print(process.memory_info().rss / float(2 ** 20))
    model1.fit_generator(g, 100, 2, use_multiprocessing=True, verbose=0)
    model1.evaluate_generator(gen(), 100, use_multiprocessing=True, verbose=0)



