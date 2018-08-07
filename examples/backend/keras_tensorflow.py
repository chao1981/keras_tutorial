import tensorflow as tf
import numpy as np
from time import time
from keras.layers import Dense
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)


# Keras layers can be called on TensorFlow tensors:
img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))
x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation


loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


start=time()
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch = mnist_data.train.next_batch(50)
        m_loss,step=sess.run([loss,train_step],feed_dict={img: batch[0],labels: batch[1]})
        if i %1000==0:
          print("epoch %d:"%i,"loss:",m_loss)

    acc_value = accuracy(labels, preds)
    result=acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels})
    print(np.sum(result)/len(mnist_data.test.labels))
    print("total time:",time()-start)
#######################################################################################################
##将代码换成原生的tensorflow程序
##
#######################################################################################################

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from time import time

Dense = tf.keras.layers.Dense
accuracy = tf.keras.metrics.categorical_accuracy
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# Keras layers can be called on TensorFlow tensors:
img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))
x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation

loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, preds))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

start=time()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10000):
    batch = mnist_data.train.next_batch(50)
    m_loss, step = sess.run([loss, train_step], feed_dict={img: batch[0], labels: batch[1]})
    if i % 1000 == 0:
      print("epoch %d:" % i, "loss:", m_loss)

  acc_value = accuracy(labels, preds)
  result = acc_value.eval(feed_dict={img: mnist_data.test.images,
                                     labels: mnist_data.test.labels})
  print(np.sum(result) / len(mnist_data.test.labels))
  print("total time:",time()-start)

