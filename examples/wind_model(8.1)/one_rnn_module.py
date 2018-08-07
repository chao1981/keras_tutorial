import scipy.io
import numpy as np
import csv

path = './data/chang/1'
reader = csv.reader(open(path + ".csv"))
train_y = []
for i in reader:
  train_y.append([float(j) for j in i])

train_y = np.array(train_y)
print(train_y.shape)

path = './data/NWP/CN0088'
reader = csv.reader(open(path + ".csv"))
train_x = []
for i in reader:
  train_x.append([float(j) for j in i])

train_x = np.array(train_x)
train_x = train_x[:, :25]
print(train_x.shape)

index = np.array([l for l in train_x[:, 0] if l in train_y[:, 0]])
trainx = []
for j in range(len(train_x)):
  if train_x[j, 0] in index:
    trainx.append(train_x[j, :])

trainy = []
for j in range(len(train_y)):
  if train_y[j, 0] in index:
    
    if len(trainy) > 0:
      if train_y[j, 0] not in trainy[:][0]:
        trainy.append(train_y[j, :])
    else:
      trainy.append(train_y[j, :])

train_y = np.array(trainy)
train_x = np.array(trainx)
print(train_x.shape)
print(train_y.shape)

from sklearn import preprocessing
train_x = preprocessing.scale(train_x[:,2:])
train_y= preprocessing.scale(train_y[:,1:])


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint

input_shape = Input(shape=(x_train.shape[1], x_train.shape[2]))
x = LSTM(64,  return_sequences=True)(input_shape)
x = LSTM(32)(x)
x = Dense(2, activation='linear')(x)
model = Model(inputs=input_shape, outputs=x)
model.summary()
# from keras.utils import plot_model
# plot_model(model,"cnn.png",show_shapes=True)
# exit()
model.compile(loss="mse", optimizer="rmsprop")

checkpoint = ModelCheckpoint('./weights/rnn_weights.h5',  # model filename
                             monitor='val_loss',  # quantity to monitor
                             verbose=1,  # verbosity - 0 or 1
                             save_best_only=True,  # The latest best model will not be overwritten
                             mode='min')  # The decision to overwrite model is m

model.load_weights("./weights/rnn_weights.h5")
# model.fit(x_train, y_train, batch_size=512, epochs=600, shuffle=True,validation_split=0.5, callbacks=[checkpoint],verbose=1)

test_mse = model.evaluate(x_test, y_test, verbose=1)
print('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(y_test)))

predicted_values = model.predict(x_test)
num_test_samples = len(predicted_values)
print(num_test_samples)
print(predicted_values.shape)
print(predicted_values[29, :])
print(y_test[29, :] )

import matplotlib.pyplot as plt

# plot the results
plt.subplot(111)
plt.plot(y_test[:300, 0], 'b',label="true power odf first channel")
plt.plot(predicted_values[:300, 0], 'r',label="predicted power of first channel")
plt.xlabel('times')
plt.ylabel('the power value of first channel')
plt.legend()
# plt.savefig('output_power0.png', bbox_inches='tight',dpi=300)
plt.show()

plt.subplot(111)
plt.plot(y_test[:300, 1], 'b',label="true power of second channel")
plt.plot(predicted_values[:300, 1], 'r',label="predicted power of second channel")
plt.xlabel('times')
plt.ylabel('the power value of second channel')
plt.legend()
# plt.savefig('output_power1.png', bbox_inches='tight',dpi=300)
plt.show()
