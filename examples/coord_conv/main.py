import keras
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.utils import plot_model
from keras.optimizers import RMSprop,Adam
from keras.models import save_model
from keras.callbacks import ModelCheckpoint

############## data loading#####################
train_Y=np.array([[i,j]for i in range(10) for j in range(10)])/10
print(train_Y.shape)
X=[]
for i in range(10):
  for j in range(10):
    temp=np.zeros((100,100))
    temp[i*10:(i+1)*10,j*10:(j+1)*10]=1
    X.append(temp.reshape(100,100,1))
train_X=np.array(X)
print(train_X.shape)
#########显示#############################
# import cv2
# cv2.imshow("test",label)
# cv2.waitKey(3000)
# cv2.destroyAllWindows()
#########显示###################################
# import matplotlib.pyplot as plt
# plt.imshow(train_X[99].reshape(100,100),cmap=plt.cm.gray)#plt.cm.gray
# plt.show()
################################################

######################模型１##################################
# model=Sequential()
# model.add(Conv2D(32, (3, 3),input_shape=train_X.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# # model.add(Dense(64))
# # model.add(Activation('relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(2,activation='sigmoid'))
#
#
# rps=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
# model.compile(loss='mse',optimizer=rps)
# model.summary()
# plot_model(model,show_shapes=True)
# checkpoint = ModelCheckpoint('sigmoid_model.h5', monitor='loss',
#                              verbose=1, save_best_only=True, save_weights_only=True)
# # model.load_weights("model.h5")
# model.fit(train_X[:80],train_Y[:80],batch_size=32,epochs=100,shuffle=True,callbacks=[checkpoint])
# model.save_weights("sigmoid_model.h5")
# model.load_weights("sigmoid_model.h5")
# temp=np.zeros((100,100))
# temp[48:58,48:58]=1
# y=model.predict(temp.reshape(1,100,100,1))
# print(y)
# # print(train_Y[:10])
###############################################################

######################模型１upsample##################################
# model=Sequential()
# model.add(Dense(512, kernel_initializer='glorot_normal',input_shape=(2,)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(8*25*25, kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Reshape([25, 25, 8]))
# model.add(UpSampling2D(size=(4, 4)))
#
# # model.add(Conv2D(32, (1, 1), padding='same', kernel_initializer='glorot_normal'))
# # model.add(BatchNormalization())
# # model.add( Activation('relu'))
#
# model.add(Conv2D(16, (1, 1), padding='same', kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Convolution2D(1, (1, 1), padding='same', kernel_initializer='glorot_normal'))
# # model.add(Activation('sigmoid'))
#
# model.summary()
# rps=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
# model.compile(loss='mse', optimizer=rps)
# model.load_weights("4upsample_model.h5")
# model.fit(train_Y,train_X,batch_size=16,epochs=100,shuffle=True,validation_split=0.2)
# model.save_weights("4upsample_model.h5")
# model.load_weights("4upsample_model.h5")
# x=model.predict(train_Y[40:41])
# import matplotlib.pyplot as plt
# plt.subplot(121)
# plt.imshow(train_X[40].reshape(100,100),cmap=plt.cm.gray)#plt.cm.gray
# plt.subplot(122)
# plt.imshow(x.reshape(100,100),cmap=plt.cm.gray)#plt.cm.gray
# plt.show()
# ###############################################################

######################模型2Deconv##################################
model=Sequential()
model.add(Dense(512, kernel_initializer='glorot_normal',input_shape=(2,)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(8*25*25, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Reshape([25, 25, 8]))
model.add(Deconvolution2D(32,kernel_size=4,strides=4))

model.add(Conv2D(16, (1, 1), padding='same', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(1, (1, 1), padding='same', kernel_initializer='glorot_normal'))
model.add(Activation('sigmoid'))

model.summary()
rps=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss='mse', optimizer=rps)

# model.load_weights("sigmoid_deconv_model.h5")
# checkpoint = ModelCheckpoint('sigmoid_deconv_model.h5', monitor='val_loss',
#                              verbose=1, save_best_only=True, save_weights_only=True)
# model.fit(train_Y,train_X,batch_size=16,epochs=100,shuffle=True,callbacks=[checkpoint],validation_split=0.2)
# model.save_weights("sigmoid_deconv_model.h5")
model.load_weights("sigmoid_deconv_model.h5")
x=model.predict(train_Y[45:46])
import matplotlib.pyplot as plt
plt.subplot(121)
plt.imshow(train_X[45].reshape(100,100),cmap=plt.cm.gray)#plt.cm.gray
plt.subplot(122)
plt.imshow(x.reshape(100,100),cmap=plt.cm.gray)#plt.cm.gray
plt.show()
# ###############################################################