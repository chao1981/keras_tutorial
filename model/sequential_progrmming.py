from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import to_categorical

import numpy as np


##数据预处理
train_x=np.random.random((1000,100))
labels=np.random.randint(10,size=(1000,1))
train_y=to_categorical(labels,num_classes=10)
test_x=np.random.random((20,100))
labels=np.random.randint(10,size=(20,1))
test_y=to_categorical(labels,num_classes=10)




## 第1种方法
# model=Sequential([
#   Dense(32,input_dim=train_x.shape[1]),
#   Activation("relu"),
#   Dense(10),
#   Activation('softmax')
# ])
# model.summary()

##第2种方法
model=Sequential()
model.add(Dense(32,input_shape=(train_x.shape[1],)))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()


##编译
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
# model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
# model.compile(optimizer='rmsprop',loss='mse')

##数据训练
model.fit(train_x,train_y,epochs=100,batch_size=32,shuffle=True,validation_data=(test_x,test_y))


##模型验证
score = model.evaluate(test_x, test_y,batch_size=16)
print(score)
print(model.predict(test_x))

print(test_y)