from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import *
from keras.utils import plot_model
from sklearn import preprocessing
from keras.optimizers import RMSprop
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

###############data loading###########################################
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

y_train=np.reshape(y_train,(-1,1))
y_test=np.reshape(y_test,(-1,1))
print(y_train.shape)
print(x_train[0])
ss_x = preprocessing.StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_x.fit_transform(y_train)
y_test = ss_x.transform(y_test)
print(x_train[0])

###################model############################################

model=Sequential()
model.add(Dense(32,activation='relu',kernel_initializer='normal',input_shape=(x_train.shape[1],)))
model.add(Dense(16,activation='relu',kernel_initializer='normal'))
model.add(Dense(1))
rps=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss='mse',optimizer=rps)

model.fit(x_train,y_train,batch_size=32,epochs=500,validation_data=(x_test,y_test))

####################################################
print("test cost",model.evaluate(x_test,y_test,batch_size=32))


##################model2################################
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
model2=Sequential()
model2.add(LSTM(512,input_shape=(x_train.shape[1],x_train.shape[2]),return_sequences=True))
model2.add(Dropout(0.5))
model2.add(LSTM(256))
model2.add(Dropout(0.5))
model2.add(Dense(1,kernel_initializer='normal'))
model2.compile(loss='mse',optimizer='sgd')
model2.fit(x_train,y_train,batch_size=32,epochs=500,validation_data=(x_test,y_test))
print("test cost",model2.evaluate(x_test,y_test,batch_size=32))



# 多层感知器-回归模型
model_mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(20, 20, 20), random_state=1)
model_mlp.fit(x_train,y_train)
mlp_score=model_mlp.score(x_test,y_test)
print('sklearn多层感知器-回归模型得分',mlp_score)


model_gbr_disorder=GradientBoostingRegressor()
model_gbr_disorder.fit(x_train,y_train)
gbr_score_disorder=model_gbr_disorder.score(x_test,y_test)
print('sklearn集成-回归模型得分',gbr_score_disorder)#准确率较高 0.853817723868



from sklearn.linear_model import LinearRegression
linear = LinearRegression().fit(x_train,y_train)
predicted_price = linear.predict(x_test)
r2_score = linear.score(x_test,y_test)*100
print("linerRegression:",r2_score)

import matplotlib.pyplot as plt


fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
axes = fig.add_subplot(1, 1, 1)
line3, = axes.plot(range(len(y_test)), y_test, 'g', label='实际')
line2, = axes.plot(range(len(y_test)), model.predict(x_test), 'r--', label='模型', linewidth=2)
line1, = axes.plot(range(len(y_test)), linear.predict(x_test), 'b--', label='ensemble', linewidth=2)
# line0, = axes.plot(range(len(y_test)), model_mlp.predict(x_test), 'g--', label='mlp', linewidth=2)
axes.grid()
fig.tight_layout()
# plt.legend(handles=[line0,line1,line2, line3])
plt.legend(handles=[line1,line2, line3])
plt.title('sklearn 回归模型')
plt.show()
