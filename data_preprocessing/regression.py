#####################################################
##均值为0操作
import numpy as np
matrix_load=[[1,2,3,4],[3,4,5,6]]
matrix_load = np.array(matrix_load,dtype=np.float32)
shifted_value = matrix_load.mean()
print("shifted_value:",shifted_value)
matrix_load -= shifted_value
print ("Data  shape: ", matrix_load.shape)
##减去均值后的数据
print(matrix_load)
##数据恢复
print(matrix_load+shifted_value)

####################################################

##################################################
##归一化标准化
train_x=np.array([[1,2,3],[3,0,8]],dtype=np.float32)
train_y=np.array([[1],[3],[4]],dtype=np.float32)
from sklearn import preprocessing
scale_x = preprocessing.scale(train_x)
scale_y= preprocessing.scale(train_y)

print("scale_x: " , scale_x)
print("scale_y: ", scale_y)

print("train_x: ",scale_x*train_x.std(axis=0)+train_x.mean(axis=0))
print("train_y: ",scale_y*train_y.std(axis=0)+train_y.mean(axis=0))
#######################################################


##################################################
##标准化
train_x=np.array([[1,2,3],[3,0,8]],dtype=np.float32)
train_y=np.array([[1],[3],[4]],dtype=np.float32)
from sklearn import preprocessing
scaler_x = preprocessing.StandardScaler().fit(train_x)
X_scaled = scaler_x.fit_transform(train_x)
scaler_y = preprocessing.StandardScaler().fit(train_y)
Y_scaled = scaler_y.fit_transform(train_y)

print("scale_x: " , X_scaled)
print("scale_y: ", Y_scaled)

print("train_x: ", X_scaled * scaler_x.scale_ + scaler_x.mean_)
print("train_y: ", Y_scaled * scaler_y.scale_ + scaler_y.mean_)
#######################################################


