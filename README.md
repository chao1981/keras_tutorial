keras tutorial
===


### keras常用层
```
from keras.layers import *
from keras.models import Sequential
model=Sequential()
##全连接层
model.add(Dense(1,input_shape=(10,)))##输出的维度为１
or
model.add(Dense(1,input_dim=10))##输出的维度为１

```
```
##Activation层
model.add(Activation('sigmoid')
model.add(Activation('softmax')
model.add(Activation('relu')
model.add(Activation('tanh')
model.add(Activation('linear')

```


```
##卷积层
model.add(Conv2D(32, (3, 3), border_mode='same'))
```

```
##池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
```
```
##循环层
LSTM,GRU,recurrent.GRU
model.add(LSTM(32, input_dim=64, input_length=10)) ##模型的第一层
model.add(LSTM(256,return_sequences=True))##(,100,10)->(,100,256)
model.add(LSTM(256))##(,100,10)->(,256)
```

```
##StackedRNNCells层
cells = [
    keras.layers.LSTMCell(output_dim),
    keras.layers.LSTMCell(output_dim),
    keras.layers.LSTMCell(output_dim),
]
inputs = keras.Input((timesteps, input_dim))
x = keras.layers.StackedRNNCells(cells)(inputs)
```

```
##CuDNNLSTM层
model.add(CuDNNLSTM(32, input_dim=64, input_length=10)) ##模型的第一层
model.add(CuDNNLSTM(256,return_sequences=True))##(,100,10)->(,100,256)
model.add(CuDNNLSTM(256))##(,100,10)->(,256)
```

```
##Embedding嵌入层

model = Sequential()
model.add(Embedding(input_dim=10000,output_dim=64, input_length=10))
input_array = np.random.randint(1000, size=(32, 10))
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```

```
##Merge层

import keras
input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])
out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

```
##批处理层
model.add(BatchNormalization())##中间无参数
##Dropout层
model.add(Dropout(0.5))##随机去掉50%
```
```
#噪声层
noise.GaussianNoise(sttdev)
```

```
包装器Wrapper
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# now model.output_shape == (None, 10, 8)

model.add(TimeDistributed(Dense(32)))
# now model.output_shape == (None, 10, 32)
```

```
#Bidirectional包装器

model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

```

### 编写自己的层

```
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```


### 损失函数
```
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
loss:
1.mse
2.mae
3.squared_hinge
4.hinge
5.catrgorical_hinge
6.binary_crossentary
7.categorical_crossentropy
from keras.utils.np_utils import to_categorical
categorical_labels = to_categorical(int_labels, num_classes=None)

8.sparse_categorical_crossentrop
9.kullback_leibler_divergence:从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.
10.poisson：即(predictions - targets * log(predictions))的均值
11.cosine_proximity：即预测值与真实标签的余弦距离平均值的相反数

```

###正则项

```
kernel_regularizer：施加在权重上的正则项，为keras.regularizer.Regularizer对象
bias_regularizer：施加在偏置向量上的正则项，为keras.regularizer.Regularizer对象
activity_regularizer：施加在输出上的正则项，为keras.regularizer.Regularizer对象

from keras import regularizers
model.add(Dense(64, input_dim=64,kernel_regularizer=regularizers.l2(0.01),
activity_regularizer=regularizers.l1(0.01)))

##添加自定义的正则项
from keras import backend as K
def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg)
```
### 陷阱注意
```
 from keras.layers import concatenate 
 from keras.layers import Concatenate
 layer1=concatenate([out_a,out_b]) ##right
 layer1=Concatenate([out_a,out_b]) ##wrong

```
### [keras后端切换](http://keras-cn.readthedocs.io/en/latest/backend/)
Keras是一个模型级的库，提供了快速构建深度学习网络的模块。Keras并不处理如张量乘法、卷积等底层操作。
这些操作依赖于某种特定的、优化良好的张量操作库。Keras依赖于处理张量的库就称为“后端引擎”。Keras提供
了三种后端引擎Theano/Tensorflow/CNTK，并将其函数统一封装，使得用户可以以同一个接口调用不同后端引擎的
函数．</br>

* 切换后端
```
##window用户
修改keras.json
$HOME/.keras/keras.json
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}

修改backend即可切换tensorflow,theano,CNTK.


也可以通过命令行修改:
>>>KERAS_BACKEND=tensorflow python -c "from keras import backend;"

```


