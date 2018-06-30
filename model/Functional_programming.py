from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import Input, Dense,Conv2D,Flatten,concatenate,MaxPooling2D,Embedding,LSTM
from keras.models import Model
from keras.layers import TimeDistributed
from keras.utils import plot_model
##模型
# 
# inputs = Input(shape=(784,))
# x = Dense(64, activation='relu')(inputs)
# x = Dense(64, activation='relu')(x)
# predictions = Dense(10, activation='softmax')(x)
# model = Model(inputs=inputs, outputs=predictions)
# # model = Model(inputs=[inputs,inputs], outputs=[predictions,predictions])
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# model.summary()
# input_sequences = Input(shape=(20, 784))
# processed_sequences = TimeDistributed(model)(input_sequences)  ##用于视频分类
# print(processed_sequences.shape)

# model的属性
# model.layers：组成模型图的各个层
# model.inputs：模型的输入张量列表
# model.outputs：模型的输出张量列表


#####################共享视觉模块#############################
"""
该模型在两个输入上重用了图像处理的模型，用来判别两个MNIST数字是否是相同的数字.
"""
# input=Input(shape=(28,28,1))
# conv1=Conv2D(32,(3,3),strides=2,activation='relu')(input)
# conv2=Conv2D(64,(3,3),strides=2,activation='relu')(conv1)
# out=Flatten()(conv2)
# out=Dense(10,activation='softmax')(out)
# visual_model=Model(inputs=input,outputs=out)
# visual_model.summary()
#
# input_a=Input(shape=(28,28,1))
# input_b=Input(shape=(28,28,1))
# out_a=visual_model(input_a)
# out_b=visual_model(input_b)
# layer1=concatenate([out_a,out_b])
# out=Dense(1,activation='sigmoid')(layer1)
#
# model=Model(inputs=[input_a,input_b],outputs=out)
# model.summary()
# plot_model(model,"model.png")


########################视觉问答模型##################################
"""
在针对一幅图片使用自然语言进行提问时，该模型能够提供关于该图片的一个单词的答案
这个模型将自然语言的问题和图片分别映射为特征向量，将二者合并后训练一个logistic回归层，从一系列可能的回答中挑选一个。
"""
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())
input=Input(shape=(224,224,3))
out_a=vision_model(input)

question_input=Input(shape=(100,),dtype='int32')
emmbding=Embedding(input_dim=10000,output_dim=256,input_length=100)(question_input)
encoded_question=LSTM(256)(emmbding)

temp=concatenate([out_a,encoded_question])
out=Dense(1,activation='sigmoid')(temp)
model=Model(inputs=[input,question_input],outputs=out)
model.summary()
plot_model(model,"model.png")



##################################################################3
"""
在做完图片问答模型后，我们可以快速将其转为视频问答的模型。在适当的训练下，你可以为模型提供一个短视频（如100帧）
然后向模型提问一个关于该视频的问题，如“what sport is the boy playing？”->“football”
"""

video_input = Input(shape=(100, 224, 224, 3))
# This is our video encoded via the previously trained vision_model (weights are reused)
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector

# This is a model-level representation of the question encoder, reusing the same weights as before:
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# Let's use it to encode the question:
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# And this is our video question answering model:
merged = concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)
video_qa_model.summary()
