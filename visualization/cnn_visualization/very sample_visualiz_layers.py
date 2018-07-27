from keras.layers import Convolution2D, MaxPooling2D, Activation
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import cv2  # only used for loading the imag

cat = cv2.imread('cat.png')
plt.imshow(cat)
# plt.show()


model = Sequential()
model.add(Convolution2D(1,    # number of filter layers
                        7,    # y dimension of kernel (we're going for a 3x3 kernel)
                        7,    # x dimension of kernel
                        input_shape=cat.shape))


# here we get rid of that added dimension and plot the image
def visualize_cat(model, cat):
    # Keras expects batches of images, so we have to add a dimension to trick it into being nice
    cat_batch = np.expand_dims(cat,axis=0)## cat shape:(320, 400, 1)-> cat_batch shape:(1,320,400,1)
    conv_cat = model.predict(cat_batch)
    conv_cat = np.squeeze(conv_cat, axis=0)## conv_cat shape:(1,318, 398, 1)-> conv_cat shape:(318, 398, 1)
    conv_cat2 = conv_cat.reshape(conv_cat.shape[:2])  ## conv_cat2 shape:(318, 398, 1)-> conv_cat2 shape:(318, 398)
    plt.imshow(conv_cat2)


visualize_cat(model, cat)
plt.show()
