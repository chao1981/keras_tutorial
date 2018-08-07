import keras.backend as K
from data import *
import cv2
import argparse
from keras.applications import vgg16

preprocess_input=vgg16.preprocess_input

def get_output_layer(model, layer_name):
  # get the symbolic outputs of each "key" layer (we gave them unique names).
  layer_dict = dict([(layer.name, layer) for layer in model.layers])
  layer = layer_dict[layer_name]
  return layer


def visualize_class_activation_map(img_path, output_path):
  model = vgg16.VGG16(weights='imagenet')
  from keras.preprocessing import image
  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)   # of size (1, 224, 224, 3)
  x = preprocess_input(x)
  preds = model.predict(np.random.rand(1,224,224,3))

  african_elephant_output = model.output[:, np.argmax(preds[0])]
 
  last_conv_layer = model.get_layer('block5_conv3')
  grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
  
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
 
  
  iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
  pooled_grads_value, conv_layer_output_value = iterate([x])
  print(pooled_grads_value.shape)
  print(conv_layer_output_value.shape)
  for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

  heatmap = np.mean(conv_layer_output_value, axis=-1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  import matplotlib.pyplot as plt
  plt.imshow(heatmap)
  plt.show()

  img = cv2.imread(img_path)
  heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  superimposed_img = heatmap * 0.4 + img
  cv2.imwrite(output_path, superimposed_img)


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", type=bool, default=False, help='Train the network or visualize a CAM')
  parser.add_argument("--image_path", type=str, help="Path of an image to run the network on")
  parser.add_argument("--output_path", type=str, default="heatmap.jpg", help="Path of an image to run the network on")
  parser.add_argument("--model_path", type=str, help="Path of the trained model")
  parser.add_argument("--dataset_path", type=str, help= \
    'Path to image dataset. Should have pos/neg folders, like in the inria person dataset. \
    http://pascal.inrialpes.fr/data/human/')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = get_args()
  visualize_class_activation_map("test.jpg", "out.png")

