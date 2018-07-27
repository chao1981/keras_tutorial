import keras.backend as K
from data import *
import cv2
import argparse
from keras.applications import vgg16


def get_output_layer(model, layer_name):
  # get the symbolic outputs of each "key" layer (we gave them unique names).
  layer_dict = dict([(layer.name, layer) for layer in model.layers])
  layer = layer_dict[layer_name]
  return layer

def visualize_class_activation_map( img_path, output_path):
        model = vgg16.VGG16(weights='imagenet')
        original_img = cv2.imread(img_path, 1)
        original_img= cv2.resize(original_img,(224,224))
        import matplotlib.pyplot as plt
        plt.imshow(original_img)
        plt.show()
        print("original_img shape:",original_img.shape)
        width, height, _ = original_img.shape

        #Reshape to the network input shape (3, w, h).
        # img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])
        img = np.array([original_img])
        print("IMG  shape:",img.shape)
        #Get the 512 input weights to the softmax.
        class_weights = model.layers[-1].get_weights()[0]
        final_conv_layer = get_output_layer(model, "block5_conv3")
        get_output = K.function([model.layers[0].input], [final_conv_layer.output])
        [conv_outputs] = get_output([img])
        conv_outputs = conv_outputs[0, :, :, :]
        print(conv_outputs.shape)
        print(class_weights.shape)
        #Create the class activation map.
        cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
        for i, w in enumerate(class_weights[:512,1]):
                cam += w * conv_outputs[ :, :,i]
        # print("predictions", predictions)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap*0.5 + original_img
        cv2.imwrite(output_path, img)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type = bool, default = False, help = 'Train the network or visualize a CAM')
    parser.add_argument("--image_path", type = str, help = "Path of an image to run the network on")
    parser.add_argument("--output_path", type = str, default = "heatmap.jpg", help = "Path of an image to run the network on")
    parser.add_argument("--model_path", type = str, help = "Path of the trained model")
    parser.add_argument("--dataset_path", type = str, help = \
        'Path to image dataset. Should have pos/neg folders, like in the inria person dataset. \
        http://pascal.inrialpes.fr/data/human/')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  args = get_args()
  visualize_class_activation_map(args.image_path, args.output_path)
 
