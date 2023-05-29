import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.utils import load_img, img_to_array
import cv2
from keras import backend as K

from keras_vggface.utils import preprocess_input #To process the image into the VGG() format
from keras_vggface.vggface import VGGFace
from keras.models import Model



#dimension of images
image_width, image_height=224, 224


base_model=VGGFace(model='resnet50', include_top=False, input_shape=(image_width,image_height,3), pooling='avg')
vgg=Model(inputs=base_model.layers[0].input, outputs=base_model.layers[-2].output)


def image_to_encoding(image_path):
    # encode the image

    #load the image
    img= load_img(image_path, target_size=(224,224))

    #convert the image to an array
    img= img_to_array(img)
    img=np.expand_dims(img, axis=0)

    #encode the image  using  vgg 
    img=preprocess_input(img, version=2)
    img_encode=vgg(img)
     
    ## reshape the imahe into a 2D dimension of (None, 2048)
    x_train=K.eval(img_encode)
    x_train=x_train.reshape(1,-1)
    return x_train

