import numpy as np
import os
from PIL import Image
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from tensorflow.keras.utils import to_categorical


model = InceptionV3(weights='imagenet')
image_encoder_model = Model(model.input, model.layers[-2].output)

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def image_encoder(image_path):
    image = preprocess(image_path) 
    fea_vec = image_encoder_model.predict(image, verbose=False) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec