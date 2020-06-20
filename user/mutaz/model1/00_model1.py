from os import listdir
from os import path
import glob
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

def load_photos(directory, n = 10000):
    images = dict()
    i = 0 

    files = glob.glob(directory + "/*.jpg", recursive= False)

    return images

# load images
directory = "data/Flickr8k_Dataset"#/Flicker8k_Dataset"
images = load_photos(directory,2)
print('Loaded Images: %d' % len(images))
