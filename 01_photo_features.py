
#%% [markdown]
# import libraries 

import os 
import datetime
from pickle import dump

#%% 
# import keras libraries for different pretrained models 
import keras.applications.vgg16 as vgg16
import keras.applications.inception_v3 as inceptionv3
import keras.applications.resnet50 as resnet50
import keras.applications.densenet as densenet121


#%% 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from keras.models import Model

import numpy as np

from utils.helpers import Config

# also setting seed
np.random.seed(42) 


#%%
# extract features from each photo in the directory
def extract_features(directory, model, processor, size = (224,224)):
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
	model.summary()
	# extract features from each photo
	features = dict()
	i = 0
	for name in os.listdir(directory):
		i += 1
		# load an image from file
		filename = os.path.join(directory, name)
		image = load_img(filename, target_size=size)
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the selected model, processor is the pre_process image for the selected model
		#preprocess_input
		image = processor(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		if i % 500 == 0: 
			print("%s %i >> %s" % (datetime.datetime.now(), i, image_id))
	return features


#%%

c = Config()
directory = c.flickr_images_directory 


#%% [markdown]
# ## Extract features using VGG16 model
#%% 
print("%s VGG feature extraction - start" % (datetime.datetime.now()))
model = vgg16.VGG16()
features = extract_features(directory, model, vgg16.preprocess_input)
c.SaveFeatures(features, 'vgg_features.pkl') 

print('Extracted Features: %d' % len(features))
print("%s Feature extraction - end" % (datetime.datetime.now()))

#%% [markdown]
# ## Extract features using Inception model
#%%
print("%s InsceptionV3 feature extraction - start" % (datetime.datetime.now()))

model = inceptionv3.InceptionV3() 
features = extract_features(
	directory, 
	model, 
	processor = inceptionv3.preprocess_input, 
	size = (299,299)) # Note Inception takes an input of 299 x 299
c.SaveFeatures(features, 'inception_features.pkl')

print('Extracted Features: %d' % len(features))
print("%s Feature extraction - end" % (datetime.datetime.now()))


#%% [markdown]
# ## Extract features using ResNet50 model
#%%
print("%s ResNet50 feature extraction - start" % (datetime.datetime.now()))
model = resnet50.ResNet50() 
features = extract_features(
	directory, 
	model, 
	processor = resnet50.preprocess_input)

c.SaveFeatures(features, "resnet_features.pkl")

print('Extracted Features: %d' % len(features))
print("%s Feature extraction - end" % (datetime.datetime.now()))

# Save to file

#%% [markdown]
# ## Extract features using DenseNet121 model
#%%
print("%s DenseNet121 feature extraction - start" % (datetime.datetime.now())) 

model = densenet121.DenseNet121()
features = extract_features(
	directory, 
	model, 
	processor = densenet121.preprocess_input)

c.SaveFeatures(features, "densenet_features.pkl")


print('Extracted Features: %d' % len(features))
print("%s Feature extraction - end" % (datetime.datetime.now()))
