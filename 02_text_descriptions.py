#%% [markdown]

# Prepare Text Descriptions 

# Data set includes text desprtions for all images in the Flickr8k_text folder
# - Flickr8k.token.txt: reference descriptions for images, 5 descriptions for each image 

#   1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
#   1000268201_693b08cb0e.jpg#1	A girl going into a wooden building .
#   1000268201_693b08cb0e.jpg#2	A little girl climbing into a wooden playhouse .
#   1000268201_693b08cb0e.jpg#3	A little girl climbing the stairs to her playhouse .
#   1000268201_693b08cb0e.jpg#4	A little girl in a pink dress going into a wooden cabin .

# list of images for data split: 
# - train: Flickr_8k.trainImages.txt
# - dev: Flickr_8k.devImages.txt
# - test:Flickr_8k.testImages.txt


# also prepare a tokenizer and save it to file 

#%%


# import libraries 
import numpy as np 
from utils.helpers import Config
from utils.dataprep import load_doc, load_descriptions, clean_descriptions, save_descriptions
from utils.dataprep import to_vocabulary, save_vocabulary, load_vocabulary
from utils.dataprep import load_set,load_clean_descriptions, create_tokenizer, get_tokenizer

# setting seed
np.random.seed(42) 
#%%

c = Config() 

filename = c.FlickrTextFilePath("Flickr8k.token.txt")

#%% 
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded descriptions: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# save to file to the same folder as image extracted features since this 
# is a processed file to be used in later stages
save_descriptions(descriptions, c.ExtractedFeaturesFilePath('descriptions.txt')) 
#%% 
# summarize vocabulary -- not saved yet
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
save_vocabulary(vocabulary, c.ExtractedFeaturesFilePath("vocabulary.pkl"))
print("Vocabulary created and saved to %s" %(c.ExtractedFeaturesFilePath("vocabulary.pkl")))

#%%# create and save tokenizer 
train = load_set(c.FlickrTextFilePath("Flickr_8k.trainImages.txt"))
train_descriptions = load_clean_descriptions(c.ExtractedFeaturesFilePath('descriptions.txt'), train)
tokenizer = create_tokenizer(train_descriptions, c.TokenizerFilePath) 

#sample use of the tokenizer: 

seq = tokenizer.texts_to_sequences(["The quick brown fox jumps over the lazy dog"])
print("Tokenizer text_to_seq output sample: %s" %(seq))
print("Tokenizer created and saved to %s" % (c.TokenizerFilePath))
#%%

tt2 = get_tokenizer(c.TokenizerFilePath)
print(tt2.texts_to_sequences(["The quick brown fox jumps over the lazy dog"])) 
print(len(tokenizer.word_index) + 1)
