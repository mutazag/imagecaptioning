import os
import string
import re
import pickle

from keras.preprocessing.text import Tokenizer

# load doc into memory
# generally used to load a text file, no processing is done in this step
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# extract descriptions for images
# for token files, the file contains 5 descriptions for each image 
# this function will create a dictionary, with image id as key, and the value 
# is a list of descriptions for the image
# function returns a the mapping dictionary (image id -> list of descriptions)
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping


# description cleaning process; 
# - convert all words to lowervas
# - remove punctuation 
# - remove words that are 1 character or less 
# - remove workds with numbers in them 
def clean_descriptions(descriptions):
	# prepare regex for char filtering
	re_punc = re.compile('[%s]' % re.escape(string.punctuation))
	for _, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [re_punc.sub('', w) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

# save cleaned up descriptions to file, one per line
# this file is similar in formate to the original description
# tokens file from the flicker text data set. the descriptions 
# in this file are cleaned up to remove punctionations, 1 char words etc 
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()




# convert the loaded descriptions into a vocabulary of words
# using a python set for this task 
# he sets module provides classes for constructing and manipulating unordered collections of 
# unique elements. Common uses include membership testing, removing duplicates from a sequence, 
# and computing standard math operations on sets such as intersection, union, difference, 
# and symmetric difference.
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc


def save_vocabulary(vocab, filename): 
	pickle.dump(vocab, open(filename, 'wb'))

def load_vocabulary(filename): 
	vocab = pickle.load(open(filename, 'rb'))
	return vocab

# load a pre-defined list of photo identifiers
# passing file name for train,dev,test image id data se
# load the data set and return it as a list of image identifiers
# after removing the file extensions 
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
# load the trimmed descriptions, filter it by train/def/test image ids
# and then add startseq and endseq to each of the descriptions 
# Function __load_clean_descriptions()__ defined below loads the cleaned text descriptions from ‘descriptions.txt‘ for a given set of identifiers and returns a dictionary of identifiers to lists of text descriptions
# Using  strings __startseq__ and __endseq__ for first-word and last word signal purpose. These tokens are added to the loaded descriptions as they are loaded. It is important to do this now before we encode the text so that the tokens are also encoded correctly.

def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions


# turn a descriptions set into a list of lines 
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# calculate the length of the description with the most words


def max_length_desc(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# fit a tokenizer given caption descriptions
# this should ceate and then save the tokeniser, ideally i will put them in two seperate 
# functions but for now i will just save the tokenizer at the end of the create_tokenizer 
# function. 
# this code is moved here from the inputprep helper as we realised that the tokenizer is part of 
# the overall data prep, which we wil lneed later 
def create_tokenizer(descriptions, filename = None ):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)


	if filename is not None: 
		pickle.dump(tokenizer, open(filename, "wb"))

	return tokenizer


# get tokenizer from file 
def get_tokenizer(filename): 
	if os.path.isfile(filename): 
		return pickle.load(open(filename, "rb"))
	else:
		return None
	

# load photo features
# load the image feature set for one of the models: vgg, resent50 or others
# then flter it by train/dev/test dataset image ids 
def load_photo_features(filename, dataset):
	# load all features
	all_features = pickle.load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features