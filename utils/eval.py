from numpy import argmax
from pickle import load


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

# remove start/end sequence tokens from a summary
def cleanup_summary(summary):
	# remove start of sequence token
	index = summary.find('startseq ')
	if index > -1:
		summary = summary[len('startseq '):]
	# remove end of sequence token
	index = summary.find(' endseq')
	if index > -1:
		summary = summary[:index]
	return summary

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for _ in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc_list in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# clean up prediction
		yhat = cleanup_summary(yhat)
		# store actual and predicted
		references = [cleanup_summary(d).split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


def prepare_evaluate_params(c, model,feature_file_name):

	history_file_name = "dense-model_run_history.pkl"
	best_epoch_filename = "dense-model-ep003-loss3.519-val_loss3.735.h5"

	from utils.dataprep import load_set, load_photo_features
	from utils.dataprep import load_clean_descriptions, get_tokenizer, max_length_desc
	from utils.inputprep import create_sequences, data_generator

	feature_file_name = c.ExtractedFeaturesFilePath(feature_file_name)

	model.summary()

	test = load_set(c.FlickrTextFilePath("Flickr_8k.testImages.txt"))
	test_features = load_photo_features(feature_file_name, test)
	test_descriptions = load_clean_descriptions(c.ExtractedFeaturesFilePath('descriptions.txt'), test)

	print("Test photos: %i" % (len(test_features)))
	print("test descriptions: %i" % (len(test_descriptions)))

	# prepare tokensizer
	tokenizer = get_tokenizer(c.TokenizerFilePath) 
	max_length = 34 # check my comment on the model summary cell. 

	vocab_size = len(tokenizer.word_index) + 1

	print( "Tokensizer vocalulary size: %i, Description max length: %i " % (vocab_size, max_length))
	# print("running bleu test ... ")
	# evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

	return tokenizer, test_features, test_descriptions, max_length


def get_demo_captions(model, photos, tokenizer, max_length, demo_list):
	captions = list()
	for key in demo_list:
		# generate description
		caption = generate_desc(model, tokenizer, photos[key], max_length)
		captions.append(caption)
	return(captions)

def demo_captions(c, model, test_features, tokenizer, test_descriptions, max_length):
	curly = '1015118661_980735411b'

	demo_list = ('3497224764_6e17544e0d','3044500219_778f9f2b71','3119076670_64b5340530','1220401002_3f44b1f3f7', '241345844_69e1c22464', '2762301555_48a0d0aa24', '3364861247_d590fa170d', '3406930103_4db7b4dde0', '1343426964_cde3fb54e8', '2984174290_a915748d77', '2913965136_2d00136697', '2862004252_53894bb28b', '3697359692_8a5cdbe4fe')
	demo_path = c.flickr_images_directory + "/"
	demo_captions_list = get_demo_captions(model, test_features, tokenizer, max_length, demo_list)

	import matplotlib.pyplot as plt
	from keras.preprocessing.image import load_img

	for k in range(len(demo_list)):
		img=load_img(demo_path + demo_list[k]+'.jpg')
		imgplot = plt.imshow(img)
		plt.show()
		print(demo_captions_list[k])


def predict_img( model, tokenizer, max_length, img_filepath, cnn_model,cnn_model_features, cnn_model_app, target_size = (224,224)):
	
	from keras.preprocessing.image import load_img, img_to_array

	img = load_img(img_filepath,target_size=target_size )

	
	img_in = img_to_array(img)
	img_in = img_in.reshape((1, target_size[0], target_size[1], 3))
	img_in = cnn_model_app.preprocess_input(img_in)

	# generate classifcation predictions for the image -- informational only 
	img_p = cnn_model.predict(img_in)
	label = cnn_model_app.decode_predictions(img_p)

	prediction = label[0][0]
	print("%s (%.2f%%)" %  (prediction[1], prediction[2]*100))
	print(label)

	# generaetion feature represenation, after poping the last softmax layer 
	# cnn_model.layers.pop()
	# cnn_model_features = Model(inputs=cnn_model_features.inputs, outputs=cnn_model_features.layers[-1].output)
	photo = cnn_model_features.predict(img_in)

	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# intermidiary print
		print("%i>> %s" %(i, in_text))
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break

	
	
	import matplotlib.pyplot as plt
	plt.imshow(img) # rgbs float will clip to 0..1, other wise us integer 
	plt.show() 
	print("CNN model classification: %s (%.2f%%)" %  (prediction[1], prediction[2]*100))
	print("caption: %s" % (in_text))
