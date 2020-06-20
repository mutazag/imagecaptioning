#%% [markdown]

# model training process: 
#
#   1. load train image feature set and dev image feature set, this is a two step process: 
#      
#  a. data already split to train,dev, test: load train and dev image id sets
#  
#   b. load image features for train and dev sets
#   2. load clean descriptions and prepare add seqstart and seqend to training and dev sequences 
#   
#   3. tokenize train and test sets
#   4. create the model input sequences, includes a photos input and a tokenized text sequence
#   5. define the model 
#   6. fit the model and assess loss

#%% # imports 

from utils.helpers import Config
from utils.dataprep import load_set, load_photo_features
from utils.dataprep import load_clean_descriptions, get_tokenizer, max_length_desc
from utils.inputprep import create_sequences, data_generator

c = Config()
#%% 
# 1. load train and dev images features 
train = load_set(c.FlickrTextFilePath("Flickr_8k.trainImages.txt"))
dev = load_set(c.FlickrTextFilePath("Flickr_8k.devImages.txt"))

# use VGG trained features 
train_features = load_photo_features(c.ExtractedFeaturesFilePath("vgg_features.pkl"), train)
dev_features = load_photo_features(c.ExtractedFeaturesFilePath("vgg_features.pkl"), dev)

print("Train ids: %i, and dev ids: %i" % (len(train), len(dev)))
print("Train photos: %i, and dev photos: %i" % (len(train_features), len(dev_features)))
#%%
# 2. load clean descriptions for data sets. and load vocabulary 

train_descriptions = load_clean_descriptions(c.ExtractedFeaturesFilePath('descriptions.txt'), train)
dev_descriptions = load_clean_descriptions(c.ExtractedFeaturesFilePath('descriptions.txt'), dev)

print("Train descriptions: %i, and dev descriptions: %i" % (len(train_descriptions), len(dev_descriptions)))

#%% 
# 3. tokensize train and dev sets 

# prepare tokensizer
tokenizer = get_tokenizer(c.TokenizerFilePath) 
max_length = max_length_desc(train_descriptions)

vocab_size = len(tokenizer.word_index) + 1

print( "Tokensizer vocalulary size: %i, Description max length: %i " % (vocab_size, max_length))
# TODO: here we should save the tokenizer for later use, it will be needed when traslating yhat vector to a description 

#%% [markdown]
# 4. create input and validation set sequences
#   sequences will contain the following components: 
#   - photo features mapped to image ids 
#   - input text sequences mapped to image ids 
#   - output text sequences
# prepare train and test sequences 
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)
X1test, X2test, ytest = create_sequences(tokenizer, max_length, dev_descriptions, dev_features)
# TODO: i am not going to need these when we switch to data generator 


#%% [markdown]

# # 4. define and fit model 

#%% 
from keras.utils import to_categorical, plot_model
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

#%%
# define the model
# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	model.summary()
	# plot_model(model, to_file='model.png', show_shapes=True)
	return model

#%% 
model = define_model(vocab_size, max_length)

#%%
# plot_model(model, to_file=c.ExtractedFeaturesFilePath('model.png'), show_shapes=True)


#%% 
train_data_generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
dev_data_generator = data_generator(dev_descriptions, dev_features, tokenizer, max_length)
# inputs, outputs = next(generator)
# print(inputs[0].shape)
# print(inputs[1].shape)
# print(outputs.shape)
#%%
# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# %%
# # "fit model
# history = model.fit(
#     [X1train, X2train], 
#     ytrain, 
#     epochs=20, 
#     verbose=2, 
#     callbacks=[checkpoint], 
#     validation_data=([X1test, X2test], ytest))

#%% [markdown]
# https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/ 
# Why do we need steps_per_epoch ?

# Keep in mind that a Keras data generator is meant to loop infinitely — it should never return or exit.

# Since the function is intended to loop infinitely, Keras has no ability to determine when one epoch starts 
# and a new epoch begins.

# Therefore, we compute the steps_per_epoch  value as the total number of training data points divided by the batch size. 
# Once Keras hits this step count it knows that it’s a new epoch.


# history = model.fit_generator(
# 	train_data_generator, 
# 	epochs=3, 
# 	steps_per_epoch=len(train_descriptions),
# 	verbose=2, # 1: progress, 2: one line per epoch 
# 	validation_data= dev_data_generator, 
# 	validation_steps=len(dev_descriptions), 
# 	callbacks=[checkpoint])

#%%
#save history
import pickle

with open(c.ExtractedFeaturesFilePath("model_run_history.pkl"), "wb") as pcklfile:
	pickle.dump(history, pcklfile)



#%%
