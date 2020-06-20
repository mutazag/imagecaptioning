#%% [markdown]
# # config library to encasulate directory paths 
import os
import pickle

#%% 
class Config:
	__data_dir__ = "data"
	__flickr_images__ = "Flickr8k_Dataset/Flicker8k_Dataset"
	__flickr_text__ = "Flickr8k_text"
	__extracted_features__ = "Extracted_Features"
	__tokenizer_file__ = "tokenizer.pkl"


	def __init__(self, data_dir = "data"): 
		self.data_dir = data_dir

	@property 
	def flickr_images_directory(self): 
		return self.__data_dir__ + "/" + self.__flickr_images__

	@property
	def flickr_images_subfolder(self): 
		return self.__flickr_images__

	@flickr_images_subfolder.setter
	def flickr_images_subfolder(self, flickr_images_subfolder = "Flickr8k_Dataset/Flicker8k_Dataset"): 
		self.__flickr_images__ = flickr_images_subfolder


	@property 
	def TokenizerFilePath(self): 
		return self.ExtractedFeaturesFilePath(self.tokenizer_filename)

	@property 
	def tokenizer_filename(self): 
		return self.__tokenizer_file__
	
	
	@tokenizer_filename.setter
	def tokenizer_filename(self, filename): 
		self.__tokenizer_file__ = filename


	@property 
	def flickr_text_direcotry(self): 
		return self.__data_dir__ + "/" + self.__flickr_text__

	@property
	def flickr_text_subfolder(self): 
		return self.__flickr_text__

	@flickr_text_subfolder.setter
	def flickr_text_subfolder(self, flickr_text_subfolder = "Flickr8k_text"):
		self.__flickr_text__ = flickr_text_subfolder

	def FlickrTextFilePath(self, text_name = "Flickr8k.token.txt"):
		directory = self.flickr_text_direcotry
		if not os.path.exists(directory):
			os.makedirs(directory)
		return directory  + "/" + text_name
	
	@property
	def data_dir(self):
		return self.__data_dir__

	@data_dir.setter
	def data_dir(self, data_dir = "data" ): 
		self.__data_dir__ = data_dir


	@property 
	def extracted_features_directory(self): 
		return self.__data_dir__ + "/" + self.__extracted_features__
	@property
	def extracted_features(self):
		return self.__extracted_features__

	@extracted_features.setter
	def extracted_features(self, extracted_features = "Extracted_Features"):
		self.__extracted_features__ = extracted_features


	def ExtractedFeaturesFilePath(self, pickle_name = "feature.pkl"):
		directory = self.extracted_features_directory
		if not os.path.exists(directory):
			os.makedirs(directory)
		return directory  + "/" + pickle_name


	def SaveFeatures(self, features, pickle_name):
		pickle.dump(features, open(self.ExtractedFeaturesFilePath(pickle_name), 'wb'))


	def LoadFeatures(self, pickle_name):
		with open(self.ExtractedFeaturesFilePath(pickle_name), "rb") as pcklfile: 
			obj = pickle.load( pcklfile)
		return obj



