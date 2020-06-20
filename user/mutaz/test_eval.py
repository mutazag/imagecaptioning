feature_file_name = "densenet_features.pkl"
history_file_name = "dense-model_run_history.pkl"
best_epoch_filename = "dense-model-ep003-loss3.519-val_loss3.735.h5"

import pickle

from utils.plothist import plot
from utils.helpers import Config 

from utils.eval import prepare_evaluate_params, demo_captions, evaluate_model, predict_img


from keras.models import Model, load_model

c = Config() 


model = load_model(best_epoch_filename)

# load model and run bleu test, it also returns the tokenizer and max_length to be used later
tokenizer, test_features, test_descriptions, max_length = prepare_evaluate_params(c, model, feature_file_name)

# prepare the photo input 
demo_path = c.flickr_images_directory + "/"
img_filepath = demo_path + "1015118661_980735411b.jpg"
#img_filepath = "user/mutaz/car1.jpg"


import keras.applications.densenet as densenet121

# load a cnn model to examine the classification
dense_model = densenet121.DenseNet121()
# load a cnn feature representation model, this one is used for producing input to the captioning model 

dense_model_features = densenet121.DenseNet121()
dense_model_features.layers.pop()
dense_model_features = Model(inputs=dense_model_features.inputs, outputs=dense_model_features.layers[-1].output)


predict_img(model, tokenizer, max_length, img_filepath, dense_model, dense_model_features, densenet121, target_size = (224,224))


predict_img(model, tokenizer, max_length, "user/mutaz/car1.jpg", dense_model, dense_model_features, densenet121, target_size = (224,224))



print("end")
