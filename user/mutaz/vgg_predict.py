#use VGG out of the box to predict classes of an image 
import os
import glob

import numpy as np
import matplotlib.pyplot as plt 

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

model = VGG16() 
os.listdir("user/mutaz/")
glob.glob("user/mutaz/*.jpg")
img = load_img("user/mutaz/car1.jpg",target_size=(224,224) )
img = img_to_array(img)


plt.imshow(img.astype(int)) # rgbs float will clip to 0..1, other wise us integer 
plt.show() 
plt.hist(img.flatten(), bins=256)
plt.show()

img = img.reshape((1,224,224,3))
img.shape
img = preprocess_input(img)
plt.hist(img.flatten(), bins=256)
plt.show()



img_p = model.predict(img)

label = decode_predictions(img_p)

prediction = label[0][0]
print("%s (%.2f%%)" %  (prediction[1], prediction[2]*100))


imgs_in = []
imgs = [img_to_array(load_img(imgpath,target_size=(224,224) )) for imgpath in glob.glob("user/mutaz/*.jpg")]

# make it a array
imgs_in = np.array(imgs)
print(imgs_in.shape)


imgs_in = [preprocess_input(img) for img in imgs_in]

plt.imshow(imgs_in[0].astype(int))
plt.show()

img_predictions = model.predict(np.array(imgs_in))

labels = decode_predictions(img_predictions)

# https://matplotlib.org/3.1.0/tutorials/introductory/images.html#sphx-glr-tutorials-introductory-images-py

figs = {}
axs = {}
for i,img in enumerate(imgs):
    print(i)

    figs[i] = plt.figure()
    axs[i]=figs[i].add_subplot(111)
    axs[i].imshow(img.astype(int))
    label = labels[i][0]
    axs[i].set_title("%s (%.2f%%)" %  (label[1], label[2]*100))
    axs[i].get_yaxis().set_visible(False)
    axs[i].get_xaxis().set_visible(False)
    axs[i].imshow(img.astype(int))
    

plt.show()