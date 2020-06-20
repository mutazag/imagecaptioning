from utils.helpers import Config

c = Config() 


print(c.tokenizer_filename)
print(c.TokenizerFilePath)
c.tokenizer_filename = "test_tokenizer_filename1.pkl"
print(c.tokenizer_filename)
print(c.TokenizerFilePath)


c.SaveFeatures({1:"dic1"}, "dict1.pkl")

vgg_feat = c.LoadFeatures('vgg_features.pkl')

len(vgg_feat)

print(vgg_feat.keys())

keys = vgg_feat.keys()

f1 = vgg_feat["3336211088_4c294a870b"]
f1.shape


f_text = c.flickr_text_direcotry
print(f_text)

f_text_file = c.FlickrTextFilePath("Flickr_8k.testImages.txt")
print(f_text_file)
file = open(f_text_file, "r")
text = file.read()
file.close()
print(text)


from utils.dataprep import load_vocabulary, load_set

vocab = load_vocabulary(c.ExtractedFeaturesFilePath("vocabulary.pkl"))
print("length of extract vocab is: %i " %(len(vocab)))

## train/test/dev 

trainset = load_set(c.FlickrTextFilePath("Flickr_8k.trainImages.txt"))
type(trainset)
len(trainset)
list(trainset)[1:10]

devset = load_set(c.FlickrTextFilePath("Flickr_8k.devImages.txt"))
len(devset)

testset = load_set(c.FlickrTextFilePath("Flickr_8k.testImages.txt"))
len(testset)

print("Train set, size: %i, first item: %s, last item: %s" %(len(trainset), list(trainset)[0], list(trainset)[-1]))
print("Test set, size: %i, first item: %s, last item: %s" %(len(testset), list(testset)[0], list(testset)[-1]))
print("Dev set, size: %i, first item: %s, last item: %s" %(len(devset), list(devset)[0], list(devset)[-1]))