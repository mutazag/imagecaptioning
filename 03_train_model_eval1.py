
import pickle

from utils.plothist import plot
from utils.helpers import Config 

c = Config() 

history = pickle.load(open(c.ExtractedFeaturesFilePath("model_run_history.pkl"), "rb"))

plot(history)

print("end")