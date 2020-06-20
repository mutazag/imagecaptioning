#conda activate py36



import sys
from azureml.core import VERSION


print("python version: " , sys.version)
print("azureml version: ", VERSION)

# enable logging 
# https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-enable-logging 


from azureml.core import Workspace, Experiment, Run

Workspace.create( )

exp = Experiment(workspace=, name='test_experiment')
run = exp.start_logging()
run.log("test-val", 10)