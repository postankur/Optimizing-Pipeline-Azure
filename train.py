from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Dataset
import clean_data
import joblib
from sklearn.linear_model import Ridge

from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

data_url_path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
ds = TabularDatasetFactory.from_delimited_files(path=data_url_path)
df = ds.to_pandas_dataframe()

# ds = ### YOUR CODE HERE ###
x, y = clean_data.clean_data(ds)

# TODO: Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=50)
    
### YOUR CODE HERE ###a

run = Run.get_context()
 
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('./outputs', exist_ok=True)  
     
    model_file_name = 'amptrain_1.pkl'
    # save model in the outputs folder so it automatically get uploaded
    with open(model_file_name, "wb") as file:
        joblib.dump(value=model, filename=os.path.join('./outputs/',
                                                        model_file_name))
                                                     
    # joblib.dump(model, 'model.joblib')
    # model1 = run.register_model(model_name='sklearn1', 
    #                       model_path='/model.joblib',
    #                       model_framework=Model.Framework.SCIKITLEARN,
    #                       model_framework_version='0.19.1',
    #                       resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5))

if __name__ == '__main__':
    main()