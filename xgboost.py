import logging
import pandas as pd
import json

# Amazon SageMaker's XGBoost container expects data in the libSVM or CSV data format.

def verify(event):
    """
    verify that event passed to the Lambda function has the correct values for the XGBoost algo
    :param event:
    :return: boolean
    """
    verified = True




    return verified



def prepareDataSet(bucket):
    """
    Prepare the dataset to:
    1.) pass only the features in the feature list
    2.) make sure the Target variable is the first column
    3.) make sure there are no headers

    :param bucket:
    :return:
    """


    # Check for missing values

    # One Hot encoding

    # Leave oddly distributed data alone

