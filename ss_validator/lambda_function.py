import json
import boto3
import smbuiltin
import logging
import time

# A few global variables
statusCode = ""                 # HTTP statuscode to return to APIGateway
msg = ""                        # Message to include with HTTP StatusCode in response to caller
origin = ""                     # The Algorithm that is being requested for training
prepped_max_dataset = 400000000 # Dataset must be 400Mb or under for Lambda to prep
asis_max_dataset = 933000000    # 933MB tested for maximum passthrough dataset size
asis = False                    # Flag to indicate if dataset will be prepped or sent directly to Sagemaker "AS-IS"
ttl = 299                       # Lifespan of an instance of this Lambda Function (in seconds)
currentAge = 0                  # Track current age of this Lambda function (in seconds)
data = None                     # Pandas dataframe form of our original dataset
timemgmt = []                   # keep a list of the time taken by each method

# Configure logging
logging.basicConfig(filename='lambda.log', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configure the bucket name and Job-ID
output_location = ""
job_id = ""
working_bucket = ""


# Master Built-in list (incoming API calls must use one of these Algorithms or return an error)
master_algo = {"kmeans", "pca", "lda", "factorization-machines", "linear-learner", \
               "ntm", "randomcutforest", "seq2seq", "xgboost", "object-detection", \
               "image-classification", "forecasting-deepar", "blazingtext", "knn"}


def writerecord(event):
    global statusCode
    global msg
    global job_id
    try:
        dyn = boto3.resource('dynamodb')
        table = dyn.Table('tableau_int')
        payload = {
            'job_id': job_id,
            'payload': event
        }
        
        response = table.put_item(Item=payload)
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            statusCode = 200
            msg = "Request added to processing queue with job_id: " + job_id
            logger.info(msg)
        else: #No error triggered but write did not succeed - check DynamoDB logs!
            statusCode = 500
            msg = "Queuing error"
            logger.error("#No error triggered but write did not succeed - \
            check DynamoDB logs!")
            
    except Exception as err:
        statusCode = 500
        msg = "Unable to add this processing request to the queue: {}".format(err)
        logger.error(msg)


def validateAPI(event):
    """
    Validate that the incoming API call contains the necessary information.
    """

    # Probably too many global variables - better Python devs should be able to improve on this.
    global statusCode
    global msg
    global origin
    global origin_bucket
    global origin_key
    global working_bucket
    global asis

    statusCode = 200
    target_field = ""
    feature_list = {}

    try:
        logger.info("Extracting payload-parameters from the event object: {}".format(event))
    
        # Get the built-in Algorithm to use
        if 'algorithm' not in event:
            msg = "Built-in algo to use not supplied in the parameters"
            logger.error(msg)
            statusCode = 400
            return False
        else:
            origin = event['algorithm']
            origin = origin.lower()
    
        if origin not in master_algo:
            msg = "The requested algo ({}) is not supported.  Supported algos: {} ".format(origin, master_algo)
            logger.error(msg)
            statusCode = 400
            return False
    
        # Get the HyperParams for this Algo
        if origin == 'xgboost':
            xgboost = smbuiltin.XGBoost(event)
            xgboost.verify()
            result = xgboost.verified
            xmsg = xgboost.msg
            if result is False:
                msg = "Required parameter values for running XGboost not detected or incorrect: {}".format(xmsg)
                logger.error(msg)
                statusCode = 400
                return False
    
        logger.info("Using {} built-in Algo".format(origin))
        logger.info("Setting Job_id = {}".format(job_id))
        working_bucket = origin + "-" + job_id
        logger.info("working bucket set to: {}".format(working_bucket))
    
        # Do we process the DataSet as-is or do we need to prepare/validate it
        if 'asisdata' not in event: # Default to AS-IS, even if not parameter is no provided
            logger.info("Processing dataset 'AS IS'")
            logger.info("'AS-IS processing not specified - will default to NOT preparing the dataset for Sagemaker")
            asis = True
        else:
            asis = event['asisdata']
    
        if asis is True:
            logger.info("Processing dataset 'AS IS'")
            logger.info("'AS-IS' processing set to TRUE - will NOT prepare the dataset for Sagemaker")
        else:
            logger.info("'AS-IS' set to FALSE - Will attempt to prepare dataset for Sagemaker")
    
            #  We need Target Field & Feature_List params to prepare the dataset - just check if they are in the API call
            #  we will verify they actually exist in the dataset later
    
            if 'target_field' not in event:
                msg = "Unable to prepare this dataset to train.  'Target_Field' has not been supplied"
                logger.error(msg)
                statusCode = 400
                return False
            else:
                target_field = event['target_field']
                logger.info("Dataset Target Field = {}".format(target_field))
    
            if 'feature_list' not in event: # Default to use all fields
                feature_list = json.loads('{"allfields": "data"}')
            else:
                feature_list = event['feature_list']
    
            logger.info("Using the following features of the data set:")
            for feature in feature_list:
                logger.info("\t {}".format(feature))
    
        if 's3_bucket' not in event:
            msg = "Unable to proceed. Dataset URL not found in the parameters - Fatal Error"
            logger.error(msg)
            statusCode = 400
            return False
        else:
            dataset = event['s3_bucket']
            dataset = str(dataset)
    
        # Make sure S3 object path is provided correctly
        if "s3" not in dataset.lower():
            msg = "Unable to proceed. Dataset URL not formatted correctly - \
                requires: s3://bucket-name/[sub-directory]/objectname - Fatal Error"
            logger.error(msg)
            statusCode = 400
            return False

        logger.info("DataSet URL provided: {}".format(dataset))
        pcs = dataset.split('/')
        origin_key = pcs[(len(pcs) - 1)]
        logger.info("ObjectKey: {}".format(origin_key))
        lastChar = dataset.rfind('/')
        origin_bucket = dataset[5:lastChar]
        logger.info("ObjectBucket = {} ".format(origin_bucket))
    except Exception as err:
        msg = "Unable to proceed. Error: {}".format(err)
        logger.error(msg)
        statusCode = 400
        return False

    StatusCode = 200
    msg = "Request passed validation checks. Processing data now"
    return True
    
    
    
def lambda_handler(event, context):
    global statusCode
    global msg
    global job_id
    
    job_id = str(time.time()).replace(".", "-")
    
    if validateAPI(event):
        writerecord(event)
        
    return {
        "statusCode": statusCode,
        "body": json.dumps(msg)
    }
  
       
