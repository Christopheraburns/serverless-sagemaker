import json
import boto3
import botocore
import logging
import io
import time

# Configure logging
statusCode = ""
msg = ""
logging.basicConfig(filename='lambda.log', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configure the bucket name and Job-ID
output_location = ""
origin = ""  # Set the origin equal to the Algo that is being requested
job_id = str(time.time()).replace(".", "-")
working_bucket = ""

# Master Built-in list (User must use one of these)
master_algo = {"kmeans", "pca", "lda", "factorization-machines", "linear-learner", \
               "ntm", "randomcutforest", "seq2seq", "xgboost", "object-detection", \
               "image-classification", "forecasting-deepar", "blazingtext", "knn"}

max_dataset = 933000000  # 933MB tested


def createS3Bucket(bucket):
    """Create an S3 bucket based on the Job-ID with a training sub-folder
    """
    success = True
    # global output_location
    s3 = boto3.client('s3')
    # output_location = 's3://{}/output'.format(bucket)
    try:
        logger.info("Creating S3 bucket '{}' and sub directories for Training data and Model output.".format(bucket))
        s3CreateResponse = s3.create_bucket(
            ACL='private',
            Bucket=bucket
        )
        if str(s3CreateResponse).__contains__(bucket):  # Bucket created successfully
            logger.info("...s3 working bucket created successfully. Creating working bucket sub-folders...")
    except Exception as err:
        logger.error("Unable to create S3 working bucket.  Exiting.  Err: {}".format(err))
        success = False

    # Create S3 bucket "train" sub-folder
    try:
        prefixOneCreateResponse = s3.put_object(
            Bucket=bucket,
            Body='',
            Key='train' + '/'
        )
        if str(prefixOneCreateResponse).__contains__("'HTTPStatusCode': 200"):
            logger.info("...'train' sub-folder created successfully")
        else:
            raise ValueError("unable to create s3 'train' sub-directory: {}".format(prefixOneCreateResponse))
    except Exception as err:
        logger.error("s3 'train' sub-directory not created.  Exiting. Err: {}".format(err))
        sucess = False

    return success


def transferFile(origin_bucket, origin_key, dest_bucket, dest_key):
    """Transfer the dataset from a source location to a correctly structured.
    S3 bucket
    """
    success = True
    global statusCode
    global msg
    try:
        start = time.time()
        s3 = boto3.resource('s3')
        s3_client = boto3.client('s3')
        poke = s3_client.list_objects_v2(
            Bucket=origin_bucket,
            Prefix=origin_key,
        )
        for obj in poke.get('Contents', []):
            if obj['Key'] == origin_key:
                keySize = obj['Size']

        if keySize > max_dataset:
            success = False
            msg = "ML Factory currently supports only datasets under 1GB in size"
            logger.info(msg)
            statusCode = 413
            return success
        else:  # Transfer the dataset to the working S3 directory
            logger.info("Streaming target dataset to working bucket...")

            s3 = boto3.resource('s3')
            bucket = s3.Bucket(origin_bucket)
            obj = bucket.Object(origin_key)

            # Read into memory (don't write to disk)
            buffer = io.BytesIO(obj.get()['Body'].read())

            # buffer.seek(0)
            s3.Object(dest_bucket, dest_key).upload_fileobj(buffer)

            stop = time.time()
            xferTime = stop - start
            logger.info("dataset transferred in {} seconds".format(xferTime))
            statusCode = 102

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            # The object does not exist.
            msg = "Object: {} does not exist!".format(origin_key)
            statusCode = 404
            logger.error(msg)
            success = False
        else:
            # Something else has gone wrong.
            msg = "Error during dataset transfer: {} ".format(e.message)
            statusCode = 500
            logger.error(msg)
            success = False
        raise

    except Exception as err:
        msg = "Unable to transfer Dataset to S3! {}".format(err)
        statusCode = 500
        logger.error(msg)
        success = False

    return success


def preprareForTraining(algo, target, features):
    print("Prepare dataset for SageMaker training")
    # TODO - probably need a function for each built in to make certain the minimum requirements
    # for that particular built-in are met.
    # make sure the dataset is in shape for training.


def lambda_handler(event, context):
    """
    Process incoming event.
    Incoming JSON sample: {'target_field':'Global_Sales','feature_list':[‘rating’,’score’],
    's3_bucket':'bucket_name','algorithm':'XGBoost','user_email':'email@domain.com'}
    """
    logger.info("Extracting payload-parameters from the event object: {}".format(event))

    statusCode = 200
    global msg
    global origin
    origin_bucket = ""
    origin_key = ""
    proceed = True

    while (proceed):

        # Get the built-in Algorithm to use
        origin = event['algorithm']
        origin = origin.lower()
        if origin is None:
            msg = "Built-in algo to use not supplied in the parameters"
            logger.error(msg)
            statusCode = 400
            break

        if origin not in master_algo:
            msg = "The supplied built-in algo ({}) is not supported.  Supported \
            algos: {} ".format(origin, master_algo)
            logger.error(msg)
            statusCode = 400
            break

        logger.info("Using {} built-in Algo".format(origin))
        logger.info("Setting Job_id = {}".format(job_id))
        working_bucket = origin + "-" + job_id
        logger.info("working bucket set to: {}".format(working_bucket))

        # Get the URL of the Dataset
        # TODO - find a way to make this extensible so origin can be anywhere - not just S3
        dataset = event['s3_bucket']
        dataset = str(dataset)
        if dataset is None:
            msg = "Unable to proceed. Dataset URL not found in the parameters - Fatal Error"
            logger.error(msg)
            statusCode = 400
            break

        # Make sure S3 object path is provided correctly
        if "s3" not in dataset.lower():
            msg = "Unable to proceed. Dataset URL not formatted correctly - \
            requires: s3://bucket-name/[sub-directory]/objectname - Fatal Error"
            logger.error(msg)
            statusCode = 400
            break
        try:
            logger.info("DataSet URL provided: {}".format(dataset))
            pcs = dataset.split('/')
            origin_key = pcs[(len(pcs) - 1)]
            logger.info("ObjectKey: {}".format(origin_key))
            lastChar = dataset.rfind('/')
            origin_bucket = dataset[5:lastChar]
            logger.info("ObjectBucket = {} ".format(origin_bucket))
        except Exception as err:
            msg = "Unable to proceed. Unable to parse Datset Origin URL - \
            required format: s3://bucket-name/[sub-directory]/objectname - \
            Error: {}".format(err)
            statusCode = 400
            break

        # Create the Working Directory
        if (createS3Bucket(working_bucket)):
            # Copy the Dataset to the Working directory
            if (transferFile(origin_bucket, origin_key, working_bucket, \
                             'train/' + origin_key)):
                preprareForTraining(origin, event['target_field'], event['feature_list'])
        else:
            statusCode = 500
            msg = "Unable to transfer dataset"

        proceed = False

    return {
        "statusCode": statusCode,
        "body": json.dumps(msg)
    }



test = json.loads('{"target_field": "Global_Sales",\
  "feature_list": [\
    "rating",\
    "score"\
  ],\
  "s3_bucket": "s3://serverless-sage-lz/Fake_DataSet",\
  "algorithm": "XGBoost",\
  "user_email": "burnsca@amazon.com"}')

lambda_handler(test, None)