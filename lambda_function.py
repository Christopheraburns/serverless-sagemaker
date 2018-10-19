"""
10/4/18 burnsca@amazon.com
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import json
import boto3
import botocore
import logging
import io
import time
import smbuiltin
import pandas as pd
import numpy as np
import sagemaker
import zipfile
import os

# A few global variables
statusCode = ""                         # HTTP statuscode to return to APIGateway
msg = ""                                # Message to include with HTTP StatusCode in response to caller
origin = ""                             # The Algorithm that is being requested for training
prepped_max_dataset = 400000000         # Dataset must be 400Mb or under for Lambda to prep
asis_max_dataset = 933000000            # 933MB tested for maximum passthrough dataset size
asis = False                            # Flag indicating if dataset will be prepped or sent directly to Sagemaker "AS-IS"
ttl = 299                               # Lifespan of an instance of this Lambda Function (in seconds)
currentAge = 0                          # Track current age of this Lambda function (in seconds)
data = None                             # Pandas dataframe form of our original dataset
timemgmt = []                           # keep a list of the time taken by each method
target_field = ""                       # Column to run predictions on
feature_list = {}                       # Columns to keep for training
train_instance_type = 'ml.m4.xlarge'    # TODO - add API param to configure instance type AND instance count
hyperparams = {}                        # list of hyperparams to pass on to training algo
topicArn = ""                           # ARN of SNS Topic for notifications
model_path_prefix = ""                  # S3 prefix where the trained model is stored
lambda_role = 'lambda_sagemaker'
function_arn = ''
train_job_name = ''

# Configure logging
#logging.basicConfig(filename='lambda.log', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configure the bucket name and Job-ID
output_location = ""
job_id = str(time.time()).replace(".", "-")
working_bucket = ""

# Master Built-in list (incoming API calls must use one of these Algorithms or return an error)
master_algo = {"kmeans", "pca", "lda", "factorization-machines", "linear-learner", \
               "ntm", "randomcutforest", "seq2seq", "xgboost", "object-detection", \
               "image-classification", "forecasting-deepar", "blazingtext", "knn"}


def sendSNSMsg(msg, subject):

    global topicArn

    try:
        sns = boto3.client('sns')

        rPublish = sns.publish(
            TopicArn=topicArn,
            Message=msg,
            Subject=subject
        )
    except Exception as err:
        logger.error("sendSNSMsg::{}".format(err))


def sagemakerTrain(event):

    global job_id
    global working_bucket
    global origin
    global hyperparams
    global model_path_prefix
    global train_job_name

    try:
        s3_train_data = sagemaker.s3_input('s3://{}/train'.format(working_bucket), content_type='csv')
        s3_valid_data = sagemaker.s3_input('s3://{}/validation'.format(working_bucket), content_type='csv')

        logger.info("Initiating Sagemaker training with data from {}".format(s3_train_data))

        session = sagemaker.Session()

        # Set up the training for this Algo
        if origin == 'xgboost':
            xgboost = smbuiltin.XGBoost(event)
            container = xgboost.getcontainer(boto3.Session().region_name)
            trainer = xgboost.buildtrainer(container, working_bucket, session)
            # set up the hyperparameters
            trainer = xgboost.sethyperparameters(trainer, hyperparams)

            trainer.fit({'train': s3_train_data, 'validation': s3_valid_data}, wait=False)
            train_job_name = trainer.latest_training_job.name
            model_path_prefix = "/" + train_job_name + "/output"
        else:
            logger.error("sagemakerTrain::Serverlesss Sagemaker process does not support the {} algorithm.".format(origin))

    except Exception as err:
        logger.error("sagemakerTrain::Error while launching SageMaker training: {}".format(err))


def createLambdaTrigger():
    global function_arn
    global working_bucket
    logger.info("Lambda Function created, now settting permissions for s3 trigger...")
    try:
        s3 = boto3.resource('s3')

        # First set up permissions
        client = boto3.client('lambda')
        response = client.add_permission(
            FunctionName=function_arn,
            StatementId='1',
            Action='lambda:InvokeFunction',
            Principal='s3.amazonaws.com',
            SourceArn='arn:aws:s3:::' + str(working_bucket),
            SourceAccount='056149205531'
        )

        logger.info(response)

        bucket_notification = s3.BucketNotification(working_bucket)

        response = bucket_notification.put(
            NotificationConfiguration={'LambdaFunctionConfigurations': [
                {
                    'LambdaFunctionArn': function_arn,
                    'Events':[
                        's3:ObjectCreated:*'
                    ],
                },
            ]})
        logger.info(response)
    except Exception as err:
        logger.error("createLambdaTrigger::Unable to create S3 trigger: {}".format(err))


def lambdaFunctionGenerator(origin):
    global train_job_name
    global working_bucket
    global job_id

    try:
        # Import statements
        code = "import os\n"
        code += "import io\n"
        code += "import boto3\n"
        code += "import sagemaker\n"
        code += "import logging\n"
        code += "from sagemaker import get_execution_role\n"
        code += "from sagemaker.amazon.amazon_estimator import get_image_uri\n"
        code += "\n"

        # logging setup
        code += "logger = logging.getLogger()\n"
        code += "logger.setLevel(logging.INFO)\n"
        code += "\n"

        # S3 setup
        code += "bucket = '" + working_bucket + "'\n"
        code += "output_location = 's3://{}/output'.format(bucket)\n"
        code += "job_id = '" + job_id + "'\n"
        code += "\n"
        code += "role = get_execution_role()\n"
        code += "\n"

        # Handler function
        code += "def lambda_handler(event, context):\n"
        code += "\ttry:\n"
        code += "\t\tsm = boto3.client('sagemaker')\n"
        code += "\n"
        code += "\t\tcontainer = get_image_uri(boto3.Session().region_name, 'xgboost')\n"
        # prepare a model context for hosting
        code += "\t\thosting_container = {\n"
        code += "\t\t\t'Image': container,\n"
        code += "\t\t\t'ModelDataUrl': 's3://" + working_bucket + "/" + train_job_name + "/output/model.tar.gz'\n"
        code += "\t\t}\n"
        code += "\t\tlogger.info('creating model object...')\n"
        code += "\t\tcreate_model_response = sm.create_model(\n"
        code += "\t\t\tModelName='" + job_id + "',\n"
        code += "\t\t\tExecutionRoleArn=role,\n"
        code += "\t\t\tPrimaryContainer=hosting_container)\n"
        code += "\n"
        code += "\t\tlogger.info('create model response: {}'.format(create_model_response))\n"

        # Create an endpoint configuration
        code += "\t\tlogger.info('creating endpoint config...')\n"
        code += "\t\tendpoint_config='" + job_id + "-endpoint-config'\n\n"
        code += "\t\tcreate_endpoint_config_response = sm.create_endpoint_config(\n"
        code += "\t\t\tEndpointConfigName=endpoint_config,\n"
        code += "\t\t\tProductionVariants=[{\n"
        code += "\t\t\t\t'InstanceType': 'ml.m4.xlarge',\n"  # TODO find a way to make these values configurable
        code += "\t\t\t\t'InitialInstanceCount':1,\n"
        code += "\t\t\t\t'ModelName':'" + job_id + "',\n"
        code += "\t\t\t\t'VariantName':'AllTraffic'}]\n"
        code += "\t\t)\n"
        code += "\t\tlogger.info('create-endpoint-config response: {}'.format(create_endpoint_config_response))\n"
        code += "\n"

        # Deploy the endpoint - but don't wait around for it
        code += "\t\tendpoint = '" + job_id + "-endpoint'\n"
        code += "\n"
        code += "\t\tlogger.info('creating endpoint...')\n"
        code += "\t\tcreate_endpoint_response = sm.create_endpoint(\n"
        code += "\t\t\tEndpointName=endpoint,\n"
        code += "\t\t\tEndpointConfigName=endpoint_config\n"
        code += "\t\t)\n"
        code += "\n"
        code += "\t\tresponse = sm.describe_endpoint(EndpointName=endpoint)\n"
        code += "\t\tstatus = response['EndpointStatus']\n"
        code += "\t\tlogger.info('EndpointStatus = {}'.format(status))\n"
        code += "\t\tsm.get_waiter('endpoint_in_service').wait(EndpointName=endpoint)\n"
        code += "\texcept Exception as err:\n"
        code += "\t\tlogger.error('lambda_handler::err:{}'.format(err))\n"
    except Exception as err:
        logger.error("lambdaFunctionGenerator:Unable to write Lambda_function.py. Exiting. err: {}".format(err))

    return code


def createLambdaFunction():
    global function_arn
    global job_id
    lambda_name = 'ss_deploy-' + str(job_id)
    try:
        s3 = boto3.resource('s3')
        logger.info("Downloading the Lambda function template...")

        s3.Bucket("serverless-sagemaker").download_file('SagemakerReadyLambdaTemplate.zip',
                                                        '/tmp/SagemakerReadyLambdaTemplate.zip')

        if (os.path.exists('/tmp/SagemakerReadyLambdaTemplate.zip')):
            logger.info("...SagemakerReadyLambdaTemplate.zip download successfully")
        else:
            raise ValueError("Unable to download SagemakerReadyLambdaTemplate.zip!")

        # write lambda_function.py
        # TODO - figure out why we have to switch from .resource('s3') to .client('s3')
        s3 = boto3.client('s3')
        logger.info("writing lambda_function.py...")

        theCode = lambdaFunctionGenerator(origin)

        with open("/tmp/lambda_function.py", "w") as f:
            f.write(theCode)

        logger.info('adding custom lambda_function.py to upload...')
        zipper = zipfile.ZipFile('/tmp/SagemakerReadyLambdaTemplate.zip', 'a')

        zipper.write('/tmp/lambda_function.py')
        zipper.close()

        logger.info('uploading new compressed file to S3')
        # Send the zip file to our newly created S3 bucket
        with open('/tmp/SagemakerReadyLambdaTemplate.zip', 'rb') as data:
            s3.upload_fileobj(data, working_bucket, 'lambdafunction.zip')

        _lambda = boto3.client('lambda')

        iam = boto3.resource('iam')
        role = iam.Role(lambda_role)
        lambdaRoleARN = role.arn

        logger.info("creating the custom Lambda Function")
        createLambdaResponse = _lambda.create_function(
            FunctionName=lambda_name,
            Runtime='python3.6',
            Role=lambdaRoleARN,
            Handler='lambda_function.lambda_handler',
            Code={
                'S3Bucket': working_bucket,
                'S3Key': 'lambdafunction.zip'
            },
            Description='Lambda deploy function for job-id ' + job_id,
            Timeout=299,
            MemorySize=3008,
            Publish=True
        )

        function_arn = createLambdaResponse['FunctionArn']

        logger.info("create_function response: {}".format(createLambdaResponse))

    except Exception as err:
        logger.error("createLambdaFunction::unable to create a lambda function: Exiting. err: {}".format(err))


# Replace the variable data with numeric data
def encodeString(column, kv):
    try:
        frame_len = len(data)
        for x in range(0, frame_len):
            target = data.iloc[x, data.columns.get_loc(column)]
            newValue = kv[target]
            data.iloc[x, data.columns.get_loc(column)] = newValue
    except Exception as err:
        logger.error("encodeString::Error encoding string: {}".format(err))


def getNameIndex(column):

    columnnames = list()
    success = True
    uniqueindex = 0

    try:
        logger.info("\t encoding column: {}".format(column))
        # Create the array of unique column values
        for index, row in data.iterrows():
            if row[column] not in columnnames:
                columnnames.append(row[column])
            uniqueindex += 1

        unique_names = len(columnnames)

        index = [None] * unique_names
        uniqueindex = 0
        for i in range(0, unique_names):
            index[i] = i
            uniqueindex += 1

        name_index = dict(zip(columnnames, index))
    except Exception as err:
        success = False
        msg = "Error creating value-encoding key pair. Row# {} : {}".format(uniqueindex, err)
        logger.error("getNameIndex::" + msg)

    return success, name_index


def prepareForTraining(event):

    try:
        logger.info("preparing sleeper Lambda function...")
        # create sleeper function trigger
        createLambdaFunction()
        # link the trigger to S3 and the sleeper
        createLambdaTrigger()
        # initiate the training process
        sagemakerTrain(event)
    except Exception as err:
        logger.error("prepareForTraining:: Error while creating and linking lambda sleeper function: {}".format(err))


def createS3Bucket(bucket):
    """Create an S3 bucket based on the Job-ID
    """
    global statusCode
    global msg

    try:
        s3 = boto3.client('s3')
        logger.info("Creating S3 bucket '{}' for Training data and Model output.".format(bucket))
        s3CreateResponse = s3.create_bucket(
            ACL='private',
            Bucket=bucket
        )
        if str(s3CreateResponse).__contains__(bucket):  # Bucket created successfully
            logger.info("...s3 working bucket created successfully.")
    except Exception as err:
        msg = "Unable to create S3 working bucket.  Exiting.  Err: {}".format(err)
        logger.error("createS3Bucket::" + msg)
        return False

    return True


def transferFile(origin_bucket, origin_key, dest_bucket, dest_key):
    """Transfer the dataset from a source location to a working
    S3 bucket
    """

    global statusCode
    global msg
    global currentAge

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

        if asis is True:
            if keySize > asis_max_dataset:
                msg = "ML Factory currently only supports passthrough datasets under 933MB in size"
                logger.info(msg)
                statusCode = 413
                return False
        else:
            if keySize > prepped_max_dataset:
                msg = "ML Factory currently only supports prepping datasets under 401MB in size"
                logger.info(msg)
                statusCode = 413
                return False

        # Transfer the dataset to the working S3 directory
        logger.info("Streaming target dataset to working bucket...")

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(origin_bucket)
        obj = bucket.Object(origin_key)

        # Read into memory (don't write to disk)
        buffer = io.BytesIO(obj.get()['Body'].read())

        # buffer.seek(0)
        s3.Object(dest_bucket, dest_key).upload_fileobj(buffer)

        stop = time.time()
        currentAge += (stop - start)
        msg = "dataset transferred in {} seconds".format(stop - start)
        logger.info(msg)

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            # The object does not exist.
            stop = time.time()
            currentAge += (stop - start)
            msg = "Object: {} does not exist!".format(origin_key)
            logger.error("transferFile::" + msg)
            return False
        else:
            # Something else has gone wrong.
            stop = time.time()
            currentAge += (stop - start)
            msg = "Error during dataset transfer: {} ".format(e)
            logger.error("transferFile::" + msg)
            return False
    except Exception as err:
        stop = time.time()
        currentAge += (stop - start)
        msg = "Unable to transfer Dataset to S3! {}".format(err)
        logger.error("transferFile::" + msg)
        return False

    return True


def preprareDataset(algo, target, features, bucket, key):

    global statusCode
    global msg
    global currentAge
    global data

    allfields = False

    start = time.time()
    try:
        logger.info("copying dataset to local storage...")
        #  download dataset to /tmp
        s3 = boto3.client('s3')
        s3.download_file(bucket, key, '/tmp/' + key)
        logger.info("...dataset downloaded to local storage.")

        #  load into pandas
        logger.info("importing dataset into Pandas dataframe...")
        data = pd.read_csv('/tmp/' + key)

        # verify Target_field is present
        if target in data.columns:
            logger.info("Target_Field: {} column IS present in dataset".format(target))
            # do we train on all features or just specific features?
            if 'allfields' not in features:  # Assuming a specific feature list has been provided
                # verify Feature_list values are in dataset
                for feature in features:
                    if feature not in data.columns:
                        msg = "{} feature not present but expected for training. \
                        Unable to prepare dataset".format(feature)
                        logger.error(msg)
                        statusCode = 400
                        return False
            else: # no need to verify if features exist - use all features
                allfields = True
                logger.info("Training on all features of the dataset")
        else:
            msg = "Unable to prepate dataset. Target Field: {} not found in this dataset".format(target)
            logger.error("preprareDataset::" + msg)
            stop = time.time()
            currentAge += (stop - start)
            return False

        # drop columns that aren't needed
        if not allfields:
            for column in data:
                if column not in features:  # drop this column
                    logger.info("Dropping the {} column".format(column))
                    data = data.drop(columns=[column])
                    logger.info(data.shape)

        # Create missing value report
        nulls = (data.isnull().sum() / len(data)) * 100

        deleteNulls = []
        # Check if features with more than 20% missing values are in our training feature list
        if allfields:  # check all features for missing values
            for index, val in nulls.iteritems():
                if val > 50:
                    deleteNulls.append(index)
                    logger.info("{} feature has {}% missing values, will delete null rows".format(index, val))

        else:
            for feature in features:  # check features in feature list for missing values
                if nulls[feature] > 50:
                    deleteNulls.append(feature)
                    logger.info("training feature '{}' is missing {}% of it's values. will delete null \
                    rows".format(feature,round(nulls[feature])))

        logger.info("deleting obsersvations from columns with greater than 50% null values...")
        logger.info("Dataset shape prior to deletion: {}".format(data.shape))
        for d in deleteNulls:
            data = data.dropna(subset=[d])
        logger.info("Dateset shape after deletion: {}".format(data.shape))

        # Determine column types - we can't train on object (string) types
        logger.info("Determining column data types - we cannot train on string values without encoding them")
        toEncode = []
        dt = data.dtypes
        for index, val in dt.iteritems():
            logger.info("{} feature is probably a(n) {} data type".format(index, val))
            if str(val) == 'object':
                toEncode.append(index)

        # See if any columns from our feature list are on the 'to-encode' list - encode if applicable
        logger.info("encoding categorical columns...")
        if allfields:
            # encode all fields of type object
            for column in data:
                result, name_index = getNameIndex(column)
                if result is True:
                    encodeString(column, name_index)
                else:
                    return  False
        else:
            for feature in features:
                if feature in toEncode:
                    result, name_index = getNameIndex(feature)
                    if result is True:
                        encodeString(feature, name_index)
                    else:
                        return False

        # Create correlation report
        logger.info("generating correlation report...")
        corrRep = data.corr()
        logger.info(corrRep)

        logger.info("Splitting into train, validation and test sets...")
        # create training, validation and test sets
        train_data, validation_data, test_data = np.split(data.sample(frac=1, random_state=1729),
                                                              [int(0.7 * len(data)), int(0.9 * len(data))])

        logger.info("Moving {} column to be the first column in the training dataset".format(target))
        # create unified training dataset from target_field and features
        pd.concat([train_data[target], train_data.drop([target], axis=1)], axis=1).to_csv('/tmp/train.csv',index=False,
                                                                                                          header=False)
        logger.info("Moving {} column to be the first column in the validation dataset".format(target))
        pd.concat([validation_data[target], validation_data.drop([target], axis=1)], axis=1).to_csv(
            '/tmp/validation.csv',index=False,header=False)

        # move train and validation files to S3
        boto3.Session().resource('s3').Bucket(working_bucket).Object('train/train.csv').upload_file('/tmp/train.csv')
        boto3.Session().resource('s3').Bucket(working_bucket).Object('validation/validation.csv').upload_file(
            '/tmp/validation.csv')

    except Exception as err:
        msg = "Unable to prepare dataset for training: {}".format(err)
        logger.error("preprareDataset::" + msg)
        stop = time.time()
        logger.info("prepareDataset method took {} seconds".format(stop - start))
        currentAge += (stop - start)
        return False

    stop = time.time()
    logger.info("prepareDataset method took {} seconds".format(stop-start))
    currentAge += (stop - start)
    return True


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
    global job_id
    global target_field
    global feature_list
    global hyperparams

    try:
        logger.info("Extracting payload-parameters from the event object: {}".format(event))

        # Convert event object back to a dict from str
        # event = json.loads(event)
        logger.info("Setting Job_id = {}".format(job_id))

        # Get the built-in Algorithm to use
        if 'algorithm' not in event:
            msg = "Built-in algo to use not supplied in the parameters"
            logger.error("validateAPI::" + msg)
            return False
        else:
            origin = event['algorithm']
            origin = origin.lower()
            logger.info("Using {} built-in Algo".format(origin))
            working_bucket = origin + "-" + job_id
            logger.info("working bucket set to: {}".format(working_bucket))

        if origin not in master_algo:
            msg = "The requested algo ({}) is not supported.  Supported algos: {} ".format(origin, master_algo)
            logger.error("validateAPI::" + msg)
            return False

        # Get the HyperParams for this Algo
        if origin == 'xgboost':
            xgboost = smbuiltin.XGBoost(event)
            xgboost.verify()
            result = xgboost.verified
            xmsg = xgboost.msg
            if result is False:
                msg = "Required parameter values for running XGboost not detected or incorrect: {}".format(xmsg)
                logger.error("validateAPI::" + msg)
                return False
            else:
                logger.info("Minimum requirements to support a(n) {} training session found".format(origin))
                hyperparams = event['hyperparams']

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
                logger.error("validateAPI::" + msg)
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


        # Get the URL of the Dataset
        # TODO - Make this extensible so origin can be anywhere - not just S3

        if 's3_bucket' not in event:
            msg = "Unable to proceed. Dataset URL not found in the parameters - Fatal Error"
            logger.error("validateAPI::" + msg)
            return False
        else:
            dataset = event['s3_bucket']
            dataset = str(dataset)

        # Make sure S3 object path is provided correctly
        if "s3" not in dataset.lower():
            msg = "Unable to proceed. Dataset URL not formatted correctly - \
                requires: s3://bucket-name/[sub-directory]/objectname - Fatal Error"
            logger.error("validateAPI::" + msg)
            return False

        logger.info("DataSet URL provided: {}".format(dataset))
        pcs = dataset.split('/')
        origin_key = pcs[(len(pcs) - 1)]
        logger.info("ObjectKey: {}".format(origin_key))
        lastChar = dataset.rfind('/')
        origin_bucket = dataset[5:lastChar]
        logger.info("ObjectBucket = {} ".format(origin_bucket))
    except Exception as err:
        msg = "Unable to proceed. Unable to parse Datset. Error: {}".format(err)
        logger.error("validateAPI::" + msg)
        return False

    return True


def setupSNS(event):

    try:
        sns = boto3.client('sns')

        logger.info('Creating the SNS topic for severless-sagemaker notifications')

        # Create the Topic (does nothing if it already exists)
        rtopic = sns.create_topic(
            Name='serverless-sagemaker-topic'
        )
        if 'TopicArn' in rtopic:
            # get the ARN from the create_topic response
            rtopicArn = rtopic['TopicArn']
            logger.info("SNS Topic ARN: {}".format(rtopicArn))
        else:
            raise ValueError("SNS Topic NOT created or unreachable")

        logger.info('Topic ARN acquired: {}'.format(rtopicArn))

        # Create the Subscription if not already subscribed
        logger.info('Creating SNS Subscription...')

        # Check for existing subscription
        isSubscribed = sns.list_subscriptions_by_topic(
            TopicArn=rtopicArn
        )

        subscribe = True

        if 'Subscriptions' in isSubscribed:
            subs = isSubscribed['Subscriptions']
            for sub in subs:
                if sub['Endpoint'] == event['user_email']:
                    # This email is already subscribed - do not create a new subscription
                    logger.info("Subscription already exists for this user - reusing this subscription")
                    subscribe = False
        else:
            raise ValueError("Unable to query the subscriptions for this Topic - this could end badly.")

        if subscribe:
            logger.info("Creating new SNS subscription for  {}".format(event['user_email']))
            rSubscribe = sns.subscribe(
                TopicArn=rtopicArn,
                Protocol='email',
                Endpoint=event['user_email']
            )

    except Exception as err:
        logger.error("setupSNS::Error creating topic or subscribing to SNS: {}".format(err))


def lambda_handler(event, context):

    global statusCode
    global msg
    global job_id
    global target_field
    global feature_list


    try:
        logger.info("ss_process function triggered with stream: {}".format(event))

        # make sure this is only an Insert event from Dynamodb
        eventType = event['Records'][0]['eventName']
        logger.info("Eventtype = {}".format(eventType))

        if eventType == 'INSERT':
            job_id = event['Records'][0]['dynamodb']['NewImage']['job_id']['S']
            logger.info("Found job_id: {}".format(job_id))

            # isolate the DynamoDB entry from the DynamoDB stream/event data
            payload = event['Records'][0]['dynamodb']['NewImage']
            if payload is None:
                msg = "unable to read API request data from DynamoDB"
                logger.error(msg)
                raise ValueError(msg + " review CloudWatch logs")
            else:
                logger.info("Deserializing original API call payload...")

                # lazy-eval dynamodb
                boto3.resource('dynamodb')

                # Convert from 'Dynamodb' type to dict
                deserializer = boto3.dynamodb.types.TypeDeserializer()
                event = {k: deserializer.deserialize(v) for k,v in payload.items()}

                logger.info("Deserialized event object({}): {}".format(type(event), event))

                # Set up the notification framework in the event of an error
                setupSNS(event['payload'])

                # Validate that the API call is structured properly
                if validateAPI(event['payload']):
                    logger.info("Total elapsed time (in seconds): {}".format(currentAge))
                    # Create the Working Directory
                    if createS3Bucket(working_bucket):
                        logger.info("Total elapsed time (in seconds): {}".format(currentAge))
                        # Copy the Dataset to the Working directory
                        if transferFile(origin_bucket, origin_key, working_bucket, origin_key):
                            logger.info("Total elapsed time (in seconds): {}".format(currentAge))
                            # Check for Process as-is flag
                            if asis is True:
                                prepareForTraining(event)
                                logger.info("Total elapsed time (in seconds): {}".format(currentAge))
                            else:
                                preprareDataset(origin, target_field, feature_list, working_bucket, origin_key)
                                logger.info("Total elapsed time (in seconds): {}".format(currentAge))
                                prepareForTraining(event)
                                logger.info("Training process initiated, exiting function")
                        else:
                            raise ValueError("transferFile::Unable to transfer dataset to s3 working bucket - review \
                            CloudWatch logs")
                    else:
                        raise ValueError("createS3Bucket::Unable to create a working directory on S3 - \
                        review CloudWatch logs")
                else:
                    raise ValueError("validateAPI::API request structured incorrectly (HTTP 400) - review \
                    CloudWatch logs")

    except Exception as err:
        msg = "lambda_handler::Unable to process API request: {}".format(err)
        logger.error(msg)
        sendSNSMsg(msg, "Serverless-SageMaker Error notification")



# Below code for Testing on Local workstation - comment out before uploading to AWS Lambda or lambda_handler runs twice
"""
fakeCall = '{"target_field":"Global_Sales",\
	"feature_list":["Platform","Year_of_Release", "Genre", "Publisher", "Global_Sales", "Critic_Score", "Critic_Count",\
	"User_Score", "User_Count", "Developer", "Rating"],\
    "s3_bucket":"s3://videogame-sales/Video_Games.csv",\
    "algorithm":"XGBoost",\
    "hyperparams": {"objective": "reg-linear"}, \
    "user_email":"email@domain.com",\
    "asisdata": "True"}'

test = json.loads(fakeCall)

localResult = lambda_handler(test, None)

print(localResult)

"""
