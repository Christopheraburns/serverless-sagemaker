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

# A few global variables
statusCode = ""
msg = ""
origin = ""  # Set the origin equal to the Algo that is being requested
max_dataset = 933000000  # 933MB tested

# Configure logging
logging.basicConfig(filename='lambda.log', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configure the bucket name and Job-ID
output_location = ""
job_id = str(time.time()).replace(".", "-")
working_bucket = ""

# Master Built-in list (incoming API calls must use one of these)
master_algo = {"kmeans", "pca", "lda", "factorization-machines", "linear-learner", \
               "ntm", "randomcutforest", "seq2seq", "xgboost", "object-detection", \
               "image-classification", "forecasting-deepar", "blazingtext", "knn"}


"""
def sagemakerTrain():
    try:
        # get the ARN of the executing role (to pass to Sagemaker for training)
        role = 'arn:aws:iam::056149205531:role/service-role/AmazonSageMaker-ExecutionRole-20180112T102983'
        s3_train_data = 's3://{}/train/{}'.format(bucket, dataset)
        container = get_image_uri(boto3.Session().region_name, 'linear-learner')

        session = sagemaker.Session()

        # set up the training params
        linear = sagemaker.estimator.Estimator(container,
                                               role,
                                               train_instance_count=1,
                                               train_instance_type='ml.c4.xlarge',
                                               output_path=output_location,
                                               sagemaker_session=session)

        # set up the hyperparameters
        linear.set_hyperparameters(feature_dim=13,
                                   predictor_type='regressor',
                                   epochs=10,
                                   loss='absolute_loss',
                                   optimizer='adam',
                                   mini_batch_size=200)

        linear.fit({'train': s3_train_data}, wait=False)


    except Exception as err:
        logger.error("Error while launching SageMaker training: {}".format(err))
        
        
def createLambdaTrigger():
    config = "<NotificationConfiguration><CloudFunctionConfiguration>"
    config += "<Filter><S3Key><FilterRule>"
    config += "<Name>prefix</Name><Value>" + "" + "/</Value>"
    config += "</FilterRule></S3Key></Filter>"
    config += "<Id>ObjectCreatedEvents</Id>"
    config += "<CloudFunction>" + "" + "</CloudFunction>"
    config += "<Event>s3:ObjectCreated:*</Event>"
    config += "</CloudFunctionConfiguration></NotificationConfiguration>"
    

def lambdaFunctionGenerator(origin):
    try:
        # Import statements
        code = "import os\n"
        code += "import io\n"
        code += "import boto3\n"
        code += "import sagemaker\n"
        code += "from sagemaker import get_execution_role\n"
        code += "\n"

        # S3 setup
        code += "bucket = '" + bucket + "'\n"
        code += "prefix = '" + prefix1 + "'\n"
        code += "output_location = 's3://{}/{}/output'.format(bucket, prefix)\n"
        code += "job_id = 'job-epoch'\n"
        code += "\n"
        code += "role = get_execution_role()\n"
        code += "\n"

        # Handler function
        code += "def lambda_handler(event, context):\n"
        code += "\tsm = boto3.client('sagemaker')\n"
        code += "\n"
        # prepare a model context for hosting
        code += "\thosting_container = {\n"
        code += "\t\t'Image': '" + origin + "',\n"
        code += "\t\t'ModelDataUrl': 's3://" + bucket + "/output/" + job_id + "/output/model.tar.gz'\n"
        code += "\t}\n"
        code += "\tcreate_model_response = sm.create_model(\n"
        code += "\t\tModelName='" + job_id + "',\n"
        code += "\t\tExecutionRoleArn=role,\n"
        code += "\t\tPrimaryContainer=hosting_container)\n"
        code += "\n"

        # Create an endpoint configuration
        code += "\tendpoint_config='" + job_id + "-endpoint-config'\n"
        code += "\tcreate_endpoint_config_response = sm.create_endpoint_config(\n"
        code += "\t\tEndpointConfigName=endpoint_config,\n"
        code += "\t\tProductionVariants=[{\n"
        code += "\t\t\t'InstanceType': 'ml.m4.xlarge',\n"  # TODO find a way to make these values configurable
        code += "\t\t\t'InitialIntanceCount':1,\n"
        code += "\t\t\t'ModelName':'" + job_id + "',\n"
        code += "\t\t\t'VariantName':'AllTraffic'}]\n"
        code += "\t)"
        code += "\n"

        # Deploy the endpoint - but don't wait around for it
        code += "\tendpoint = '" + job_id + "-endpoint'\n"
        code += "\n"
        code += "\tcreate_endpoint_response=sm.create_endpoint(\n"
        code += "\t\tEndpointName=endpoint,\n"
        code += "\t\tEndpointConfigName=endpoint_config\n"
        code += "\t)"
    except Exception as err:
        logger.error("Unable to write Lambda_function.py. Exiting. err: {}".format(err))

    return code


def createLambdaFunction():
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

        zipper.write('/tmp/lambda_function.py', '/tmp/SagemakerReadyLambdaTemplate.zip')
        zipper.close()

        logger.info('uploading new compressed file to S3')
        # Send the zip file to our newly created S3 bucket
        with open('/tmp/SagemakerReadyLambdaTemplate.zip', 'rb') as data:
            s3.upload_fileobj(data, bucket, 'lambdafunction.zip')

        _lambda = boto3.client('lambda')

        logger.info("creating the custom Lambda Function")
        createLambdaResponse = _lambda.create_function(
            FunctionName='lambda-deploy-' + job_id,
            Runtime='python3.6',
            Role=lambdaRoleARN,
            Handler='lambda_function.lambda_handler',
            Code={
                'S3Bucket': bucket,
                'S3Key': 'lambdafunction.zip'
            },
            Description='Lambda deploy function for job-id ' + job_id,
            Timeout=299,
            MemorySize=2048,
            Publish=True
        )

        function_arn = createLambdaResponse['FunctionArn']

        logger.info(createLambdaResponse)
    except Exception as err:
        logger.error("unable to create a lambda function: Exiting. err: {}".format(err))
"""


def createS3Bucket(bucket):
    """Create an S3 bucket based on the Job-ID with a training sub-folder
    """
    global statusCode
    global msg

    s3 = boto3.client('s3')
    try:
        logger.info("Creating S3 bucket '{}' and sub directories for Training data and Model output.".format(bucket))
        s3CreateResponse = s3.create_bucket(
            ACL='private',
            Bucket=bucket
        )
        if str(s3CreateResponse).__contains__(bucket):  # Bucket created successfully
            logger.info("...s3 working bucket created successfully. Creating working bucket sub-folders...")
    except Exception as err:
        msg = "Unable to create S3 working bucket.  Exiting.  Err: {}".format(err)
        logger.error(msg)
        statusCode = 500
        return False

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
        msg = "s3 'train' sub-directory not created.  Exiting. Err: {}".format(err)
        logger.error(msg)
        statusCode = 500
        return False

    return True


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
            msg = "dataset transferred in {} seconds".format(xferTime)
            logger.info(msg)
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
            msg = "Error during dataset transfer: {} ".format(e)
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
    print("Safe to Prepare dataset for SageMaker training")
    # TODO - probably need a function for each built in to make certain the minimum requirements
    # for that particular built-in are met.
    # make sure the dataset is in shape for training.


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

    statusCode = 200
    dataready = True
    target_field = ""
    feature_list = {}

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
    asis = event['asisdata']
    if asis is None:  # Default to as-is
        logger.info("Processing dataset 'AS IS'")
        logger.info("'AS-IS processing not specified - will default to NOT preparing the dataset for Sagemaker")
    elif asis is True:
        logger.info("Processing dataset 'AS IS'")
        logger.info("'AS-IS processing set to TRUE - will default to NOT preparing the dataset for Sagemaker")
    else:
        logger.info("'AS-IS' set to FALSE - Will attempt to prepare dataset for Sagemaker")
        dataready = False
        #  We need Target Field & Feature_List - just check if they are in the API call - we will verify they
        #  actually exist in the data later

        target_field = event['target_field']
        if target_field is None:
            msg = 'Unable to train on this dataset.  Target_Field has not been supplied'
            logger.error(msg)
            statusCode = 400
            return False

        logger.info("Dataset Target Field = {}".format(target_field))
        feature_list = event['feature_list']
        if feature_list is None:  # Default to use all fields
            feature_list = {'all'}
        logger.info("Using the following features of the data set:")
        for feature in feature_list:
            logger.info(feature)

    # Get the URL of the Dataset
    # TODO - Make this extensible so origin can be anywhere - not just S3

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
        logger.error(msg)
        statusCode = 400
        return False

    return True


def lambda_handler(event, context):
    global statusCode
    global msg

    # Validate the API call is structured properly
    if validateAPI(event):
        # Create the Working Directory
        if createS3Bucket(working_bucket):
            # Copy the Dataset to the Working directory
            if transferFile(origin_bucket, origin_key, working_bucket,'train/' + origin_key):
                preprareForTraining(origin, event['target_field'], event['feature_list'])
        else:
            statusCode = 500
            msg = "Unable to transfer dataset"

    return {
        "statusCode": statusCode,
        "body": json.dumps(msg)
    }


# Below code for Testing on Local workstation - comment out before uploading to AWS Lambda or lambda_handler runs twice

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
