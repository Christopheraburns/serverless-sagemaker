#!/bin/bash


file = "lambda_function.zip"

if [ -f $file ] ; then
    rm $file
fi

# zip the directory contents (exclude stuff we don't want to to send to Lambda)
zip -r lambda_function.zip . -x proc.sh -x lambda.log -x datasets/**\* -x assets/**\* -x venv/**\* -x .git/**\* -x botocore/**\* \
-x boto3-1.9.23.dist-info/**\* -x botocore-1.12.23.dist-info/**\* -x boto3/**\* -x dateutil/**\* \
-x scipy/.libs/libopenblasp-r0-39a31c03.2.18.so

# copy the function.zip file to S3
aws s3 cp lambda_function.zip s3://chris-misc/lambda_function.zip

# Update the lambda function the latest code
aws lambda update-function-code --function-name ss_process --s3-bucket chris-misc --s3-key lambda_function.zip
