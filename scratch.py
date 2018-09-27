import boto3
import botocore
import io

def transfer():
    origin_bucket = 'serverless-sage-lz'
    origin_key = 'Fake_DataSet'
    dest_bucket = 'xgboost-1537985974-530944'
    dest_key = 'train/Fake_DataSet'


    try:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(origin_bucket)
        obj = bucket.Object(origin_key)

        buffer = io.BytesIO(obj.get()['Body'].read())
        # Read into memory (don't write to disk)

        #buffer.seek(0)
        s3.Object(dest_bucket, dest_key).upload_fileobj(buffer)

    except Exception as err:
        print(err)




transfer()