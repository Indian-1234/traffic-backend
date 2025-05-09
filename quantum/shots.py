import boto3

# Initialize the S3 client
s3_client = boto3.client('s3')

def count_s3_objects(bucket_name):
    # Get the list of objects in the specified S3 bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name)

    # Count the total number of objects in the bucket
    if 'Contents' in response:
        total_shots = len(response['Contents'])
        print(f"Total shots in the S3 bucket '{bucket_name}': {total_shots}")
    else:
        print(f"No objects found in the bucket '{bucket_name}'")

# Example usage
bucket_name = 'amazon-braket-quantiumhitter'  # Replace with your S3 bucket name
count_s3_objects(bucket_name)
