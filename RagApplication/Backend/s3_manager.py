import logging
from typing import Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("enhanced_rag")

class S3Manager:
    def __init__(self, config):
        self.config = config
        self.s3_client = None
        if self.config.s3_enabled:
            self.s3_client = boto3.client(
                "s3",
                region_name=self.config.aws_region,
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key
            )

    def upload_file(self, file_path: str) -> Optional[str]:
        if not self.s3_client:
            logger.info("S3 not enabled; skipping upload.")
            return None
        try:
            file_name = file_path.split("/")[-1]
            object_key = f"documents/{file_name}"
            self.s3_client.upload_file(
                Filename=file_path,
                Bucket=self.config.s3_bucket_name,
                Key=object_key
            )
            logger.info(f"Uploaded to S3: s3://{self.config.s3_bucket_name}/{object_key}")
            return object_key
        except ClientError as e:
            logger.error(f"S3 upload error: {str(e)}")
            return None

    def generate_presigned_url(self, object_key: str, expiration: int = 3600) -> Optional[str]:
        if not self.s3_client or not object_key:
            return None
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.config.s3_bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Presigned URL error: {str(e)}")
            return None

    def delete_file(self, object_key: str) -> bool:
        if not self.s3_client:
            logger.info("S3 not enabled; skipping deletion.")
            return False
        try:
            self.s3_client.delete_object(Bucket=self.config.s3_bucket_name, Key=object_key)
            logger.info(f"Deleted S3 object: {object_key}")
            return True
        except ClientError as e:
            logger.error(f"S3 deletion error: {str(e)}")
            return False
