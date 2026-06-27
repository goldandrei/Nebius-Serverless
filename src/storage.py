"""S3-compatible object storage via boto3."""
import json
import os
from pathlib import Path


def _client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["STORAGE_ENDPOINT"],
        aws_access_key_id=os.environ["STORAGE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["STORAGE_SECRET_KEY"],
        region_name=os.environ.get("NEBIUS_REGION", "eu-north1"),
    )


def put_results(results: dict, bucket: str = None, key: str = "results.json") -> None:
    bucket = bucket or os.environ["STORAGE_BUCKET"]
    body   = json.dumps(results, indent=2).encode()
    _client().put_object(Bucket=bucket, Key=key, Body=body,
                         ContentType="application/json")
    print(f"  uploaded {key} to s3://{bucket}/{key}")


def get_results(bucket: str = None, key: str = "results.json") -> dict:
    bucket = bucket or os.environ["STORAGE_BUCKET"]
    obj    = _client().get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def save_local(results: dict, path: Path) -> None:
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
