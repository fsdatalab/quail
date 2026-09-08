"""Copy the labels and the sf=0.1 corpus from the Modal volume to S3.

The copy runs inside Modal, so nothing is downloaded to the machine
that launches it. AWS keys come from the local environment and are
sent to Modal only for this run:

    export AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
    export AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)
    uv run modal run tools/publish_data_to_s3.py --bucket quail-bench

Paths in the bucket match the paths on the volume, so the label loader
can read either one.
"""

import modal

PATHS = ("ground_truth/quailb/schema_v1", "quailb_data/sf0.1")

app = modal.App("quail-milestone1")
image = modal.Image.debian_slim(python_version="3.12").pip_install("awscli")
results = modal.Volume.from_name("quail-results")
aws_keys = modal.Secret.from_local_environ(
    ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])


@app.function(image=image, volumes={"/results": results},
              secrets=[aws_keys], timeout=3600)
def publish(bucket: str) -> None:
    import subprocess

    for path in PATHS:
        print(f"[publish] {path} -> s3://{bucket}/{path}", flush=True)
        subprocess.run(
            ["aws", "s3", "sync", "--no-progress", f"/results/{path}",
             f"s3://{bucket}/{path}"],
            check=True)


@app.local_entrypoint()
def main(bucket: str) -> None:
    publish.remote(bucket)
    print(f"[publish] done; check with: aws s3 ls "
          f"s3://{bucket}/{PATHS[0]}/collections/ --no-sign-request")
