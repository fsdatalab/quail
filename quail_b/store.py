"""Where the benchmark's files live: the public bucket or a directory.

Every store holds the same relative layout, so the label loader and
the corpus fetch do not care which one they read.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pyarrow as pa
from pyarrow import parquet as pq

GROUND_TRUTH_ROOT = "ground_truth/quailb/schema_v1"
PUBLIC_BUCKET = "quail-bench"


class LocalFiles:
    """A directory that holds the same layout as the bucket."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def read_bytes(self, path: str) -> bytes:
        return (self.root / path.lstrip("/")).read_bytes()

    def list_files(self, path: str) -> list[str]:
        base = self.root / path.lstrip("/")
        return sorted(
            item.relative_to(self.root).as_posix()
            for item in base.rglob("*") if item.is_file())

    def write_json(self, path: str, payload: dict) -> None:
        destination = self.root / path.lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def write_parquet(self, path: str, table: pa.Table) -> None:
        destination = self.root / path.lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd")



class S3Files:
    """Read-only, anonymous access to a public bucket.

    Plain HTTPS against the S3 REST API, so no AWS SDK or credentials
    are needed to run the benchmark.
    """

    _NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

    def __init__(self, bucket: str = PUBLIC_BUCKET, retries: int = 3):
        self.bucket = bucket
        self.base_url = f"https://{bucket}.s3.amazonaws.com/"
        self.retries = retries

    def _get(self, url: str) -> bytes:
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code < 500 or attempt == self.retries - 1:
                    raise
            except urllib.error.URLError:
                if attempt == self.retries - 1:
                    raise
            time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def read_bytes(self, path: str) -> bytes:
        key = path.lstrip("/")
        try:
            return self._get(self.base_url + urllib.parse.quote(key))
        except urllib.error.HTTPError as error:
            if error.code in (403, 404):
                raise FileNotFoundError(
                    f"s3://{self.bucket}/{key}") from error
            raise

    def list_files(self, path: str) -> list[str]:
        prefix = path.lstrip("/").rstrip("/") + "/"
        keys = []
        token = None
        while True:
            query = {"list-type": "2", "prefix": prefix}
            if token:
                query["continuation-token"] = token
            page = ElementTree.fromstring(
                self._get(self.base_url + "?" + urllib.parse.urlencode(query)))
            keys.extend(
                key.text for key in page.iter(f"{self._NS}Key")
                if key.text.endswith((".json", ".parquet")))
            token = page.findtext(f"{self._NS}NextContinuationToken")
            if not token:
                return sorted(keys)

    def write_json(self, path: str, payload: dict) -> None:
        raise PermissionError("the public bucket is read-only")

    def write_parquet(self, path: str, table: pa.Table) -> None:
        raise PermissionError("the public bucket is read-only")
