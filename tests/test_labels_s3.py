"""CPU checks for the anonymous S3 reader, with the HTTP layer faked."""

import urllib.error

import pytest

from quail_b.labels import GROUND_TRUTH_ROOT, S3Files

_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>quail-bench</Name><IsTruncated>{truncated}</IsTruncated>
  {token}
  {contents}
</ListBucketResult>"""


def _page(keys, token=None):
    contents = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    return _PAGE.format(
        truncated="true" if token else "false",
        token=(f"<NextContinuationToken>{token}</NextContinuationToken>"
               if token else ""),
        contents=contents).encode()


def test_list_files_follows_pages_and_keeps_only_data_files(monkeypatch):
    files = S3Files()
    calls = []

    def fake_get(url):
        calls.append(url)
        if "continuation-token=next" in url:
            return _page([f"{GROUND_TRUTH_ROOT}/collections/gt_b/manifest.json",
                          f"{GROUND_TRUTH_ROOT}/collections/gt_b/notes.txt"])
        return _page([f"{GROUND_TRUTH_ROOT}/collections/gt_a/manifest.json",
                      f"{GROUND_TRUTH_ROOT}/collections/gt_a/labels.parquet"],
                     token="next")

    monkeypatch.setattr(files, "_get", fake_get)

    listed = files.list_files(f"/{GROUND_TRUTH_ROOT}/collections")

    assert listed == sorted([
        f"{GROUND_TRUTH_ROOT}/collections/gt_a/labels.parquet",
        f"{GROUND_TRUTH_ROOT}/collections/gt_a/manifest.json",
        f"{GROUND_TRUTH_ROOT}/collections/gt_b/manifest.json",
    ])
    assert len(calls) == 2
    assert calls[0].startswith("https://quail-bench.s3.amazonaws.com/?")
    prefix = GROUND_TRUTH_ROOT.replace("/", "%2F")
    assert f"prefix={prefix}%2Fcollections%2F" in calls[0]


def test_read_bytes_maps_a_missing_key_to_file_not_found(monkeypatch):
    files = S3Files(bucket="example")

    def fake_get(url):
        assert url == "https://example.s3.amazonaws.com/a/b%20c.json"
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(files, "_get", fake_get)

    with pytest.raises(FileNotFoundError):
        files.read_bytes("/a/b c.json")


def test_the_public_bucket_is_read_only():
    with pytest.raises(PermissionError):
        S3Files().write_json("x.json", {})
