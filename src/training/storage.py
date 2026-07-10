"""Minimal blob-storage abstraction: local disk, S3, or GCS behind one interface.

This replaces `/content/drive/MyDrive/...` scattered through every notebook
cell. The training code never needs to know which backend it's talking to —
same pattern as the provider-agnostic GPU dispatch layer you built at rapBot,
just applied to storage instead of compute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class BlobStore(Protocol):
    def download(self, remote_path: str, local_path: Path) -> None: ...
    def download_prefix(self, remote_prefix: str, local_dir: Path) -> None: ...
    def upload(self, local_path: Path, remote_path: str) -> None: ...
    def exists(self, remote_path: str) -> bool: ...


class LocalBlobStore:
    """For dev/test: `remote_path` is just a path under `root`."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, remote_path: str) -> Path:
        return self.root / remote_path

    def download(self, remote_path: str, local_path: Path) -> None:
        src = self._resolve(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(src.read_bytes())

    def download_prefix(self, remote_prefix: str, local_dir: Path) -> None:
        """Copies every file under `remote_prefix` into `local_dir`,
        preserving the relative directory structure — e.g. the `wavs/`
        folder alongside a training filelist."""
        src_dir = self._resolve(remote_prefix)
        local_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.rglob("*"):
            if f.is_file():
                dst = local_dir / f.relative_to(src_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(f.read_bytes())

    def upload(self, local_path: Path, remote_path: str) -> None:
        dst = self._resolve(remote_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(local_path.read_bytes())

    def exists(self, remote_path: str) -> bool:
        return self._resolve(remote_path).exists()


class S3BlobStore:
    """Thin wrapper over boto3. Kept separate from LocalBlobStore so any run
    can point `storage.backend` at `local` and skip AWS credentials entirely
    — useful for a dry run on your own machine before a real S3 upload."""

    def __init__(self, bucket: str):
        import boto3  # imported lazily so this module is importable without boto3

        self.bucket = bucket
        self._client = boto3.client("s3")

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, remote_path, str(local_path))

    def download_prefix(self, remote_prefix: str, local_dir: Path) -> None:
        """Downloads every object under `remote_prefix` (e.g. an S3 "folder"
        like `datasets/lupefiasco/.../wavs/`) into `local_dir`, preserving
        the relative key structure. Mirrors what `aws s3 sync` did in
        RUNBOOK.md's upload step, on the way back down."""
        local_dir.mkdir(parents=True, exist_ok=True)
        prefix = remote_prefix.rstrip("/") + "/"
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue  # S3 "directory marker" objects, not real files
                local_path = local_dir / key[len(prefix):]
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self._client.download_file(self.bucket, key, str(local_path))

    def upload(self, local_path: Path, remote_path: str) -> None:
        self._client.upload_file(str(local_path), self.bucket, remote_path)

    def exists(self, remote_path: str) -> bool:
        import botocore

        try:
            self._client.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except botocore.exceptions.ClientError:
            return False


def build_blob_store(backend: str, bucket: str, local_root: Path = Path("./_local_blob_store")) -> BlobStore:
    if backend == "local":
        return LocalBlobStore(local_root)
    if backend == "s3":
        return S3BlobStore(bucket)
    if backend == "gcs":
        raise NotImplementedError("GCS backend: same shape as S3BlobStore, swap boto3 for google-cloud-storage")
    raise ValueError(f"unknown storage backend: {backend}")
