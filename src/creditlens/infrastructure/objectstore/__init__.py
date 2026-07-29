"""对象存储 Adapter：本地文件系统兜底 + MinIO。"""

from pathlib import Path
from urllib.parse import urlparse

from creditlens.common.config import Settings


class LocalFsObjectStore:
    """本地文件系统实现，接口与 MinIO Adapter 一致。

    uri 形态：file://{root}/{bucket}/{key}
    """

    def __init__(self, root: str):
        self._root = Path(root)

    def _path(self, bucket: str, key: str) -> Path:
        # key 由服务端构造；此处仍拒绝路径穿越作为纵深防御
        if ".." in key.replace("\\", "/").split("/"):
            raise ValueError("object key must not contain '..'")
        return self._root / bucket / key

    def put(self, bucket: str, key: str, data: bytes, content_type: str) -> str:
        path = self._path(bucket, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"file://{path.as_posix()}"

    def get(self, uri: str) -> bytes:
        return Path(self._uri_to_path(uri)).read_bytes()

    def exists(self, uri: str) -> bool:
        return Path(self._uri_to_path(uri)).exists()

    @staticmethod
    def _uri_to_path(uri: str) -> str:
        if not uri.startswith("file://"):
            raise ValueError(f"not a file uri: {uri}")
        return uri.removeprefix("file://")


class MinioObjectStore:
    """MinIO 实现；uri 形态 s3://{bucket}/{key}。"""

    def __init__(self, settings: Settings):
        from minio import Minio

        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    def put(self, bucket: str, key: str, data: bytes, content_type: str) -> str:
        import io

        self._client.put_object(
            bucket, key, io.BytesIO(data), length=len(data), content_type=content_type
        )
        return f"s3://{bucket}/{key}"

    def get(self, uri: str) -> bytes:
        bucket, key = self._parse(uri)
        response = self._client.get_object(bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def exists(self, uri: str) -> bool:
        bucket, key = self._parse(uri)
        try:
            self._client.stat_object(bucket, key)
            return True
        except Exception:
            return False

    @staticmethod
    def _parse(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise ValueError(f"not an s3 uri: {uri}")
        return parsed.netloc, parsed.path.lstrip("/")


def build_object_store(settings: Settings):
    if settings.object_store_backend == "minio":
        return MinioObjectStore(settings)
    return LocalFsObjectStore(settings.local_object_root)
