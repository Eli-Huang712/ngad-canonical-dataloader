"""Internal canonical storage backend selection."""

from pathlib import Path
from typing import Any

from ngad_canonical_dataloader.backends.image import H264ImageBackend, JpegImageBackend
from ngad_canonical_dataloader.backends.table import LanceTableBackend, ParquetTableBackend


HY_CANONICAL_SCHEMA = "ngad_hy_canonical_lance_v2"
LANCE_JPEG_BACKEND = "lance_jpeg"
PARQUET_H264_BACKEND = "parquet_h264"


def create_storage_backends(
    table_root: Path,
    table_name: str,
    table_from_index: int,
    info: dict[str, Any],
) -> tuple[LanceTableBackend | ParquetTableBackend, JpegImageBackend | H264ImageBackend]:
    """Build the backend pair explicitly declared by one canonical table."""
    storage_backend = info.get("storage_backend")
    if storage_backend == LANCE_JPEG_BACKEND:
        if info.get("canonical_schema") != HY_CANONICAL_SCHEMA:
            raise ValueError(
                f"{table_root} Lance table must declare "
                f"canonical_schema={HY_CANONICAL_SCHEMA!r}."
            )
        lance_root = table_root / f"{table_name}.lance"
        if (table_root / "data").exists() or (table_root / "videos").exists():
            raise ValueError(
                f"{table_root} cannot contain Parquet/H.264 payload beside Lance."
            )
        if (
            not (lance_root / "_versions").is_dir()
            or not (lance_root / "data").is_dir()
            or not any((lance_root / "data").glob("*.lance"))
        ):
            raise ValueError(
                f"{table_root} must contain a published {table_name}.lance dataset."
            )
        return (
            LanceTableBackend(table_root, lance_root, table_from_index),
            JpegImageBackend(),
        )
    if storage_backend == PARQUET_H264_BACKEND:
        if (table_root / f"{table_name}.lance").exists():
            raise ValueError(
                f"{table_root} cannot contain Lance payload beside Parquet/H.264."
            )
        if not info.get("data_path") or not info.get("video_path"):
            raise ValueError(
                f"{table_root} parquet_h264 table must declare data_path and video_path."
            )
        data_path = Path(str(info["data_path"]))
        video_path = Path(str(info["video_path"]))
        if (
            data_path.is_absolute()
            or not data_path.parts
            or data_path.parts[0] != "data"
            or ".." in data_path.parts
            or video_path.is_absolute()
            or not video_path.parts
            or video_path.parts[0] != "videos"
            or ".." in video_path.parts
        ):
            raise ValueError(
                f"{table_root} data_path and video_path must stay under "
                "data/ and videos/."
            )
        if not (table_root / "data").is_dir() or not (table_root / "videos").is_dir():
            raise ValueError(
                f"{table_root} parquet_h264 table must contain data/ and videos/."
            )
        return (
            ParquetTableBackend(table_root, info),
            H264ImageBackend(table_root, info),
        )
    raise ValueError(
        f"{table_root} storage_backend must be "
        f"{LANCE_JPEG_BACKEND!r} or {PARQUET_H264_BACKEND!r}."
    )


__all__ = ["create_storage_backends"]
