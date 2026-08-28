"""Internal canonical storage backend selection."""

from pathlib import Path
from typing import Any

from ngad_canonical_dataloader.backends.image import H264ImageBackend, JpegImageBackend
from ngad_canonical_dataloader.backends.table import LanceTableBackend, ParquetTableBackend


HY_CANONICAL_SCHEMA = "ngad_hy_canonical_lance_v2"


def create_storage_backends(
    root: Path,
    info: dict[str, Any],
) -> tuple[LanceTableBackend | ParquetTableBackend, JpegImageBackend | H264ImageBackend]:
    """Build the one supported table/image backend pair identified under a root."""
    if (
        info.get("canonical_schema") == HY_CANONICAL_SCHEMA
        and (root / "_versions").is_dir()
        and len(list((root / "data").glob("*.lance"))) == 1
    ):
        return LanceTableBackend(root), JpegImageBackend()
    if info.get("data_path") and info.get("video_path"):
        return ParquetTableBackend(root, info), H264ImageBackend(root, info)
    raise ValueError(f"Cannot identify a supported NGAD canonical storage backend under {root}.")


__all__ = ["create_storage_backends"]
