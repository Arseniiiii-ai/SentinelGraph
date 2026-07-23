"""Acquire PaySim and write auditable provenance metadata."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATASET_SLUG = "ealaxi/paysim1"
DATASET_VERSION = 2
DATASET_PAGE = "https://www.kaggle.com/datasets/ealaxi/paysim1"
DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "ealaxi/paysim1?datasetVersionNumber=2"
)
SIMULATOR_REPOSITORY = "https://github.com/EdgarLopezPhD/PaySim"
ARCHIVE_NAME = "paysim1-v2.zip"
CSV_NAME = "PS_20174392719_1491204439457_log.csv"
EXPECTED_CSV_SHA256 = "16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b"
DATASET_LICENSE = "CC BY-SA 4.0"
DATASET_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
CITATION = (
    "E. A. Lopez-Rojas, A. Elmir, and S. Axelsson. "
    '"PaySim: A financial mobile money simulator for fraud detection". '
    "28th European Modeling and Simulation Symposium, 2016."
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(
    path: Path,
    *,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    """Create checksum and size metadata for a file."""
    recorded_path = (
        path.resolve().relative_to(relative_to.resolve())
        if relative_to is not None
        else path
    )
    return {
        "path": str(recorded_path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def download_archive(raw_dir: Path, *, force: bool = False) -> Path:
    """Download the public Kaggle versioned archive atomically."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / ARCHIVE_NAME
    if destination.exists() and not force:
        return destination

    partial = destination.with_suffix(".zip.part")
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={"User-Agent": "SentinelGraph/0.1 data-acquisition"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        with partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)
    return destination


def extract_and_verify(archive_path: Path, raw_dir: Path) -> Path:
    """Extract only the canonical CSV and verify its known content hash."""
    csv_path = raw_dir / CSV_NAME
    if csv_path.exists() and sha256_file(csv_path) == EXPECTED_CSV_SHA256:
        return csv_path

    with zipfile.ZipFile(archive_path) as archive:
        member_names = archive.namelist()
        if CSV_NAME not in member_names:
            raise ValueError(
                f"{CSV_NAME} is absent from archive; members={member_names!r}"
            )
        partial = csv_path.with_suffix(".csv.part")
        with archive.open(CSV_NAME) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        partial.replace(csv_path)

    actual_hash = sha256_file(csv_path)
    if actual_hash != EXPECTED_CSV_SHA256:
        raise ValueError(
            f"PaySim CSV checksum mismatch: {actual_hash}; "
            f"expected {EXPECTED_CSV_SHA256}"
        )
    return csv_path


def acquire_dataset(
    project_root: Path,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Acquire PaySim, verify it, and persist a machine-readable manifest."""
    raw_dir = project_root / "data" / "raw"
    metadata_dir = project_root / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    archive_path = download_archive(raw_dir, force=force)
    csv_path = extract_and_verify(archive_path, raw_dir)
    manifest = {
        "dataset": "PaySim1 synthetic financial transactions",
        "dataset_slug": DATASET_SLUG,
        "dataset_version": DATASET_VERSION,
        "source_page": DATASET_PAGE,
        "download_url": DOWNLOAD_URL,
        "simulator_repository": SIMULATOR_REPOSITORY,
        "dataset_license": DATASET_LICENSE,
        "dataset_license_url": DATASET_LICENSE_URL,
        "simulator_code_license": "GPL-3.0",
        "citation": CITATION,
        "synthetic_data": True,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": file_record(archive_path, relative_to=project_root),
        "csv": file_record(csv_path, relative_to=project_root),
        "expected_csv_sha256": EXPECTED_CSV_SHA256,
        "checksum_verified": True,
    }
    manifest_path = metadata_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, manifest
