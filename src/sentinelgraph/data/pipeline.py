"""Command-line orchestration for the SentinelGraph v0.1 data release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import duckdb

from sentinelgraph.data.contract import (
    create_transaction_table,
    validate_transaction_table,
)
from sentinelgraph.data.provenance import (
    CSV_NAME,
    acquire_dataset,
)
from sentinelgraph.data.quality import profile_transactions
from sentinelgraph.data.report import write_quality_report
from sentinelgraph.data.splits import TRAIN_FRACTION_BY_TIME, build_splits


def default_project_root() -> Path:
    """Resolve the repository root from this installed source tree."""
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _connection(project_root: Path) -> duckdb.DuckDBPyConnection:
    interim = project_root / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(interim / "paysim-v0.1.duckdb"))


def prepare_table(
    project_root: Path,
    csv_path: Path,
) -> duckdb.DuckDBPyConnection:
    """Create the typed analytical table used by all v0.1 checks."""
    connection = _connection(project_root)
    connection.execute("SET preserve_insertion_order = true")
    create_transaction_table(connection, csv_path)
    return connection


def run_all(
    project_root: Path,
    *,
    force_download: bool = False,
    train_fraction: float = TRAIN_FRACTION_BY_TIME,
) -> dict[str, Any]:
    """Run acquisition, validation, profiling, splitting, and reporting."""
    print("[1/5] Acquiring and verifying PaySim")
    csv_path, manifest = acquire_dataset(
        project_root,
        force=force_download,
    )

    print("[2/5] Loading typed DuckDB table")
    connection = prepare_table(project_root, csv_path)
    metadata_dir = project_root / "data" / "metadata"
    try:
        print("[3/5] Running data contract and exact quality profile")
        contract = validate_transaction_table(connection)
        _write_json(metadata_dir / "contract_results.json", contract)
        if not contract["passed"]:
            raise RuntimeError(
                "PaySim failed the release-blocking data contract; "
                "see data/metadata/contract_results.json"
            )
        profile = profile_transactions(connection)
        _write_json(metadata_dir / "quality_profile.json", profile)

        print("[4/5] Materialising chronological holdout artifacts")
        splits = build_splits(
            connection,
            project_root / "data" / "processed",
            train_fraction=train_fraction,
        )
        _write_json(metadata_dir / "split_manifest.json", splits)
    finally:
        connection.close()

    print("[5/5] Rendering the first data-quality report")
    write_quality_report(
        project_root / "docs" / "DATA_QUALITY_REPORT.md",
        manifest,
        contract,
        profile,
        splits,
    )
    return {
        "manifest": manifest,
        "contract": contract,
        "profile": profile,
        "splits": splits,
    }


def run_existing(project_root: Path, command: str, train_fraction: float) -> None:
    """Run a selected local stage after acquisition."""
    csv_path = project_root / "data" / "raw" / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist; run the 'acquire' or 'all' command"
        )
    metadata_dir = project_root / "data" / "metadata"
    connection = prepare_table(project_root, csv_path)
    try:
        if command == "validate":
            contract = validate_transaction_table(connection)
            _write_json(metadata_dir / "contract_results.json", contract)
            if not contract["passed"]:
                raise RuntimeError("data contract failed")
        elif command == "profile":
            _write_json(
                metadata_dir / "quality_profile.json",
                profile_transactions(connection),
            )
        elif command == "split":
            _write_json(
                metadata_dir / "split_manifest.json",
                build_splits(
                    connection,
                    project_root / "data" / "processed",
                    train_fraction=train_fraction,
                ),
            )
        else:
            raise ValueError(f"unsupported command: {command}")
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the v0.1 command-line parser."""
    parser = argparse.ArgumentParser(
        description="SentinelGraph v0.1 PaySim data pipeline"
    )
    parser.add_argument(
        "command",
        choices=("all", "acquire", "validate", "profile", "split", "report"),
        nargs="?",
        default="all",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root(),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="replace the existing versioned Kaggle archive",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=TRAIN_FRACTION_BY_TIME,
        help="fraction of observed time steps assigned to training",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected CLI stage."""
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    if args.command == "all":
        run_all(
            project_root,
            force_download=args.force_download,
            train_fraction=args.train_fraction,
        )
    elif args.command == "acquire":
        acquire_dataset(project_root, force=args.force_download)
    elif args.command in {"validate", "profile", "split"}:
        run_existing(project_root, args.command, args.train_fraction)
    elif args.command == "report":
        metadata_dir = project_root / "data" / "metadata"
        write_quality_report(
            project_root / "docs" / "DATA_QUALITY_REPORT.md",
            _read_json(metadata_dir / "dataset_manifest.json"),
            _read_json(metadata_dir / "contract_results.json"),
            _read_json(metadata_dir / "quality_profile.json"),
            _read_json(metadata_dir / "split_manifest.json"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
