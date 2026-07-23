# PaySim Data Source and Usage

## Dataset

- Name: **Synthetic Financial Datasets For Fraud Detection (PaySim1)**
- Publisher: Edgar Lopez-Rojas (`ealaxi` on Kaggle)
- Canonical distribution:
  <https://www.kaggle.com/datasets/ealaxi/paysim1>
- Pinned dataset version: **2**
- Versioned download:
  <https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1?datasetVersionNumber=2>
- Canonical file: `PS_20174392719_1491204439457_log.csv`
- Dataset license: **Creative Commons Attribution-ShareAlike 4.0**
  (<https://creativecommons.org/licenses/by-sa/4.0/>)

The dataset license is distinct from the GPL-3.0 license applied to the PaySim
simulator source code at <https://github.com/EdgarLopezPhD/PaySim>.

## Provenance

PaySim is synthetic. The simulator was calibrated from aggregated transaction
logs and injects simulated malicious behaviour. The Kaggle file is described by
the publisher as a one-quarter-scale sample created for Kaggle.

The pipeline records the downloaded archive and extracted CSV byte sizes and
SHA-256 checksums in `data/metadata/dataset_manifest.json`. The expected
canonical CSV digest is:

```text
16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b
```

Acquisition fails closed if the extracted content does not match this digest.

## Required attribution

When using or redistributing the dataset, preserve CC BY-SA 4.0 attribution and
cite:

> E. A. Lopez-Rojas, A. Elmir, and S. Axelsson. “PaySim: A financial mobile
> money simulator for fraud detection.” 28th European Modeling and Simulation
> Symposium, 2016.

## Reproducible acquisition

```bash
uv sync --extra dev
uv run sentinelgraph-data acquire
```

Raw files are intentionally ignored by Git. The machine-readable provenance,
contract results, profile, and split manifest are retained.
