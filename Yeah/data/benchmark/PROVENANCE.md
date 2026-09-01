# Benchmark dataset provenance

This directory is the input contract for Large-scale Transfer Benchmark v0.

## Layout

- `images/` copied files used by the benchmark
- `manifest.csv` columns: image_id, image_path, identity_id, source
- `provenance.json` machine-readable source record

## Sources

1. **local_raw** — existing `data/raw` images already in this repository.
2. **lfw** — Labeled Faces in the Wild (University of Massachusetts), http://vis-www.cs.umass.edu/lfw/ — public academic dataset for research. One image per identity is copied when fetch succeeds.

No undocumented downloads are performed.
This benchmark does not require identity-pair calibration.
