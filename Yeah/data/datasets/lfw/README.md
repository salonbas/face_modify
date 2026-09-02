# Labeled Faces in the Wild (LFW)

This repository entry describes the standard **lfw-funneled** variant of
Labeled Faces in the Wild: 13,233 JPEG images across 5,749 identities.

## Storage and provenance

The image directory is an `images` symlink to the existing scikit-learn cache:
`~/scikit_learn_data/lfw_home/lfw_funneled`. No image binary is copied into
this repository. The cache was audited locally and contains the complete
standard image count and the sklearn/official pair protocol files. The source
is the public LFW dataset from the University of Massachusetts:
<http://vis-www.cs.umass.edu/lfw/>. Use is subject to the dataset's original
research-use/provenance terms; consult the source for licensing details.

`manifest.csv` is the canonical image inventory. `pairs/` contains only pair
definitions: development train/test and the ten official evaluation folds,
plus their combined `evaluation_10fold.csv`. No embeddings, similarities,
ROC data, or thresholds are stored here.

## Usage

```python
from advface.paths import get_dataset_path

lfw = get_dataset_path("lfw")
image = lfw / "images" / "George_W_Bush" / "George_W_Bush_0001.jpg"
```

The external cache path can be changed when generating metadata with
`--data-home PATH` (or `SKLEARN_DATA_HOME`). Re-run
`python3 scripts/prepare_lfw_dataset.py` only after auditing that existing
cache; the script never downloads LFW.

Do not commit the LFW image binaries, archives, sklearn joblib cache, or any
derived embeddings/results. Commit the lightweight metadata and this README.
