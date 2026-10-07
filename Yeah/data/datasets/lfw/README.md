# Labeled Faces in the Wild (LFW)

This entry describes the standard **lfw-funneled** variant: 13,233 JPEG images
across 5,749 identities. Its canonical runtime directory is `images/` here.

The images are not committed because their upstream redistribution rights are
not recorded for this repository. After reviewing the original UMass LFW terms,
install them from the upstream scikit-learn fetcher without using any global
cache:

```bash
python3 scripts/setup_research_assets.py --accept-upstream-research-terms
```

`manifest.csv` is the image inventory; `pairs/` contains official pair
definitions; `evaluation_splits/` contains the frozen experiment splits.
`prepare_lfw_dataset.py` only creates metadata and never downloads LFW.

Do not commit raw images, archives, sklearn caches, or derived embeddings
without a documented redistribution approval.
