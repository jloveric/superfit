# Dataset & Evaluation Setup

## 1. Download Datasets

Download the [Toys4k](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k) dataset (Stojanov et al., CVPR 2021).
It contains 4,000 toy-object meshes across 105 categories.
Once downloaded, set `TOY4K_PATH_PREFIX` in `superfit/utils/constants.py` to point to the root directory (the folder containing the per-category subdirectories, e.g., `truck/`, `chair/`, etc.).

Download the [PartObjaverse-Tiny](https://github.com/Pointcept/SAMPart3D/blob/main/PartObjaverse-Tiny/PartObjaverse-Tiny.md) dataset by following instructions on [this page](https://github.com/Pointcept/SAMPart3D/blob/main/PartObjaverse-Tiny/PartObjaverse-Tiny.md). Set `PARTOBJAVERSE_BASE`, `PARTOBJAVERSE_MESH_DIR`, `PARTOBJAVERSE_INSTANCE_DIR` in `superfit/utils/constants.py` accordingly.

## 2. Evaluation split

We release `dataset/new_testset.csv` with the repository.
This file lists shapes from Toys4k ordered by **farthest-point sampling (FPS)** using **Chamfer Distance (CD)** between shapes, seeded from a randomly chosen starting shape. The first *N* entries (e.g., the first 500) therefore form a maximally diverse evaluation subset.

## 3. Qualitative split

We also release `dataset/qual_testset.csv`, a small hand-selected set of shapes chosen for visual quality and diversity. Additionally, `dataset/select_superfrustum_toys4k.csv` and `dataset/select_superfrustum_partobjaverse.csv` contain some hand-selected samples where the inferred assemblies appear to have good quality. 

## 4. Released primitive assemblies

We release pre-computed primitive-assembly results so that users can inspect outputs, run evaluation, or render visualizations without re-fitting. Please download them from huggingface [here]().

1. SuperFrustum fitting for toys4k. 

| ablation | iou | bidir_iou | cd | n_prims | overlap | unoverlap | total_time | n_iters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Toys4k | 0.8967 | 0.8494 | 0.1931 | 15.7253 | 0.2560 | 0.0392 | 802.4408 | 4.4640 |


2. Toys4k 500 subset fitting with different primitives. Check `primitives.md` for their evaluation.

Root: `/users/aganesh8/data/aganesh8/data/project_sf/sf_release/toys4k`

| Type | Folder |
|------|--------|
| Cuboid | `cuboid/` |
| SuperQuadric (SQ) | `superquadric/` |
| SPProto (SPP) | `sp_proto/` |
| SuperFrustum (SF) | `superfrustum/` |
| SuperGeon (SG) | `supergeon/` |
| SF — paper (CVPR) | `sf_cvpr/` |


3. PartObjaverse fitting with superfrustum along with texture fitting. 

| ablation | iou | bidir_iou | cd | n_prims | overlap | unoverlap | total_time | n_iters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PartObjaverse | 0.8983 | 0.8183 | 0.1345 | 25.1050 | 0.2530 | 0.0343 | 1270.9879 | 5.9750 |

