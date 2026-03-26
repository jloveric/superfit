# Dataset & Evaluation Setup

## 1. Install Toys4k

Download the [Toys4k](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k) dataset (Stojanov et al., CVPR 2021).
It contains 4,179 toy-object meshes across 105 categories.
Once downloaded, set `TOY4K_PATH_PREFIX` in `superfit/utils/constants.py` to point to the root directory (the folder containing the per-category subdirectories, e.g., `truck/`, `chair/`, etc.).

## 2. Evaluation split

We release `dataset/test_set_1.csv` with the repository.
This file lists shapes from Toys4k ordered by **farthest-point sampling (FPS)** using **Chamfer Distance (CD)** between shapes, seeded from a randomly chosen starting shape. The first *N* entries (e.g., the first 500) therefore form a maximally diverse evaluation subset.

Format: `category,<absolute_path_to_mesh>` — update the path prefix to match your local Toys4k install before use, or rely on the relative-path version referenced by `TOY4K_CSV_FILE` in `constants.py`.

## 3. Qualitative split

We also release `dataset/qual_testset.csv`, a small hand-selected set of shapes chosen for visual quality and diversity.
These are useful for generating figures and qualitative comparisons.

## 4. Released primitive assemblies

We release pre-computed primitive-assembly results so that users can inspect outputs, run evaluation, or render visualizations without re-fitting.

### Primitive-type comparison (test set)

Fitting results on the evaluation split using each supported primitive type under matched settings:

| Primitive | Ablation | Type key |
|-----------|----------|----------|
| Cuboid | 34 | `Cuboid` |
| SuperQuadric (SQ) | 35 | `VarAxisSQ` |
| SPProto (SPP) | 32 | `VarAxisSPP` |
| SuperFrustum (SF) | 31 | `VarAxisSF` |
| SuperGeon (SG) | 33 | `VarAxisSG` |
| SF — paper setting | 0 (default) | `VarAxisSF` + post-prune + bidir |

### Full-dataset runs

| Run | Dataset | Description |
|-----|---------|-------------|
| SQ on all Toys4k | Toys4k (4k shapes) | SuperQuadric fitting across the entire dataset. |
| Non-smooth SF on all Toys4k | Toys4k (4k shapes) | SuperFrustum fitting without the smoothing step. |

### Cross-dataset chair fitting

SuperFrustum (SF) fitting on **chair** meshes drawn from three sources:

- **Toys4k** — chair category
- **3DCoMPaT++** — chair category ([Li et al., 2024](https://3dcompat-dataset.org/v2/))
- **PartNet** — chair category ([Mo et al., CVPR 2019](https://partnet.cs.stanford.edu/))
