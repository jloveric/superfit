# Post-Submission Improvements

This note summarizes the main improvements after submission in a compact form, with one implementation pointer per item.

## 1) VarAxis Primitives

We use VarAxis variants (`VarAxisSF`, `VarAxisSQ`, `VarAxisSPP`, `VarAxisSG`) so axis choice is learned inside the same differentiable fitting loop instead of being hard-coded.

EMS context (one line): EMS fits superquadrics with an axis-aware strategy plus rule-based stages; we keep axis adaptation but optimize it end-to-end with gradients in the same objective.

Primary pointer: `PRIM_TYPE` family and ablations in `superfit/superfit/utils/config.py`.

## 2) Adaptive Stochastic Preconditioning

Stochastic preconditioning helps escape local minima and keep assemblies compact early, but too much noise harms thin-part fitting late. We therefore reduce the lower preconditioning level as reconstruction quality improves.

Primary pointer: `LOWER_SP`, `STOCHASTIC_PRECONDITION_INIT_VAL`, `STOCHASTIC_PRECONDITION_INIT_VAL_LOWER` in `superfit/superfit/utils/config.py`.

## 3) Curvature-Aware Weighting Beyond Surface Samples

We propagate curvature emphasis beyond direct surface samples so optimization is more sensitive in geometrically fragile regions (especially thin/high-curvature areas).

Primary pointer: `USE_CURVATURE_WEIGHTS` and `CURVATURE_WEIGHTS_SCALE` in `superfit/superfit/utils/config.py`.

## 4) Bidirectional Surface-Adjacent Sampling

In addition to target-surface perturbations, we periodically sample points from the current predicted surface and mix them in. This is slower, but reduces artifacts such as thin floaters.

Primary pointer: `BIDIR`, `BIDIR_SAMPLE_RATIO`, `RENEW_PTS_ITER` in `superfit/superfit/utils/config.py`.

## 5) SDF Loss on Surface-Adjacent Points

We add an L2 SDF term on adjacent points (not just occupancy). This provides extra geometric signal when occupancy alone is ambiguous, especially around thin sheets.

Primary pointer: `LOSS_SURFACE_ADJ_SDF_ALPHA` in `superfit/superfit/utils/config.py`.

## 6) Mesh Processing Improvements for Thin/Open Structures

Earlier preprocessing could lose thin-sheet structure. We strengthened the mesh-to-SDF pipeline using cleanup and a CD-triggered fallback path so difficult meshes are handled more robustly.

Primary pointer: `cd_based_process_mesh_to_sdf()` in `superfit/superfit/utils/mesh_preprocess.py`.

## 7) Hyperparameter Rebalancing for Better Primitive Quality

Increasing quality-focused regularizers improves part compactness/quality with minimal drop in reconstruction quality.

Primary pointer: quality-loss weights and `new_loss_lambda()` in `superfit/superfit/utils/config.py`.

## Extra: SuperGeon vs SuperFrustum

`SuperGeon` is more expressive than `SuperFrustum`, but that extra flexibility also increases optimization difficulty and local-minima risk. A promising next direction is intermediate primitives between SF and SG (for example, adding only one extra degree of freedom at a time).

Primary pointer: `superfit/notes/primitives.md`.

