# Primitive Notes

This document summarizes primitive families supported in SuperFit and where they are usable (optimization, shader export, editing view, batched torch ops).

## Support Matrix

| Primitive | VarAxis | Shader | Editing Shader | Torch | Batched Torch | Notes |
|---|---|---|---|---|---|---|
| Cuboid | No | Yes | Yes | Yes | Yes | Baseline primitive |
| SuperQuadric (SQ) | Yes | Inexact | No | Yes | Yes | Baseline primitive |
| SPProto (SuperPrimitivePrototype) | Yes | Yes | Yes | Yes | Yes | Base SuperPrimitive |
| SuperFrustum (SF) | Yes | Yes | Yes | Yes | Yes | Main paper primitive |
| SuperGeon (SG) | Yes | Yes | Yes | Yes | Yes | Geon-inspired extension of SF |
| Solid SuperFrustum (solidSF) | No | No | No | Yes | Yes | Primarily for generating CSG formula |

## Legacy Primitive Variants

These are earlier variants explored during the design of SuperFrustum. Symbolic definitions live in `superfit/symbolic/old_primitives.py` and shader implementations in `superfit/shader/old_primitives.py`.

| Variant | Symbolic | Shader | Description |
|---|---|---|---|
| SPTaperedOnion | Yes | -- | UberPrim-style (Paniq) with onion shell |
| SPTaperedWrongV1 | Yes | Yes | Early taper attempt, single-scale |
| SPTaperedWrongV2 | Yes | Yes | Dual-roundness / dual-scale taper |
| SPTaperedNewtonV1 | Yes | Yes | Newton-solver based taper |
| SPTaperedNewtonV2 | -- | Yes | Multi-step Newton taper (shader only) |
| SPTaperedApproxV1 | Yes | Yes | Approximate taper, single roundness/scale |
| SPTaperedApproxV2 | Yes | Yes | Approximate taper, dual roundness/scale |
| SPChamferedV1 | Yes | Yes | Chamfer-box style primitive |
| SPChamferedV2 | Yes | Yes | Chamfer with dual roundness/scale |
| SPTaperedQuarticV1 | Yes | Yes | Quartic-solver based taper |

None of these have batched torch compute or optimization support -- they are retained for shader visualization and historical reference.

## Notes on SuperGeon

SuperGeon is motivated by Biederman's Recognition-by-Components (RBC) theory (1987), which proposes that humans perceive objects by decomposing them into a small set of volumetric primitives called *geons* -- cylinders, cones, prisms, truncated cones, and their curved/tapered/bent variants. Biederman catalogued fewer than 36 such shapes, arguing they form a visual alphabet sufficient to compose virtually any everyday object.

SuperGeon extends SuperFrustum with three additional parameters -- `trapeze` (asymmetric cross-section), `taper_bulge` (profile-dependent scaling along the axis), and `rot2d` (in-plane rotation of the cross-section) -- bringing the total to 11 parameters. These additions allow SuperGeon to smoothly and differentiably represent nearly all of Biederman's geon catalogue, including triangular prisms, asymmetric tapered forms, and twisted profiles that SuperFrustum cannot reach. Critically, this is done within a single continuous SDF parameterization: different geon types are not discrete choices but smooth regions of the same parameter space, making SuperGeon fully optimizable with gradient descent.

## Notes on SuperFrustum

SuperFrustum is the primary primitive used in the paper. It is parameterized by 8 values: `size` (3), `roundness`, `dilate_3d`, `taper`, `bulge`, and `onion`. This compact parameterization spans cuboids, cylinders, spheres, cones, capsules, toroidal variants, and their tapered, bent, and hollow forms.

The construction builds on analytic SDF functions developed by the ShaderToy and demoscene communities -- specifically Paniq's [sdUberprim](https://www.shadertoy.com/view/MsVGWG) and related work by Inigo Quilez ([sdSuperprim](https://www.shadertoy.com/view/Xdy3Rm), [ChamferBox SP](https://www.shadertoy.com/view/3lBGzt)). SuperFit extends `sdUberprim` by introducing the "bulge" based bending. More importantly, we show that with some reparameterization, these super-primitives are suitable for inverse design: the SDF is C0-continuous and differentiable almost everywhere with respect to all parameters, enabling robust gradient-based fitting without non-differentiable heuristics.

The name "SuperFrustum" was coined by [Matheus Gadelha](http://mgadelha.me).

## Reconstruction Performance. 

Note for all 

| ablation | iou | bidir_iou | cd | n_prims | overlap | unoverlap | total_time | n_iters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cuboid | 0.8516 | 0.7810 | 0.2952 | 21.1695 | 0.2144 | 0.2201 | 620.4176 | 5.5377 |
| SuperQuadric | 0.8496 | 0.7980 | 0.9823 | 17.4268 | 0.2071 | 0.0199 | 533.5698 | 4.6590 |
| SPProto | 0.8746 | 0.8226 | 0.2745 | 20.9937 | 0.2406 | 0.0590 | 735.9650 | 5.4854 |
| SuperFrustum | 0.8868 | 0.8373 | 0.2204 | 20.3556 | 0.2499 | 0.0438 | 1118.7865 | 5.4644 |
| SuperGeon | 0.8876 | 0.8392 | 0.2347 | 19.4184 | 0.2540 | 0.0300 | 1130.2016 | 5.2720 |


## Attribution

The primitive work is strongly inspired by the procedural and SDF modeling community, including Inigo Quilez and Paniq. SuperFit builds on these ideas for optimization-driven fitting and composition.

