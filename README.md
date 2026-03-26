# SuperFit

Paper / project links.
Banner img.
# TODO: 
1. Release Models: 
  a. Toy4k 6 versions (Cuboid/SQ/SPP/SF/SG/Orig/orig_mesh)
  b. SF -> Toy4k Qual (or all?).
  c. SF -> PartObjaverse (with textures).
  d. ChairAssembly6k | No Smooth

2. Test and make sure things are working. Particularly around the following: 
First create an ablation which has just 10 iterations of optimization for fast checking. 

a. Using different prune metrics. 
b. decompose mode (MSD / COACD / VHACD) 
c. Optimizer Adamw. 
d. LOWER_SP-True / False/ 
e. DO_Prune True / False.
f. Primtype -> cuboid / sq / varaxis sq/ sf/ varaxis sf / sg / varaxis sg/ solidsf / spp / varaxisspp.
e. Smoothen is off. 
g. TARGET_MODE bboxed vs dilated
h. SAVE_JIT_CACHE False / True. 
i. TorchCompile True False. 
j. USE_CURVATURE_WEIGHTS 
h. Stochastic dropout testing
i. Morph loss, Tversky loss.

3. Readme Cleaning. 
   => What this is / overview / banner / Link to paper / project page. License (ADOBE LICENSE)
   => Instructions for running mesh-to-pa / texturing / editing / opt videos. Help ablation - fit cuboids /varaxissq / varaxissf/ varaxissg. Mesh-to 
   => Eval results replication. 
   => EVAL data link -> datasets.md. 
   => Primitives discussion -> link to primitives. 
   => Acknowledgements (Adobe team) Inigo / Anton Mikalove / Paniq / 

## Install Instructions

Clone and create the conda environment:

```bash
git clone <repo-url>
cd superfit
conda env create -f env.yml
conda activate superfit
```

### Path Configuration

Before running any scripts, edit `superfit/utils/constants.py` and set the three base paths for your machine:

```python
DATA_BASE = "/path/to/your/data"
PROJECT_BASE = "/path/to/your/projects/project_neo"
OUTPUTS_BASE = "/path/to/your/outputs"
```

All dataset and artifact locations are derived from these. See [notes/dataset.md](notes/dataset.md) for details on expected data layout.

### Additional Dependencies

The following packages are not on PyPI and must be installed separately per their own instructions:

- [geolipi](https://github.com/bardofcodes/geolipi)
- [sysl](https://github.com/bardofcodes/sysl)
- [cubvh](https://github.com/ashawkey/cubvh)
- [kaolin](https://github.com/NVIDIAGameWorks/kaolin) (optional, for some notebooks)

## What You Can Do With It

### Fit a single mesh to a primitive assembly

```bash
python scripts/mesh_to_pa.py \
  --input_file path/to/mesh.obj \
  --save_dir path/to/output \
  --fastmode --ablation 0
```

Add `--save_html` or `--save_edit_html` to export interactive viewer files alongside the `.pkl` output.

### Fit primitives over a dataset split

```bash
python scripts/testset_fit_primitives.py \
  --dataset toys4k \
  --start_ind 0 --end_ind 100 \
  --ablation 0 --fastmode
```

For PartObjaverse part-wise fitting:

```bash
python scripts/testset_fit_partwise.py \
  --start_ind 0 --end_ind 50 \
  --ablation 0 --fastmode
```

### Fit textures on existing outputs

```bash
python scripts/testset_fit_textures.py \
  --input_dir path/to/outputs/toys4k/ablation_0_v6 \
  --ablation 0 --save_html
```

### Generate optimization videos

```bash
python scripts/testset_opt_video.py \
  --input_dir path/to/outputs/toys4k/ablation_0_v6 \
  --ablation 0
```

To render specific shapes only: `--folders truck_028 apple_003`.

## Eval

### Matching paper settings

The paper results use `ablation 0` with `--fastmode` on the Toys4k test split. Config presets are applied through `set_config_ablation()` in `superfit/utils/config.py`.

### Running evaluation on fitted outputs

```bash
python scripts/testset_eval.py \
  --input_dir path/to/outputs/toys4k/ablation_0_v6 \
  --eval last
```

This produces a summary `.pkl` and a markdown table in the input directory. Add `--save_per_instance_metrics` for per-shape eval files. Add `--include_semantic` if semantic annotations are available.

### Multi-GPU batch runs

Use the job script to distribute fitting across GPUs:

```bash
bash job_scripts/all_toy4k.sh 0 500 4 0 aott
# args: start_ind end_ind num_gpus [ablation] [aot_postfix]
```

Edit `LOG_DIR`, `SCRIPT_DIR`, and conda env name at the top of the script for your cluster.

## Other Resources

1. [Primitive notes](notes/primitives.md) -- supported primitive types and their capabilities.
2. [Dataset notes](notes/dataset.md) -- data layout, split format, and PartObjaverse usage.
3. Exploratory notebooks in `notebooks/` covering primitives, pruning, timing, mesh curvature, semantic loss, texture, and paper figure generation.
4. Space Control based editing -- see the [SpaceControl repo](https://github.com/bardofcodes/spacecontrol).

## Updated Version

Improvements over the initial release:

1. VarAxis primitives (`VarAxisSF`, `VarAxisSG`, `VarAxisSPP`, `VarAxisSQ`)
2. Bi-directional sampling
3. SDF-on-points loss
4. Curvature-aware relaxation and smoothening
5. Semantic / macro loss support

## BibTeX

```bibtex
@misc{ganeshan2026superfit,
  title         = {Residual Primitive Fitting of 3D Shapes with SuperFrusta},
  author        = {Aditya Ganeshan and Matheus Gadelha and Thibault Groueix and Zhiqin Chen and Siddhartha Chaudhuri and Vladimir G. Kim and Wang Yifan and Daniel Ritchie},
  year          = {2026},
  booktitle     = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month         = {June},
}
```

## Acknowledgements

This work was done during an internship at Adobe Research.
Thanks to all co-authors -- [Matheus Gadelha](https://mgadelha.me/), [Thibault Groueix](https://www.tgroueix.com/), [Zhiqin Chen](https://czq142857.github.io/), [Siddhartha Chaudhuri](https://www.cse.iitb.ac.in/~sidch/), [Vladimir G. Kim](http://www.vovakim.com/), [Wang Yifan](https://yifita.github.io/), and [Daniel Ritchie](https://dritchie.github.io/) -- and to Inigo Quilez, Anton Mikhailov, Luc Chamerlat at Adobe, and the ShaderToy community, particularly Paniq, for foundational primitive functions.

For questions reach out at adityaganeshan@gmail.com.
