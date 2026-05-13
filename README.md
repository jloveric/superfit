# SuperFit: Residual Primitive Fitting with SuperFrusta

<p align="center">
  <a href="https://arxiv.org/abs/2512.09201">
    <img src="https://img.shields.io/badge/arXiv-2512.09201-b31b1b?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://bardofcodes.github.io/superfit">
    <img src="https://img.shields.io/badge/Project%20Page-Online-brightgreen?logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
</p>

<p align="center">
  <img src="assets/banner.jpeg" alt="Banner" />
</p>


SuperFit fits compact assemblies of **SuperFrusta** and other primitives to 3D shapes, built on top of **[SySL](https://github.com/BardOfCodes/sysl)**.
See **[Install](#install-instructions)** below and **[BibTeX](#bibtex)**.

## Install Instructions

### 1. Create the conda environment

```bash
git clone https://github.com/BardOfCodes/superfit.git
cd superfit
conda env create -f env.yml
conda activate superfit
```

This installs PyTorch 2.9.1 (CUDA 12.8), all core Python dependencies, and `superfit` itself in editable mode.

### 2. Install cubvh

[cubvh](https://github.com/ashawkey/cubvh) provides GPU-accelerated BVH queries and is required by the fitting pipeline.

```bash
git clone https://github.com/ashawkey/cubvh.git
cd cubvh
python setup.py install
cd ..
```

### 3. Install kaolin

[Kaolin](https://github.com/NVIDIAGameWorks/kaolin) is used for FlexiCubes meshing. See the [kaolin installation docs](https://kaolin.readthedocs.io/en/latest/notes/installation.html) for full details.

```bash
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt
export IGNORE_TORCH_VER=1
python setup.py install
cd ..
```

> **Note:** `IGNORE_TORCH_VER=1` is needed because kaolin's version check may not yet list PyTorch 2.9.

### 4. Path Configuration

Before running any scripts, edit `superfit/utils/constants.py` and set the three base paths for your machine:

```python
DATA_BASE = "/path/to/your/data"
PROJECT_BASE = "/path/to/your/projects/project_sf"
OUTPUTS_BASE = "/path/to/your/outputs"
```

All dataset and artifact locations are derived from these. See [notes/dataset.md](notes/dataset.md) for details on expected data layout.

## What can you do with this repository?

### 1. Convert (watertight) meshes into Primitive Assemblies

```bash
python scripts/mesh_to_assembly.py --input_path <path> --save_dir <save-dir> --fastmode --save_html --save_edit_html --save_mesh
```

This will convert an input image into a compact assembly of SuperFrusta. Use different `--ablation` options to generate assemblies of cuboids/superquadrics/supergeons etc. Note that `--fastmode` saves torch compile artifacts at `AOT_ARTIFACT_DIR` as specified in `superfit/utils/constants.py`.

<p align="center">
  <img src="assets/mesh_to_assembly.jpg" alt="Mesh to primitive assembly" style="max-width: 512px; width: 100%;" />
</p>

If the input mesh contains textures, you can run `fit_textures.py` to add textures to the primitive assembly. This will add 2D spherical textures to each primitive. We also have the `testset_fit_textures.py` for running this process across multiple inputs. 

```bash
python scripts/fit_texture.py --input_path <path-to-assembly-pkl> --save_html --save_edit_html
```

<p align="center">
  <img src="assets/textured.jpeg" alt="Texture fitting" />
</p>


Finally, you can also generate videos of the optimization process: 

```bash
python scripts/generate_opt_video.py --input_path <path-to-assembly-pkl> --save_dir <save-dir> 
```

<p align="center">
  <img src="assets/opt_video.gif" alt="Optimization video" />
</p>


Note that we don't generate html shaders for SuperQuadric since we don't have analytical sphere-tracable SDF functions for them. Additionally, please change the configuration in `superfit/utils/config.py` and `superfit/utils/render_seq.py` if needed.

### 2. Evaluation on TestSet

1. Fit primitives to toys5k

```bash
python scripts/testset_fit_primitives.py --start_ind 0 --end_ind 500 --fastmode --save_dir <save-path>
```

This will generate primitive assemblies for our Toy4k evaluation subset. You can use additionally use `--dataset partobjaverse` to generate the fits for PartObjaverse dataset. Finally, [`job_scripts/all_toy4k.sh`](job_scripts/all_toy4k.sh) shows how to run the process in parallel across gpus if need be.

2. Run Evaluation

Once the primitives are generated, you can run: 

```bash
python scripts/testset_eval.py --input_path <save-path> --save_per_instance_metrics --start_ind 0 --end_ind 500 --include_semantic
```

For the semantic metrics, we require [faiss](https://github.com/facebookresearch/faiss) as well as [PartField](https://github.com/nv-tlabs/PartField). Add `--include_semantic` to evaluate the semantic metrics.

### 3. Explore Fitting Results & Generate Detailed Meshes

| Assembly Visualizer | Primitive-guided Generation |
| --- | --- |
| <p align="center"><img src="assets/app_visualizer.png" alt="Assembly Visualizer" /></p> | <p align="center"><img src="assets/app_edit_mode.png" alt="Primitive-guided Generation" /></p> |


We also provide a complimentary web app to explore the primitive assemblies - the fitting process, the metrics, the generated shapes etc. 


The inferred primitive assembly can also be used as spatial guidance to generate higher fidelity meshes by combining our method with [SpaceControl](https://spacecontrol3d.github.io), by E. Fedele et al.


Find instructions to install and run this at [superfit_app](https://github.com/BardOfCodes/superfit_app).

## Additional Details

Details regarding the dataset are provided in [notes/dataset.md](notes/dataset.md). For details regarding the primitives check out [notes/primitives.md](notes/primitives.md). We also made quite a few improvements over the results after our CVPR submission. These are listed in [notes/post_submission.md](notes/post_submission.md).

The [`notebooks/`](notebooks/) folder contains a few iPython Notebooks which demo different aspects of our method such as (a) primitive design exploration in [notebooks/primitive_design.ipynb](notebooks/primitive_design.ipynb), (b) curvature exploration in [notebooks/curvature.ipynb](notebooks/curvature.ipynb), and (c) morphological decomposition in [notebooks/msd.ipynb](notebooks/msd.ipynb). 


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

This project was developed during an internship at Adobe Research. I (Aditya) am grateful to all co-authors for their invaluable contributions: [Matheus Gadelha](https://mgadelha.me/), [Thibault Groueix](https://www.tgroueix.com/), [Zhiqin Chen](https://czq142857.github.io/), [Siddhartha Chaudhuri](https://www.cse.iitb.ac.in/~sidch/), [Vladimir G. Kim](http://www.vovakim.com/), [Wang Yifan](https://yifita.github.io/), and [Daniel Ritchie](https://dritchie.github.io/).

Our sphere-tracing primitives and shaders draw heavily on foundational work by Inigo Quilez, and the broader [ShaderToy](https://www.shadertoy.com/) community — with special thanks to Paniq for creating the `superprimitive` and `uberprimitive` functions. 

Finally, Anton Mikhailov, Daichi Ito, and Luc Chamerlat from Adobe provided valuable feedback aroud primitive design, and artist needs.

For questions reach out at `adityaganeshan@gmail.com`.
