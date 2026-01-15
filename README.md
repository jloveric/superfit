# SuperFit

Code Release for Resfit. 

To provide: 

1. All the different primitives: 
    (Cuboid, SQ, NeoPrim, SF, SFSolid)
    1. Torch Execution
    2. Shader Code. 

2. Different Decomposition Procedures: 
    1. All three (VHACD, COACD, MSD)

3. ResFit (Only for SF): 
    1. Decomposition + initialization.
    2. ResFit Optimization.
    3. ResFit Pruning. 
    4. Iterative Loop code.

4. Applications:
    1. Mesh to fit.  
    2. Fit Color To generate Textured Assets.
    3. Prim Assembly to Editing setup (support export (to Mesh / etc)
    4. Text to 3D to Primitive 
    5. Primitive to 3D Gen with Space Control.  
    6. Mesh to Solid Fitting. 

5. Eval Code.

# Ablations to try

1. Set 3 possible initializations via probabilistic init. with tied params. 
2. Set stochastic noise to loss function weighting Dirichlet random scalarization.
3. Adding noise to gradients. 
4. Sharpness aware minimization, parameter space noising.
# Simple: 
    1. Gradually lowering the noise 
    2. Bidir. Sampling. 
    3. 