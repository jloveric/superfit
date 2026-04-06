"""
ADOBE

Copyright 2026 Adobe

All Rights Reserved.

NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.

Utility functions for material texture optimization.
"""
import trimesh
import cubvh
import torch as th
import numpy as np
import geolipi.symbolic as gls
import sysl.symbolic as sls
from geolipi.symbolic.symbol_types import PRIM_TYPE
from sysl.shader import evaluate_to_shader
from sysl.shader_runtime import create_multibuffer_shader_html
from sysl.shader.utils.texture import convert_to_atlased_encoded
from ..symbolic.utils import fetch_singular_expr_eval
from sysl.utils import recursive_sm_to_smg
from .color_utils import srgb_to_linear

GRID_SIZE = (128, 128)
def initialize_texture_grid(device="cuda", grid_size=GRID_SIZE, init_value=0.5):
    """
    Initialize a texture grid tensor for SphericalRGBGrid3D.
    
    Args:
        device: Device to create tensor on
        grid_size: (H, W) size of the texture grid
        init_value: Initial value for RGB channels (0-1 range)
    
    Returns:
        Tensor of shape (H, W, 3) for RGB channels
    """
    h, w = grid_size
    # Initialize with neutral gray color
    texture = th.rand((h, w, 3), device=device, dtype=th.float32) * init_value
    return texture


def recursive_add_spherical_tex(gls_expr, ind=0, version="v4", device="cuda", grid_size=GRID_SIZE):
    """
    Recursively convert geometric expression to material expression with textures.
    
    Converts geometric primitives to material expressions with SphericalRGBGrid3D textures.
    
    Args:
        gls_expr: Geometric expression (GLBase)
        ind: Index counter for texture naming
        version: Version string (currently only "v4" supported)
        device: Device for tensor creation
        grid_size: Size of texture grid (H, W)
    
    Returns:
        Tuple of (material_expression, updated_index)
    """
    if isinstance(gls_expr, gls.GLBase):
        if isinstance(gls_expr, PRIM_TYPE):
            if version == "v4":
                # Initialize texture grid with neutral gray
                texture_tensor = initialize_texture_grid(device=device, grid_size=grid_size, init_value=0.5)
                texture_tensor = texture_tensor.requires_grad_(True)
                
                new_expr = sls.MatSolidV4(
                    gls_expr, 
                    sls.SphericalRGBGrid3D(
                        texture_tensor, 
                        f'texture_demo_{ind}', 
                        (0.1,), 
                        (0.5,)
                    )
                )
            else:
                raise ValueError(f"Invalid version: {version}")
            ind += 1
            return new_expr, ind
        else:
            new_args = []
            for i, arg in enumerate(gls_expr.args):
                if isinstance(arg, gls.GLBase):
                    out_expr, ind = recursive_add_spherical_tex(arg, ind, version=version, device=device, grid_size=grid_size)
                    new_args.append(out_expr)
                else:
                    new_arg = gls_expr.get_arg(i)
                    new_args.append(new_arg)
            return gls_expr.__class__(*new_args), ind
    else:
        return gls_expr, ind


def reconf_sp_rgb(gls_expr):
    """
    Reconfigure SphericalRGBGrid3D textures after optimization.
    
    Converts optimized texture values from float [0,1] to uint8 [0,255] format.
    
    Args:
        gls_expr: Expression containing SphericalRGBGrid3D nodes
    
    Returns:
        Expression with reconfigured textures
    """
    if isinstance(gls_expr, gls.GLBase):
        if isinstance(gls_expr, sls.SphericalRGBGrid3D):
            args = [gls_expr.get_arg(i) for i in range(len(gls_expr.args))]
            opt_color = args[0]
            opt_color = opt_color.reshape(-1, 3)
            opt_color = th.clamp(opt_color, 0.0, 1.0)
            colors = (opt_color * 255).to(th.uint8)
            colors = colors.T
            args[0] = colors
            new_expr = gls_expr.__class__(*args)
            return new_expr
        else:
            new_args = []
            for i, arg in enumerate(gls_expr.args):
                if isinstance(arg, gls.GLBase):
                    out_expr = reconf_sp_rgb(arg)
                    new_args.append(out_expr)
                else:
                    new_arg = gls_expr.get_arg(i)
                    new_args.append(new_arg)
            return gls_expr.__class__(*new_args)
    else:
        return gls_expr

def query_materials_from_surface_cubvh(
    mesh: trimesh.Trimesh,
    query_pts: th.Tensor,  # (3, N), float32
    device: str = "cuda"
):
    """
    Use cubvh to find nearest mesh surface point for each query, and fetch RGB + MR from UV textures.
    """
    query_pts = query_pts.contiguous()  # (N, 3)
    verts = th.from_numpy(mesh.vertices).float().to(device)
    faces = th.from_numpy(mesh.faces).int().to(device)
    bvh = cubvh.cuBVH(verts, faces)

    # Find closest point and interpolate UVs
    dists, face_idx, uvw = bvh.unsigned_distance(query_pts, return_uvw=True)  # (N,), (N,), (N, 3)
    uv = th.from_numpy(mesh.visual.uv).float().to(device)          # (V, 2)
    face_uv = uv[faces[face_idx]]                                  # (N, 3, 2)
    sampled_uv = (uvw[..., None] * face_uv).sum(dim=1)             # (N, 2)

    # Load texture maps
    base_img = mesh.visual.material._data['baseColorTexture'].convert("RGB")
    try:
        mr_img = mesh.visual.material._data['metallicRoughnessTexture'].convert("RGB")
    except:
        mr_img = np.ones_like(base_img)
    base_np = np.array(base_img) / 255.0
    mr_np = np.array(mr_img) / 255.0
    if isinstance(mesh.visual.material.baseColorFactor, list):
        factor = np.array(mesh.visual.material.baseColorFactor)
    else:
        factor = np.array([1.0, 1.0, 1.0])
    def uv_to_texel(uvs, image_shape):
        h, w = image_shape
        px = (uvs[:, 0] * w).long().clamp(0, w - 1)
        py = ((1.0 - uvs[:, 1]) * h).long().clamp(0, h - 1)
        return px, py

    sampled_uv = sampled_uv.clamp(0.0, 1.0)
    px_base, py_base = uv_to_texel(sampled_uv, base_np.shape[:2])
    px_mr, py_mr     = uv_to_texel(sampled_uv, mr_np.shape[:2])

    # Convert to tensors and sample
    base_tex = th.from_numpy(base_np).float().to(device)  # (H, W, 4)
    base_tex *= th.tensor(factor[:3], device=device)
    base_tex = srgb_to_linear(base_tex)
    mr_tex = th.from_numpy(mr_np).float().to(device)      # (H, W, 3)

    rgba = base_tex[py_base, px_base]  # (N, 4)
    rgb = rgba[:, :3]
    mr = mr_tex[py_mr, px_mr][:, :2]   # (N, 2)

    return rgb, mr

def get_material_expr(init_expr):
    sampled_expr = fetch_singular_expr_eval(init_expr.sympy(), relaxed_eval=False, remove_marker=False)
    # new_expr = recursive_sm_to_smg(sampled_expr.sympy())
    new_expr = sampled_expr
    new_expr, _ = recursive_add_spherical_tex(new_expr.sympy(), 2, version="v4")
    return new_expr


def save_html_mat_expr(mat_expr, sketcher_3d, save_file_name="matexpr.html"):
    mat_expr_reconf = reconf_sp_rgb(mat_expr)
    output_expr = convert_to_atlased_encoded(mat_expr_reconf.tensor(), sketcher_3d)
    final_expression = fetch_singular_expr_eval(output_expr.sympy(), relaxed_eval=False, remove_marker=True)
    shader_info = evaluate_to_shader(final_expression, mode="multipass", post_process_shader=["part_outline_nobg"])
    html_code = create_multibuffer_shader_html(shader_info)
    with open(save_file_name, "w") as f:
        f.write(html_code)
    return html_code

