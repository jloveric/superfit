import torch as th
import geolipi.symbolic as gls
from geolipi.torch_compute.transforms import axis_angle_to_rotation_matrix


def convert_axis_angle_to_euler(axis_angle: th.Tensor) -> th.Tensor:
    """
    Map axis-angle ``(axis * angle)`` (3-vector) to Euler angles for ``EulerRotate3D``.

    Evaluator composes rotations as in `get_affine_rotate_euler_3D`:
      R = R_x(θ_0) @ R_y(θ_1) @ R_z(θ_2)
    """
    aa = axis_angle
    if aa.dim() == 2 and aa.shape == (3, 1):
        aa = aa.squeeze(-1)
    if aa.shape[-1] != 3:
        raise ValueError(
            f"convert_axis_angle_to_euler: need last dim 3, got shape {tuple(aa.shape)}"
        )

    lead = aa.shape[:-1]
    aa_b = aa.reshape(-1, 3)
    R = axis_angle_to_rotation_matrix(aa_b)  # (N, 3, 3)

    sb = R[..., 0, 2].clamp(-1.0, 1.0)
    e_y = th.asin(sb)
    e_x = th.atan2(-R[..., 1, 2], R[..., 2, 2])
    e_z = th.atan2(-R[..., 0, 1], R[..., 0, 0])
    euler = th.stack([e_x, e_y, e_z], dim=-1)
    return euler.reshape(*lead, 3)


def recursive_axisangle_to_eulerangle(gls_expr):
    """
    Walk the expression tree and replace each ``gls.AxisAngleRotate3D`` with
    ``gls.EulerRotate3D``.
    """
    if isinstance(gls_expr, gls.AxisAngleRotate3D):
        args = gls_expr.get_args()
        if len(args) < 2:
            return gls_expr
        child = args[0]
        if isinstance(child, gls.GLBase):
            child = recursive_axisangle_to_eulerangle(child)
        rot_aa = gls_expr.tensor().get_arg(1)
        euler = convert_axis_angle_to_euler(rot_aa)
        return gls.EulerRotate3D(child, euler).sympy()

    if isinstance(gls_expr, gls.GLFunction):
        in_args = gls_expr.get_args()
        new_args = []
        for arg in in_args:
            if isinstance(arg, gls.GLBase):
                new_args.append(recursive_axisangle_to_eulerangle(arg))
            else:
                new_args.append(arg)
        return gls_expr.__class__(*new_args)

    return gls_expr


def _rotation_matrix_to_euler_xyz(R: th.Tensor) -> th.Tensor:
    # Inverse for R = Rx(e0) @ Ry(e1) @ Rz(e2) (same convention as convert_axis_angle_to_euler).
    sb = R[..., 0, 2].clamp(-1.0, 1.0)
    e_y = th.asin(sb)
    e_x = th.atan2(-R[..., 1, 2], R[..., 2, 2])
    e_z = th.atan2(-R[..., 0, 1], R[..., 0, 0])
    return th.stack([e_x, e_y, e_z], dim=-1)


def _euler_xyz_to_rotation_matrix(euler: th.Tensor) -> th.Tensor:
    """
    Convert Euler angles (X, Y, Z) into a rotation matrix using the same convention
    as `geolipi.torch_compute.transforms.get_affine_rotate_euler_3D`:
      R = Rx(euler[..., 0]) @ Ry(euler[..., 1]) @ Rz(euler[..., 2])
    """
    if euler.shape[-1] != 3:
        raise ValueError(
            f"_euler_xyz_to_rotation_matrix: expected last dim 3, got {tuple(euler.shape)}"
        )

    ex = euler[..., 0]
    ey = euler[..., 1]
    ez = euler[..., 2]

    cx, sx = th.cos(ex), th.sin(ex)
    cy, sy = th.cos(ey), th.sin(ey)
    cz, sz = th.cos(ez), th.sin(ez)

    # Rx
    Rx = th.stack(
        [
            th.stack([th.ones_like(cx), th.zeros_like(cx), th.zeros_like(cx)], dim=-1),
            th.stack([th.zeros_like(cx), cx, -sx], dim=-1),
            th.stack([th.zeros_like(cx), sx, cx], dim=-1),
        ],
        dim=-2,
    )

    # Ry
    Ry = th.stack(
        [
            th.stack([cy, th.zeros_like(cy), sy], dim=-1),
            th.stack([th.zeros_like(cy), th.ones_like(cy), th.zeros_like(cy)], dim=-1),
            th.stack([-sy, th.zeros_like(cy), cy], dim=-1),
        ],
        dim=-2,
    )

    # Rz
    Rz = th.stack(
        [
            th.stack([cz, -sz, th.zeros_like(cz)], dim=-1),
            th.stack([sz, cz, th.zeros_like(cz)], dim=-1),
            th.stack([th.zeros_like(cz), th.zeros_like(cz), th.ones_like(cz)], dim=-1),
        ],
        dim=-2,
    )

    return Rx @ Ry @ Rz


def rotation_matrix_to_axis_angle(R: th.Tensor, variant: str = "default", eps: float = 1e-6) -> th.Tensor:
    """
    Known-good conversion: rotation matrix -> axis-angle vector.

    Copied from `superfit/superfit/algos/estimate_init_params.py`.
    Supports batched rotation matrices with shape (..., 3, 3).
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"rotation_matrix_to_axis_angle: expected (...,3,3), got {tuple(R.shape)}")

    def _single(R_single: th.Tensor) -> th.Tensor:
        device, dtype = R_single.device, R_single.dtype
        trace = R_single.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        cos_theta = th.clamp((trace - 1) / 2, -1.0, 1.0)
        theta = th.acos(cos_theta)

        if th.isclose(theta, th.tensor(0.0, device=device, dtype=dtype), atol=eps):
            return th.tensor([eps, 0.0, 0.0], device=device, dtype=dtype)

        if th.isclose(theta, th.tensor(th.pi, device=device, dtype=dtype), atol=eps):
            R_plus = (R_single + th.eye(3, device=device, dtype=dtype)) / 2
            axis = th.sqrt(th.clamp(R_plus.diagonal(), min=0))
            if axis.norm() < eps:
                axis = th.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
            axis = axis / axis.norm()
            return theta * axis

        skew = (R_single - R_single.transpose(-1, -2)) / (2 * th.sin(theta))
        axis = th.stack([skew[2, 1], skew[0, 2], skew[1, 0]])

        if variant == "flip":
            axis = -axis
        elif variant == "xzy":
            axis = axis[[0, 2, 1]]
        elif variant == "yzx":
            axis = axis[[1, 2, 0]]
        elif variant == "zxy":
            axis = axis[[2, 0, 1]]
        elif variant == "neg_x":
            axis = axis * th.tensor([-1.0, 1.0, 1.0], device=axis.device)
        elif variant == "neg_y":
            axis = axis * th.tensor([1.0, -1.0, 1.0], device=axis.device)
        elif variant == "neg_z":
            axis = axis * th.tensor([1.0, 1.0, -1.0], device=axis.device)
        elif variant != "default":
            raise ValueError(f"rotation_matrix_to_axis_angle: unknown variant '{variant}'.")

        return theta * axis

    if R.dim() == 2:
        return _single(R)

    flat = R.reshape(-1, 3, 3)
    outs = [_single(flat[i]) for i in range(flat.shape[0])]
    out = th.stack(outs, dim=0)
    return out.reshape(R.shape[:-2] + (3,))


def recursive_eulerangle_to_axisangleangle(gls_expr):
    """
    Walk the expression tree and replace each ``gls.EulerRotate3D`` with
    ``gls.AxisAngleRotate3D`` (reverse of `recursive_axisangle_to_eulerangle`).
    """
    if isinstance(gls_expr, gls.EulerRotate3D):
        args = gls_expr.get_args()
        if len(args) < 2:
            return gls_expr

        child = args[0]
        if isinstance(child, gls.GLBase):
            child = recursive_eulerangle_to_axisangleangle(child)

        angles = gls_expr.tensor().get_arg(1)

        if angles.dim() == 2 and angles.shape == (3, 1):
            angles = angles.squeeze(-1)
        if angles.dim() == 2 and angles.shape[0] == 1 and angles.shape[-1] == 3:
            angles = angles.squeeze(0)

        if angles.shape[-1] != 3:
            raise ValueError(
                "recursive_eulerangle_to_axisangleangle: expected angles last dim 3, "
                f"got shape {tuple(angles.shape)}"
            )

        lead = angles.shape[:-1]
        angles_b = angles.reshape(-1, 3)
        R = _euler_xyz_to_rotation_matrix(angles_b)
        axis_angle_b = rotation_matrix_to_axis_angle(R)
        axis_angle = axis_angle_b.reshape(*lead, 3)
        return gls.AxisAngleRotate3D(child, axis_angle).sympy()

    if isinstance(gls_expr, gls.GLFunction):
        new_args = []
        for arg in gls_expr.get_args():
            if isinstance(arg, gls.GLBase):
                new_args.append(recursive_eulerangle_to_axisangleangle(arg))
            else:
                new_args.append(arg)
        return gls_expr.__class__(*new_args)

    return gls_expr


def recursive_euler_angle_to_axisangle(gls_expr):
    return recursive_eulerangle_to_axisangleangle(gls_expr)


def recursive_eulerangle_to_axisangle(gls_expr):
    return recursive_euler_angle_to_axisangle(gls_expr)

