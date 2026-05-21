from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


if not torch.cuda.is_available():
    pytestmark = pytest.mark.skip(reason="CUDA is required for VarAxisSF custom op tests")


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from superfit.torch_compute.batched_sf import (
        batched_sf_packed_eval_part_2,
        common_transform_coords,
        unpacked_params_varsf,
    )
except Exception:  # pragma: no cover - fallback keeps the test bed standalone

    def _axis_angle_to_rotation_matrix(axis_angle, eps=1e-8):
        B = axis_angle.shape[0]
        theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
        axis = axis_angle / theta.clamp_min(eps)
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        K = torch.stack(
            [
                zero,
                -z,
                y,
                z,
                zero,
                -x,
                -y,
                x,
                zero,
            ],
            dim=-1,
        ).reshape(B, 3, 3)
        I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand(B, 3, 3)
        sin = torch.sin(theta).unsqueeze(-1)
        cos = torch.cos(theta).unsqueeze(-1)
        return I + sin * K + (1.0 - cos) * (K @ K)

    def common_transform_coords(coords, translate, rotate):
        R = _axis_angle_to_rotation_matrix(rotate)
        return torch.matmul(coords - translate.unsqueeze(1), R.transpose(-1, -2))

    def unpacked_params_varsf(params):
        return (
            params[..., :3],
            params[..., 3:6],
            params[..., 6:7],
            params[..., 7:8],
            params[..., 8:9],
            params[..., 9:10],
            params[..., 10:11],
            params[..., 11:14],
            params[..., 14:17],
        )

    def _map_arc_bulge(points, z, bulge, eps=1e-5):
        px = points[..., 0] * torch.sign(bulge)
        py = points[..., 1]
        half_z = 0.5 * z
        theta_top = torch.clamp_min(torch.abs(bulge) * (torch.pi * 0.5), eps)
        center_pos = half_z / torch.tan(theta_top)
        dx = px - center_pos
        dy = py
        radius = torch.sqrt(torch.square(center_pos) + torch.square(half_z))
        point_angle = torch.atan2(dy, -dx)
        angle_ratio = torch.clamp(point_angle / theta_top, -1.0, 1.0)
        new_y = angle_ratio * half_z
        new_x = torch.sqrt(torch.square(dx) + torch.square(dy)) - radius
        inside_point = torch.stack((new_x, new_y), dim=-1)
        s = torch.sin(theta_top)
        c = torch.cos(theta_top)
        along_top = px * s + (py - half_z) * c
        perp_top = -px * c + (py - half_z) * s
        above_point = torch.stack((perp_top, half_z + along_top), dim=-1)
        along_bot = -px * s + (py + half_z) * c
        perp_bot = -px * c - (py + half_z) * s
        below_point = torch.stack((perp_bot, -half_z + along_bot), dim=-1)
        out = torch.where((point_angle > theta_top)[..., None], above_point, inside_point)
        out = torch.where((point_angle < -theta_top)[..., None], below_point, out)
        return torch.stack((out[..., 0] * torch.sign(bulge), out[..., 1]), dim=-1)

    def _sd_taper_trapezoid_onion_exact_batched(pos_2d, inner, half_height, x3, onion_ratio):
        B, _, _ = pos_2d.shape
        inner = inner.view(B, 1)
        half_height = half_height.view(B, 1)
        x3 = x3.view(B, 1)
        onion_ratio = onion_ratio.view(B, 1)
        zero = torch.zeros_like(inner)
        A = torch.stack(
            (
                torch.cat([-inner + (x3 + inner) * onion_ratio, +half_height], dim=1),
                torch.cat([-inner * (1 - onion_ratio), -half_height], dim=1),
                torch.cat([zero, -half_height], dim=1),
                torch.cat([x3, +half_height], dim=1),
            ),
            dim=1,
        )
        Bv = A.roll(shifts=-1, dims=1)
        E = Bv - A
        P = pos_2d.unsqueeze(2)
        A_ = A.unsqueeze(1)
        E_ = E.unsqueeze(1)
        PA = P - A_
        denom = (E_ * E_).sum(dim=-1).clamp_min(1e-18)
        t = ((PA * E_).sum(dim=-1) / denom).clamp(0.0, 1.0)
        closest = A_ + t.unsqueeze(-1) * E_
        dists = (P - closest).norm(dim=-1)
        dmin = dists.min(dim=-1).values
        cross = E_[..., 0] * PA[..., 1] - E_[..., 1] * PA[..., 0]
        inside = (cross >= 0).all(dim=-1)
        return torch.where(inside, -dmin, dmin)

    def batched_sf_packed_eval_part_2(
        transformed_coords,
        size,
        roundness,
        dilate_3d,
        scale,
        bulge_ratio,
        onion_ratio,
    ):
        new_p_xz = _map_arc_bulge(
            transformed_coords[..., (0, 2)],
            size[..., 2:3],
            bulge_ratio,
        )
        transformed_coords = torch.stack(
            (new_p_xz[..., 0], transformed_coords[..., 1], new_p_xz[..., 1]), dim=-1
        )
        xy = transformed_coords[..., :2]
        z = transformed_coords[..., 2]
        inner = 0.5 * size[..., :2].amin(dim=-1)
        h = 0.5 * size[..., 2]
        r = (roundness.squeeze(-1) * inner).unsqueeze(-1)
        q = xy.abs() - (size[..., :2] * 0.5).unsqueeze(1) + r.unsqueeze(-1)
        outside = torch.linalg.vector_norm(torch.clamp_min(q, 0.0), dim=-1)
        inside = torch.clamp_max(torch.maximum(q[..., 0], q[..., 1]), 0.0)
        sdf2d = outside + inside - r
        x3 = -(1.0 - scale.squeeze(-1)) * inner
        pos_2d = torch.stack((sdf2d, z), dim=-1)
        sd = _sd_taper_trapezoid_onion_exact_batched(pos_2d, inner, h, x3, onion_ratio)
        return sd - dilate_3d

from superfit.custom_ops.varaxis_sf_cuda import varaxis_sf_assembly_cuda, varaxis_sf_cuda


def _smooth_union_pair(sdf_a, sdf_b, k):
    h = torch.clamp(0.5 + 0.5 * (sdf_b - sdf_a) / (k + 1.0e-9), min=0.0, max=1.0)
    return torch.lerp(sdf_b, sdf_a, h) - k * h * (1.0 - h)


def reference_varaxis_sf(coords, params, temperature, gumbel):
    translate, size, roundness, dilate_3d, scale, bulge, onion, logits, rotate = (
        unpacked_params_varsf(params)
    )
    transformed = common_transform_coords(coords, translate, rotate)
    sdf_y = batched_sf_packed_eval_part_2(
        transformed, size, roundness, dilate_3d, scale, bulge, onion
    )
    sdf_z = batched_sf_packed_eval_part_2(
        transformed[:, :, [1, 2, 0]],
        size[:, [1, 2, 0]],
        roundness,
        dilate_3d,
        scale,
        bulge,
        onion,
    )
    sdf_x = batched_sf_packed_eval_part_2(
        transformed[:, :, [2, 0, 1]],
        size[:, [2, 0, 1]],
        roundness,
        dilate_3d,
        scale,
        bulge,
        onion,
    )
    w = torch.softmax((logits + gumbel) / float(temperature), dim=-1)
    return w[:, 0:1] * sdf_y + w[:, 1:2] * sdf_z + w[:, 2:3] * sdf_x


def reference_assembly(coords, params, su_vals, logits, temperature, inner_gumbel, outer_gumbel):
    primitive_sdfs = reference_varaxis_sf(coords, params, temperature, inner_gumbel)
    outer_w = torch.softmax((logits + outer_gumbel) / float(temperature), dim=-1)
    primitive_sdfs = primitive_sdfs * outer_w[:, 0:1] + outer_w[:, 1:2]
    out = primitive_sdfs[0]
    for i in range(1, primitive_sdfs.shape[0]):
        out = _smooth_union_pair(out, primitive_sdfs[i], su_vals[i - 1].unsqueeze(-1))
    return primitive_sdfs, out


def make_inputs(B=4, M=97, seed=0):
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    coords = torch.empty(B, M, 3, device=device).uniform_(-0.65, 0.65, generator=gen)

    translate = torch.empty(B, 3, device=device).uniform_(-0.2, 0.2, generator=gen)
    size = torch.empty(B, 3, device=device).uniform_(0.35, 1.1, generator=gen)
    size[:, 1] = size[:, 1] + 0.071
    size[:, 2] = size[:, 2] + 0.113
    roundness = torch.empty(B, 1, device=device).uniform_(0.05, 0.45, generator=gen)
    dilate = torch.empty(B, 1, device=device).uniform_(-0.04, 0.04, generator=gen)
    scale = torch.empty(B, 1, device=device).uniform_(0.45, 1.45, generator=gen)

    bulge_mag = torch.empty(B, 1, device=device).uniform_(0.12, 0.75, generator=gen)
    bulge_sign = torch.where(
        torch.arange(B, device=device).view(B, 1) % 2 == 0,
        torch.ones(B, 1, device=device),
        -torch.ones(B, 1, device=device),
    )
    bulge = bulge_mag * bulge_sign
    onion = torch.empty(B, 1, device=device).uniform_(0.08, 0.75, generator=gen)

    base_logits = torch.tensor(
        [[3.0, -1.0, -2.0], [-1.0, 3.0, -2.0], [-2.0, -1.0, 3.0]],
        device=device,
    )
    logits = base_logits[torch.arange(B, device=device) % 3].clone()
    logits = logits + torch.empty(B, 3, device=device).uniform_(-0.25, 0.25, generator=gen)

    rotate = torch.empty(B, 3, device=device).uniform_(-0.75, 0.75, generator=gen)
    rotate = rotate + torch.tensor([0.17, -0.11, 0.23], device=device)

    params = torch.cat(
        [
            translate,
            size,
            roundness,
            dilate,
            scale,
            bulge,
            onion,
            logits,
            rotate,
        ],
        dim=-1,
    ).contiguous()
    gumbel = torch.empty(B, 3, device=device).uniform_(-0.6, 0.6, generator=gen).contiguous()
    return coords.contiguous(), params, 0.73, gumbel


def make_assembly_inputs(K=4, M=97, seed=0):
    coords, params, temperature, inner_gumbel = make_inputs(B=K, M=M, seed=seed)
    coords = coords[:1].contiguous()
    gen = torch.Generator(device="cuda").manual_seed(seed + 7000)
    su_vals = torch.empty(K - 1, 1, device="cuda").uniform_(0.01, 0.08, generator=gen)
    logits = torch.empty(K, 2, device="cuda").normal_(0.0, 1.0, generator=gen)
    outer_gumbel = torch.empty(K, 2, device="cuda").uniform_(-0.6, 0.6, generator=gen)
    prim_grad = torch.empty(K, M, device="cuda").uniform_(-0.7, 0.7, generator=gen)
    sdf_grad = torch.empty(M, device="cuda").uniform_(-0.7, 0.7, generator=gen)
    return (
        coords,
        params.contiguous(),
        su_vals.contiguous(),
        logits.contiguous(),
        temperature,
        inner_gumbel.contiguous(),
        outer_gumbel.contiguous(),
        prim_grad.contiguous(),
        sdf_grad.contiguous(),
    )


@pytest.mark.parametrize("B,M", [(1, 17), (3, 67), (6, 129)])
def test_forward_matches_reference(B, M):
    coords, params, temperature, gumbel = make_inputs(B=B, M=M, seed=10 + B + M)
    actual = varaxis_sf_cuda(coords, params, temperature, gumbel, grad_mode="full")
    expected = reference_varaxis_sf(coords, params, temperature, gumbel)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)


def test_backward_matches_reference():
    coords, params, temperature, gumbel = make_inputs(B=3, M=53, seed=123)
    grad_out = torch.empty(3, 53, device="cuda").uniform_(-0.7, 0.7)

    coords_ref = coords.detach().clone().requires_grad_(True)
    params_ref = params.detach().clone().requires_grad_(True)
    expected = reference_varaxis_sf(coords_ref, params_ref, temperature, gumbel)
    expected.backward(grad_out)

    coords_cuda = coords.detach().clone().requires_grad_(True)
    params_cuda = params.detach().clone().requires_grad_(True)
    actual = varaxis_sf_cuda(coords_cuda, params_cuda, temperature, gumbel, grad_mode="full")
    actual.backward(grad_out)

    torch.testing.assert_close(actual, expected.detach(), rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(coords_cuda.grad, coords_ref.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(params_cuda.grad, params_ref.grad, rtol=8e-3, atol=8e-3)


def test_backward_params_mode_matches_reference():
    coords, params, temperature, gumbel = make_inputs(B=5, M=89, seed=321)
    grad_out = torch.empty(5, 89, device="cuda").uniform_(-0.7, 0.7)

    coords_ref = coords.detach().clone()
    params_ref = params.detach().clone().requires_grad_(True)
    expected = reference_varaxis_sf(coords_ref, params_ref, temperature, gumbel)
    expected.backward(grad_out)

    coords_cuda = coords.detach().clone()
    params_cuda = params.detach().clone().requires_grad_(True)
    actual = varaxis_sf_cuda(coords_cuda, params_cuda, temperature, gumbel, grad_mode="params")
    actual.backward(grad_out)

    torch.testing.assert_close(actual, expected.detach(), rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(params_cuda.grad, params_ref.grad, rtol=8e-3, atol=8e-3)


def test_broadcast_coords_forward_and_params_grad_match_reference():
    coords, params, temperature, gumbel = make_inputs(B=4, M=61, seed=432)
    coords = coords[:1].contiguous()
    grad_out = torch.empty(4, 61, device="cuda").uniform_(-0.7, 0.7)

    params_ref = params.detach().clone().requires_grad_(True)
    expected = reference_varaxis_sf(coords.detach().clone(), params_ref, temperature, gumbel)
    expected.backward(grad_out)

    params_cuda = params.detach().clone().requires_grad_(True)
    actual = varaxis_sf_cuda(
        coords.detach().clone(),
        params_cuda,
        temperature,
        gumbel,
        grad_mode="params",
    )
    actual.backward(grad_out)

    assert actual.shape == (4, 61)
    torch.testing.assert_close(actual, expected.detach(), rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(params_cuda.grad, params_ref.grad, rtol=8e-3, atol=8e-3)


def test_broadcast_coords_with_coord_grad_is_rejected():
    coords, params, temperature, gumbel = make_inputs(B=3, M=23, seed=433)
    coords = coords[:1].contiguous().requires_grad_(True)
    with pytest.raises(ValueError, match="broadcast coords"):
        varaxis_sf_cuda(coords, params, temperature, gumbel, grad_mode="auto")


@pytest.mark.parametrize("K,M", [(32, 193), (64, 257)])
def test_assembly_params_grad_matches_reference_at_large_k(K, M):
    (
        coords,
        params,
        su_vals,
        logits,
        temperature,
        inner_gumbel,
        outer_gumbel,
        prim_grad,
        sdf_grad,
    ) = make_assembly_inputs(K=K, M=M, seed=800 + K)

    params_ref = params.detach().clone().requires_grad_(True)
    su_ref = su_vals.detach().clone().requires_grad_(True)
    logits_ref = logits.detach().clone().requires_grad_(True)
    ref_prim, ref_sdf = reference_assembly(
        coords.detach(),
        params_ref,
        su_ref,
        logits_ref,
        temperature,
        inner_gumbel,
        outer_gumbel,
    )
    torch.autograd.backward((ref_prim, ref_sdf), (prim_grad, sdf_grad.reshape_as(ref_sdf)))

    params_cuda = params.detach().clone().requires_grad_(True)
    su_cuda = su_vals.detach().clone().requires_grad_(True)
    logits_cuda = logits.detach().clone().requires_grad_(True)
    custom_prim, custom_sdf = varaxis_sf_assembly_cuda(
        coords.detach(),
        params_cuda,
        su_cuda,
        logits_cuda,
        temperature,
        inner_gumbel,
        outer_gumbel,
    )
    torch.autograd.backward((custom_prim, custom_sdf), (prim_grad, sdf_grad.reshape_as(custom_sdf)))

    torch.testing.assert_close(custom_prim, ref_prim.detach(), rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(custom_sdf.reshape_as(ref_sdf), ref_sdf.detach(), rtol=3e-4, atol=3e-4)

    param_diff = (params_cuda.grad - params_ref.grad).abs()
    su_diff = (su_cuda.grad - su_ref.grad).abs()
    logits_diff = (logits_cuda.grad - logits_ref.grad).abs()

    assert param_diff.mean().item() <= 1e-4
    assert su_diff.mean().item() <= 1e-5
    assert logits_diff.mean().item() <= 1e-5
    assert int((param_diff > 1e-3).sum().item()) <= 8
    assert int((su_diff > 1e-3).sum().item()) == 0
    assert int((logits_diff > 1e-3).sum().item()) == 0


def test_auto_mode_uses_params_only_when_coords_do_not_require_grad():
    coords, params, temperature, gumbel = make_inputs(B=4, M=73, seed=654)
    grad_out = torch.empty(4, 73, device="cuda").uniform_(-0.7, 0.7)

    params_ref = params.detach().clone().requires_grad_(True)
    expected = reference_varaxis_sf(coords.detach().clone(), params_ref, temperature, gumbel)
    expected.backward(grad_out)

    params_cuda = params.detach().clone().requires_grad_(True)
    actual = varaxis_sf_cuda(
        coords.detach().clone(),
        params_cuda,
        temperature,
        gumbel,
        grad_mode="auto",
    )
    actual.backward(grad_out)

    torch.testing.assert_close(actual, expected.detach(), rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(params_cuda.grad, params_ref.grad, rtol=8e-3, atol=8e-3)


def test_randomized_stress_branch_diagnostics():
    B, M, seed = 7, 1024, 1003
    coords, params, temperature, gumbel = make_inputs(B=B, M=M, seed=seed)
    gen = torch.Generator(device="cuda").manual_seed(seed + 999)
    coords = torch.empty(B, M, 3, device="cuda").uniform_(-0.8, 0.8, generator=gen).contiguous()
    params[:, 11:14] = torch.empty(B, 3, device="cuda").normal_(0.0, 1.6, generator=gen)
    params[:, 14:17] = torch.empty(B, 3, device="cuda").uniform_(-1.4, 1.4, generator=gen)
    gumbel = torch.empty(B, 3, device="cuda").uniform_(-1.0, 1.0, generator=gen).contiguous()
    grad_out = torch.empty(B, M, device="cuda").uniform_(-0.8, 0.8, generator=gen)

    coords_ref = coords.detach().clone().requires_grad_(True)
    params_ref = params.detach().clone().requires_grad_(True)
    expected = reference_varaxis_sf(coords_ref, params_ref, temperature, gumbel)
    expected.backward(grad_out)

    coords_cuda = coords.detach().clone().requires_grad_(True)
    params_cuda = params.detach().clone().requires_grad_(True)
    actual = varaxis_sf_cuda(coords_cuda, params_cuda, temperature, gumbel, grad_mode="full")
    actual.backward(grad_out)

    forward_diff = (actual - expected.detach()).abs()
    coord_grad_diff = (coords_cuda.grad - coords_ref.grad).abs()
    param_grad_diff = (params_cuda.grad - params_ref.grad).abs()

    assert forward_diff.max().item() <= 5e-6
    assert coord_grad_diff.mean().item() <= 1e-6
    assert param_grad_diff.mean().item() <= 1e-4
    assert int((coord_grad_diff > 1e-3).sum().item()) <= 4
    assert int((param_grad_diff > 1e-3).sum().item()) <= 4


def test_input_validation():
    coords, params, temperature, gumbel = make_inputs(B=2, M=11, seed=77)
    with pytest.raises(ValueError, match="contiguous"):
        varaxis_sf_cuda(coords[:, ::2, :], params, temperature, gumbel)
    with pytest.raises(ValueError, match="shape"):
        varaxis_sf_cuda(coords, params[:, :16].contiguous(), temperature, gumbel)
    with pytest.raises(ValueError, match="positive"):
        varaxis_sf_cuda(coords, params, 0.0, gumbel)
    with pytest.raises(ValueError, match="grad_mode"):
        varaxis_sf_cuda(coords, params, temperature, gumbel, grad_mode="bad")


@pytest.mark.skipif(
    os.environ.get("RUN_VARAXIS_SF_BENCH") != "1",
    reason="set RUN_VARAXIS_SF_BENCH=1 to run the benchmark smoke test",
)
def test_benchmark_smoke():
    coords, params, temperature, gumbel = make_inputs(B=100, M=200_000, seed=5)
    torch.cuda.synchronize()
    out = varaxis_sf_cuda(coords, params, temperature, gumbel)
    torch.cuda.synchronize()
    assert out.shape == (100, 200_000)
