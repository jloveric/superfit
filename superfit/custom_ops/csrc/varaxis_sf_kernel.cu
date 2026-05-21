// SPDX-FileCopyrightText: 2026 Aditya Ganeshan
// SPDX-License-Identifier: MIT

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cmath>

namespace {

constexpr int kBlockSize = 128;
constexpr int kForwardBlockSize = 256;
constexpr int kNumVars = 20;
constexpr int kNumParams = 17;
constexpr int kMaxAssemblyPrims = 128;
constexpr int kFastAssemblyPrims = 32;
constexpr int kFastAssemblyPrims64 = 64;
constexpr int kReduceBlockSize = 256;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kAxisEps = 1.0e-8f;
constexpr float kBulgeEps = 1.0e-5f;
constexpr float kSmoothEps = 1.0e-9f;

template <int N>
struct Dual {
    float v;
    float d[N];

    __device__ Dual() : v(0.0f) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            d[i] = 0.0f;
        }
    }

    __device__ Dual(float value) : v(value) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            d[i] = 0.0f;
        }
    }

    __device__ static Dual variable(float value, int index) {
        Dual out(value);
        out.d[index] = 1.0f;
        return out;
    }
};

__device__ inline float value(float x) {
    return x;
}

template <int N>
__device__ inline float value(const Dual<N>& x) {
    return x.v;
}

template <int N>
__device__ inline Dual<N> operator+(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> out;
    out.v = a.v + b.v;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = a.d[i] + b.d[i];
    }
    return out;
}

template <int N>
__device__ inline Dual<N> operator+(const Dual<N>& a, float b) {
    Dual<N> out = a;
    out.v += b;
    return out;
}

template <int N>
__device__ inline Dual<N> operator+(float a, const Dual<N>& b) {
    return b + a;
}

template <int N>
__device__ inline Dual<N> operator-(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> out;
    out.v = a.v - b.v;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = a.d[i] - b.d[i];
    }
    return out;
}

template <int N>
__device__ inline Dual<N> operator-(const Dual<N>& a, float b) {
    Dual<N> out = a;
    out.v -= b;
    return out;
}

template <int N>
__device__ inline Dual<N> operator-(float a, const Dual<N>& b) {
    Dual<N> out;
    out.v = a - b.v;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = -b.d[i];
    }
    return out;
}

template <int N>
__device__ inline Dual<N> operator-(const Dual<N>& a) {
    Dual<N> out;
    out.v = -a.v;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = -a.d[i];
    }
    return out;
}

template <int N>
__device__ inline Dual<N> operator*(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> out;
    out.v = a.v * b.v;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = a.d[i] * b.v + a.v * b.d[i];
    }
    return out;
}

template <int N>
__device__ inline Dual<N> operator*(const Dual<N>& a, float b) {
    Dual<N> out;
    out.v = a.v * b;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = a.d[i] * b;
    }
    return out;
}

template <int N>
__device__ inline Dual<N> operator*(float a, const Dual<N>& b) {
    return b * a;
}

template <int N>
__device__ inline Dual<N> operator/(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> out;
    const float inv = 1.0f / b.v;
    const float inv2 = inv * inv;
    out.v = a.v * inv;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = (a.d[i] * b.v - a.v * b.d[i]) * inv2;
    }
    return out;
}

template <int N>
__device__ inline Dual<N> operator/(const Dual<N>& a, float b) {
    Dual<N> out;
    const float inv = 1.0f / b;
    out.v = a.v * inv;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = a.d[i] * inv;
    }
    return out;
}

template <int N>
__device__ inline Dual<N> operator/(float a, const Dual<N>& b) {
    Dual<N> out;
    const float inv = 1.0f / b.v;
    const float inv2 = inv * inv;
    out.v = a * inv;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = -a * b.d[i] * inv2;
    }
    return out;
}

__device__ inline float abs_op(float x) {
    return fabsf(x);
}

template <int N>
__device__ inline Dual<N> abs_op(const Dual<N>& x) {
    if (x.v > 0.0f) {
        return x;
    }
    if (x.v < 0.0f) {
        return -x;
    }
    return Dual<N>(0.0f);
}

__device__ inline float sqrt_op(float x) {
    return x > 0.0f ? sqrtf(x) : 0.0f;
}

template <int N>
__device__ inline Dual<N> sqrt_op(const Dual<N>& x) {
    if (x.v <= 0.0f) {
        return Dual<N>(0.0f);
    }
    Dual<N> out;
    out.v = sqrtf(x.v);
    const float scale = 0.5f / out.v;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

__device__ inline float sin_op(float x) {
    return sinf(x);
}

template <int N>
__device__ inline Dual<N> sin_op(const Dual<N>& x) {
    Dual<N> out;
    out.v = sinf(x.v);
    const float c = cosf(x.v);
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = c * x.d[i];
    }
    return out;
}

__device__ inline float cos_op(float x) {
    return cosf(x);
}

template <int N>
__device__ inline Dual<N> cos_op(const Dual<N>& x) {
    Dual<N> out;
    out.v = cosf(x.v);
    const float s = -sinf(x.v);
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = s * x.d[i];
    }
    return out;
}

__device__ inline void sincos_op(float x, float& s, float& c) {
    sincosf(x, &s, &c);
}

template <int N>
__device__ inline void sincos_op(const Dual<N>& x, Dual<N>& s, Dual<N>& c) {
    float sv;
    float cv;
    sincosf(x.v, &sv, &cv);
    s.v = sv;
    c.v = cv;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        s.d[i] = cv * x.d[i];
        c.d[i] = -sv * x.d[i];
    }
}

__device__ inline float tan_op(float x) {
    return tanf(x);
}

template <int N>
__device__ inline Dual<N> tan_op(const Dual<N>& x) {
    Dual<N> out;
    out.v = tanf(x.v);
    const float c = cosf(x.v);
    const float scale = 1.0f / (c * c);
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = scale * x.d[i];
    }
    return out;
}

__device__ inline float exp_op(float x) {
    return expf(x);
}

template <int N>
__device__ inline Dual<N> exp_op(const Dual<N>& x) {
    Dual<N> out;
    out.v = expf(x.v);
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = out.v * x.d[i];
    }
    return out;
}

__device__ inline float atan2_op(float y, float x) {
    return atan2f(y, x);
}

template <int N>
__device__ inline Dual<N> atan2_op(const Dual<N>& y, const Dual<N>& x) {
    Dual<N> out;
    out.v = atan2f(y.v, x.v);
    const float denom = x.v * x.v + y.v * y.v;
    if (denom <= 0.0f) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            out.d[i] = 0.0f;
        }
        return out;
    }
    const float inv = 1.0f / denom;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        out.d[i] = (x.v * y.d[i] - y.v * x.d[i]) * inv;
    }
    return out;
}

__device__ inline float clamp_min_op(float x, float min_v) {
    return fmaxf(x, min_v);
}

template <int N>
__device__ inline Dual<N> clamp_min_op(const Dual<N>& x, float min_v) {
    return x.v >= min_v ? x : Dual<N>(min_v);
}

__device__ inline float clamp_max_op(float x, float max_v) {
    return fminf(x, max_v);
}

template <int N>
__device__ inline Dual<N> clamp_max_op(const Dual<N>& x, float max_v) {
    return x.v <= max_v ? x : Dual<N>(max_v);
}

__device__ inline float clamp_op(float x, float min_v, float max_v) {
    return fminf(fmaxf(x, min_v), max_v);
}

template <int N>
__device__ inline Dual<N> clamp_op(const Dual<N>& x, float min_v, float max_v) {
    if (x.v < min_v) {
        return Dual<N>(min_v);
    }
    if (x.v > max_v) {
        return Dual<N>(max_v);
    }
    return x;
}

__device__ inline float min_op(float a, float b) {
    return fminf(a, b);
}

template <int N>
__device__ inline Dual<N> min_op(const Dual<N>& a, const Dual<N>& b) {
    return a.v <= b.v ? a : b;
}

__device__ inline float max_op(float a, float b) {
    return fmaxf(a, b);
}

template <int N>
__device__ inline Dual<N> max_op(const Dual<N>& a, const Dual<N>& b) {
    return a.v >= b.v ? a : b;
}

__device__ inline float sign_no_grad(float x) {
    return (x > 0.0f) - (x < 0.0f);
}

template <typename S>
__device__ inline void axis_angle_to_rotation_matrix(const S aa[3], S R[9]) {
    const S theta = sqrt_op(aa[0] * aa[0] + aa[1] * aa[1] + aa[2] * aa[2]);
    const S denom = clamp_min_op(theta, kAxisEps);
    const S x = aa[0] / denom;
    const S y = aa[1] / denom;
    const S z = aa[2] / denom;

    S K[9];
    K[0] = S(0.0f);
    K[1] = -z;
    K[2] = y;
    K[3] = z;
    K[4] = S(0.0f);
    K[5] = -x;
    K[6] = -y;
    K[7] = x;
    K[8] = S(0.0f);

    S K2[9];
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            K2[row * 3 + col] =
                K[row * 3 + 0] * K[0 * 3 + col] +
                K[row * 3 + 1] * K[1 * 3 + col] +
                K[row * 3 + 2] * K[2 * 3 + col];
        }
    }

    S s;
    S c;
    sincos_op(theta, s, c);
    const S one_minus_c = S(1.0f) - c;
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            const float ident = row == col ? 1.0f : 0.0f;
            R[row * 3 + col] = S(ident) + s * K[row * 3 + col] + one_minus_c * K2[row * 3 + col];
        }
    }
}

template <typename S>
__device__ inline void precompute_bulge_terms(const S& bulge,
                                              float& bulge_sign,
                                              S& theta_top,
                                              S& sin_theta,
                                              S& cos_theta,
                                              S& cot_theta) {
    bulge_sign = sign_no_grad(value(bulge));
    theta_top = clamp_min_op(abs_op(bulge) * (kPi * 0.5f), kBulgeEps);
    sincos_op(theta_top, sin_theta, cos_theta);
    cot_theta = cos_theta / sin_theta;
}

template <typename S>
__device__ inline S sf_part2_with_bulge(const S coord_in[3], const S size[3], const S& roundness,
                                        const S& dilate_3d, const S& scale,
                                        const S& onion_ratio, float bulge_sign,
                                        const S& theta_top, const S& sin_theta,
                                        const S& cos_theta, const S& cot_theta) {
    const S px0 = coord_in[0] * bulge_sign;
    const S py0 = coord_in[2];
    const S half_z = size[2] * 0.5f;
    const S center_pos = half_z * cot_theta;
    const S dx = px0 - center_pos;
    const S dy = py0;
    const S radius = sqrt_op(center_pos * center_pos + half_z * half_z);
    const S point_angle = atan2_op(dy, -dx);

    const S angle_ratio = clamp_op(point_angle / theta_top, -1.0f, 1.0f);
    const S inside_x = sqrt_op(dx * dx + dy * dy) - radius;
    const S inside_y = angle_ratio * half_z;

    const S s = sin_theta;
    const S c = cos_theta;
    const S along_top = px0 * s + (py0 - half_z) * c;
    const S perp_top = -px0 * c + (py0 - half_z) * s;
    const S above_x = perp_top;
    const S above_y = half_z + along_top;

    const S along_bot = -px0 * s + (py0 + half_z) * c;
    const S perp_bot = -px0 * c - (py0 + half_z) * s;
    const S below_x = perp_bot;
    const S below_y = -half_z + along_bot;

    S mapped_x = inside_x;
    S mapped_y = inside_y;
    if (value(point_angle) > value(theta_top)) {
        mapped_x = above_x;
        mapped_y = above_y;
    }
    if (value(point_angle) < -value(theta_top)) {
        mapped_x = below_x;
        mapped_y = below_y;
    }

    S p[3];
    p[0] = mapped_x * bulge_sign;
    p[1] = coord_in[1];
    p[2] = mapped_y;

    const S inner = min_op(size[0], size[1]) * 0.5f;
    const S h = size[2] * 0.5f;
    const S r = roundness * inner;
    const S q0 = abs_op(p[0]) - size[0] * 0.5f + r;
    const S q1 = abs_op(p[1]) - size[1] * 0.5f + r;
    const S qp0 = clamp_min_op(q0, 0.0f);
    const S qp1 = clamp_min_op(q1, 0.0f);
    const S outside = sqrt_op(qp0 * qp0 + qp1 * qp1);
    const S inside = clamp_max_op(max_op(q0, q1), 0.0f);
    const S sdf2d = outside + inside - r;
    const S x3 = -(S(1.0f) - scale) * inner;

    S ax[4];
    S ay[4];
    ax[0] = -inner + (x3 + inner) * onion_ratio;
    ay[0] = h;
    ax[1] = -inner * (S(1.0f) - onion_ratio);
    ay[1] = -h;
    ax[2] = S(0.0f);
    ay[2] = -h;
    ax[3] = x3;
    ay[3] = h;

    S dmin;
    bool has_dmin = false;
    bool inside_poly = true;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) & 3;
        const S ex = ax[j] - ax[i];
        const S ey = ay[j] - ay[i];
        const S pax = sdf2d - ax[i];
        const S pay = p[2] - ay[i];
        const S denom = clamp_min_op(ex * ex + ey * ey, 1.0e-18f);
        const S t = clamp_op((pax * ex + pay * ey) / denom, 0.0f, 1.0f);
        const S closest_x = ax[i] + t * ex;
        const S closest_y = ay[i] + t * ey;
        const S ddx = sdf2d - closest_x;
        const S ddy = p[2] - closest_y;
        const S dist = sqrt_op(ddx * ddx + ddy * ddy);
        if (!has_dmin || value(dist) < value(dmin)) {
            dmin = dist;
            has_dmin = true;
        }
        const S cross = ex * pay - ey * pax;
        inside_poly = inside_poly && (value(cross) >= 0.0f);
    }

    const S sd = inside_poly ? -dmin : dmin;
    return sd - dilate_3d;
}

template <typename S>
__device__ inline S sf_part2(const S coord_in[3], const S size[3], const S& roundness,
                             const S& dilate_3d, const S& scale, const S& bulge,
                             const S& onion_ratio) {
    float bulge_sign;
    S theta_top;
    S sin_theta;
    S cos_theta;
    S cot_theta;
    precompute_bulge_terms(bulge, bulge_sign, theta_top, sin_theta, cos_theta, cot_theta);
    return sf_part2_with_bulge(
        coord_in,
        size,
        roundness,
        dilate_3d,
        scale,
        onion_ratio,
        bulge_sign,
        theta_top,
        sin_theta,
        cos_theta,
        cot_theta);
}

template <typename S>
__device__ inline S eval_varaxis_sf(const S coord[3], const S params[17], const float gumbel[3],
                                    float temperature) {
    S translate[3] = {params[0], params[1], params[2]};
    S size[3] = {params[3], params[4], params[5]};
    const S roundness = params[6];
    const S dilate_3d = params[7];
    const S scale = params[8];
    const S bulge = params[9];
    const S onion_ratio = params[10];
    S logits[3] = {params[11], params[12], params[13]};
    S rotate[3] = {params[14], params[15], params[16]};

    S R[9];
    axis_angle_to_rotation_matrix(rotate, R);

    float bulge_sign;
    S theta_top;
    S sin_theta;
    S cos_theta;
    S cot_theta;
    precompute_bulge_terms(bulge, bulge_sign, theta_top, sin_theta, cos_theta, cot_theta);

    const S p0 = coord[0] - translate[0];
    const S p1 = coord[1] - translate[1];
    const S p2 = coord[2] - translate[2];
    S tc[3];
    tc[0] = p0 * R[0] + p1 * R[1] + p2 * R[2];
    tc[1] = p0 * R[3] + p1 * R[4] + p2 * R[5];
    tc[2] = p0 * R[6] + p1 * R[7] + p2 * R[8];

    const S sdf_y = sf_part2_with_bulge(
        tc, size, roundness, dilate_3d, scale, onion_ratio,
        bulge_sign, theta_top, sin_theta, cos_theta, cot_theta);

    S coord_z[3] = {tc[1], tc[2], tc[0]};
    S size_z[3] = {size[1], size[2], size[0]};
    const S sdf_z = sf_part2_with_bulge(
        coord_z, size_z, roundness, dilate_3d, scale, onion_ratio,
        bulge_sign, theta_top, sin_theta, cos_theta, cot_theta);

    S coord_x[3] = {tc[2], tc[0], tc[1]};
    S size_x[3] = {size[2], size[0], size[1]};
    const S sdf_x = sf_part2_with_bulge(
        coord_x, size_x, roundness, dilate_3d, scale, onion_ratio,
        bulge_sign, theta_top, sin_theta, cos_theta, cot_theta);

    const S a0 = (logits[0] + gumbel[0]) / temperature;
    const S a1 = (logits[1] + gumbel[1]) / temperature;
    const S a2 = (logits[2] + gumbel[2]) / temperature;
    const float max_a = fmaxf(fmaxf(value(a0), value(a1)), value(a2));
    const S e0 = exp_op(a0 - max_a);
    const S e1 = exp_op(a1 - max_a);
    const S e2 = exp_op(a2 - max_a);
    const S denom = e0 + e1 + e2;
    const S wy = e0 / denom;
    const S wz = e1 / denom;
    const S wx = e2 / denom;

    return wy * sdf_y + wz * sdf_z + wx * sdf_x;
}

template <int N>
struct VarAxisNoLogitEval {
    Dual<N> out;
    float sdf[3];
    float weight[3];
};

template <int N>
__device__ inline VarAxisNoLogitEval<N> eval_varaxis_sf_no_logit_duals(
    const Dual<N> coord[3],
    const Dual<N> params[17],
    const float gumbel[3],
    float temperature) {
    Dual<N> translate[3] = {params[0], params[1], params[2]};
    Dual<N> size[3] = {params[3], params[4], params[5]};
    const Dual<N> roundness = params[6];
    const Dual<N> dilate_3d = params[7];
    const Dual<N> scale = params[8];
    const Dual<N> bulge = params[9];
    const Dual<N> onion_ratio = params[10];
    Dual<N> rotate[3] = {params[14], params[15], params[16]};

    Dual<N> R[9];
    axis_angle_to_rotation_matrix(rotate, R);

    float bulge_sign;
    Dual<N> theta_top;
    Dual<N> sin_theta;
    Dual<N> cos_theta;
    Dual<N> cot_theta;
    precompute_bulge_terms(bulge, bulge_sign, theta_top, sin_theta, cos_theta, cot_theta);

    const Dual<N> p0 = coord[0] - translate[0];
    const Dual<N> p1 = coord[1] - translate[1];
    const Dual<N> p2 = coord[2] - translate[2];
    Dual<N> tc[3];
    tc[0] = p0 * R[0] + p1 * R[1] + p2 * R[2];
    tc[1] = p0 * R[3] + p1 * R[4] + p2 * R[5];
    tc[2] = p0 * R[6] + p1 * R[7] + p2 * R[8];

    const Dual<N> sdf_y = sf_part2_with_bulge(
        tc, size, roundness, dilate_3d, scale, onion_ratio,
        bulge_sign, theta_top, sin_theta, cos_theta, cot_theta);

    Dual<N> coord_z[3] = {tc[1], tc[2], tc[0]};
    Dual<N> size_z[3] = {size[1], size[2], size[0]};
    const Dual<N> sdf_z = sf_part2_with_bulge(
        coord_z, size_z, roundness, dilate_3d, scale, onion_ratio,
        bulge_sign, theta_top, sin_theta, cos_theta, cot_theta);

    Dual<N> coord_x[3] = {tc[2], tc[0], tc[1]};
    Dual<N> size_x[3] = {size[2], size[0], size[1]};
    const Dual<N> sdf_x = sf_part2_with_bulge(
        coord_x, size_x, roundness, dilate_3d, scale, onion_ratio,
        bulge_sign, theta_top, sin_theta, cos_theta, cot_theta);

    const float a0 = (params[11].v + gumbel[0]) / temperature;
    const float a1 = (params[12].v + gumbel[1]) / temperature;
    const float a2 = (params[13].v + gumbel[2]) / temperature;
    const float max_a = fmaxf(fmaxf(a0, a1), a2);
    const float e0 = expf(a0 - max_a);
    const float e1 = expf(a1 - max_a);
    const float e2 = expf(a2 - max_a);
    const float inv_denom = 1.0f / (e0 + e1 + e2);

    VarAxisNoLogitEval<N> eval;
    eval.weight[0] = e0 * inv_denom;
    eval.weight[1] = e1 * inv_denom;
    eval.weight[2] = e2 * inv_denom;
    eval.sdf[0] = sdf_y.v;
    eval.sdf[1] = sdf_z.v;
    eval.sdf[2] = sdf_x.v;
    eval.out = sdf_y * eval.weight[0] + sdf_z * eval.weight[1] + sdf_x * eval.weight[2];
    return eval;
}

struct VarAxisParamVjp {
    float out;
    float grad[kNumParams];
};

template <int N>
__device__ inline Dual<N> eval_sf_part2_axis_duals(float coord0,
                                                   float coord1,
                                                   float coord2,
                                                   float size0,
                                                   float size1,
                                                   float size2,
                                                   float roundness,
                                                   float dilate_3d,
                                                   float scale,
                                                   float bulge,
                                                   float onion_ratio) {
    Dual<N> coord[3] = {
        Dual<N>::variable(coord0, 0),
        Dual<N>::variable(coord1, 1),
        Dual<N>::variable(coord2, 2),
    };
    Dual<N> size[3] = {
        Dual<N>::variable(size0, 3),
        Dual<N>::variable(size1, 4),
        Dual<N>::variable(size2, 5),
    };
    float bulge_sign;
    Dual<N> theta_top;
    Dual<N> sin_theta;
    Dual<N> cos_theta;
    Dual<N> cot_theta;
    precompute_bulge_terms(
        Dual<N>::variable(bulge, 9),
        bulge_sign,
        theta_top,
        sin_theta,
        cos_theta,
        cot_theta);
    return sf_part2_with_bulge(
        coord,
        size,
        Dual<N>::variable(roundness, 6),
        Dual<N>::variable(dilate_3d, 7),
        Dual<N>::variable(scale, 8),
        Dual<N>::variable(onion_ratio, 10),
        bulge_sign,
        theta_top,
        sin_theta,
        cos_theta,
        cot_theta);
}

template <int N>
__device__ inline void accumulate_axis_partials(const Dual<N>& sdf,
                                                float weight,
                                                const int coord_map[3],
                                                const int size_map[3],
                                                float tc_grad[3],
                                                float grad[kNumParams]) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        tc_grad[coord_map[i]] += weight * sdf.d[i];
        grad[3 + size_map[i]] += weight * sdf.d[3 + i];
    }
#pragma unroll
    for (int i = 6; i < 11; ++i) {
        grad[i] += weight * sdf.d[i];
    }
}

__device__ inline VarAxisParamVjp eval_varaxis_sf_param_vjp_fast(
    const float coord[3],
    const float* __restrict__ params,
    const float gumbel[3],
    float temperature) {
    constexpr int kAxisVars = 11;
    VarAxisParamVjp eval;
#pragma unroll
    for (int i = 0; i < kNumParams; ++i) {
        eval.grad[i] = 0.0f;
    }

    const float translate[3] = {params[0], params[1], params[2]};
    const float size[3] = {params[3], params[4], params[5]};
    const float roundness = params[6];
    const float dilate_3d = params[7];
    const float scale = params[8];
    const float bulge = params[9];
    const float onion_ratio = params[10];
    const float rotate[3] = {params[14], params[15], params[16]};

    float R[9];
    axis_angle_to_rotation_matrix(rotate, R);

    const float p[3] = {
        coord[0] - translate[0],
        coord[1] - translate[1],
        coord[2] - translate[2],
    };
    const float tc[3] = {
        p[0] * R[0] + p[1] * R[1] + p[2] * R[2],
        p[0] * R[3] + p[1] * R[4] + p[2] * R[5],
        p[0] * R[6] + p[1] * R[7] + p[2] * R[8],
    };

    const Dual<kAxisVars> sdf_y = eval_sf_part2_axis_duals<kAxisVars>(
        tc[0], tc[1], tc[2], size[0], size[1], size[2],
        roundness, dilate_3d, scale, bulge, onion_ratio);
    const Dual<kAxisVars> sdf_z = eval_sf_part2_axis_duals<kAxisVars>(
        tc[1], tc[2], tc[0], size[1], size[2], size[0],
        roundness, dilate_3d, scale, bulge, onion_ratio);
    const Dual<kAxisVars> sdf_x = eval_sf_part2_axis_duals<kAxisVars>(
        tc[2], tc[0], tc[1], size[2], size[0], size[1],
        roundness, dilate_3d, scale, bulge, onion_ratio);

    const float a0 = (params[11] + gumbel[0]) / temperature;
    const float a1 = (params[12] + gumbel[1]) / temperature;
    const float a2 = (params[13] + gumbel[2]) / temperature;
    const float max_a = fmaxf(fmaxf(a0, a1), a2);
    const float e0 = expf(a0 - max_a);
    const float e1 = expf(a1 - max_a);
    const float e2 = expf(a2 - max_a);
    const float inv_denom = 1.0f / (e0 + e1 + e2);
    const float w[3] = {e0 * inv_denom, e1 * inv_denom, e2 * inv_denom};
    const float sdf[3] = {sdf_y.v, sdf_z.v, sdf_x.v};
    eval.out = w[0] * sdf[0] + w[1] * sdf[1] + w[2] * sdf[2];

    float tc_grad[3] = {0.0f, 0.0f, 0.0f};
    const int coord_y[3] = {0, 1, 2};
    const int size_y[3] = {0, 1, 2};
    accumulate_axis_partials(sdf_y, w[0], coord_y, size_y, tc_grad, eval.grad);

    const int coord_z[3] = {1, 2, 0};
    const int size_z[3] = {1, 2, 0};
    accumulate_axis_partials(sdf_z, w[1], coord_z, size_z, tc_grad, eval.grad);

    const int coord_x[3] = {2, 0, 1};
    const int size_x[3] = {2, 0, 1};
    accumulate_axis_partials(sdf_x, w[2], coord_x, size_x, tc_grad, eval.grad);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        eval.grad[11 + i] = w[i] * (sdf[i] - eval.out) / temperature;
    }

#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float dp =
            tc_grad[0] * R[0 * 3 + j] +
            tc_grad[1] * R[1 * 3 + j] +
            tc_grad[2] * R[2 * 3 + j];
        eval.grad[j] = -dp;
    }

    Dual<3> rotate_dual[3] = {
        Dual<3>::variable(rotate[0], 0),
        Dual<3>::variable(rotate[1], 1),
        Dual<3>::variable(rotate[2], 2),
    };
    Dual<3> R_dual[9];
    axis_angle_to_rotation_matrix(rotate_dual, R_dual);
    Dual<3> tc_dual[3];
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        tc_dual[row] =
            R_dual[row * 3 + 0] * p[0] +
            R_dual[row * 3 + 1] * p[1] +
            R_dual[row * 3 + 2] * p[2];
    }
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        eval.grad[14 + r] =
            tc_grad[0] * tc_dual[0].d[r] +
            tc_grad[1] * tc_dual[1].d[r] +
            tc_grad[2] * tc_dual[2].d[r];
    }
    return eval;
}

__device__ inline float eval_varaxis_sf_cached_forward(const float coord[3],
                                                       const float* __restrict__ sh) {
    constexpr int kTranslate = 0;
    constexpr int kSize = 3;
    constexpr int kRoundness = 6;
    constexpr int kDilate = 7;
    constexpr int kScale = 8;
    constexpr int kOnion = 9;
    constexpr int kR = 10;
    constexpr int kWeight = 19;
    constexpr int kBulgeSign = 22;
    constexpr int kTheta = 23;
    constexpr int kSinTheta = 24;
    constexpr int kCosTheta = 25;
    constexpr int kCotTheta = 26;

    const float p0 = coord[0] - sh[kTranslate + 0];
    const float p1 = coord[1] - sh[kTranslate + 1];
    const float p2 = coord[2] - sh[kTranslate + 2];

    float tc[3];
    tc[0] = p0 * sh[kR + 0] + p1 * sh[kR + 1] + p2 * sh[kR + 2];
    tc[1] = p0 * sh[kR + 3] + p1 * sh[kR + 4] + p2 * sh[kR + 5];
    tc[2] = p0 * sh[kR + 6] + p1 * sh[kR + 7] + p2 * sh[kR + 8];

    float size_y[3] = {sh[kSize + 0], sh[kSize + 1], sh[kSize + 2]};
    const float sdf_y = sf_part2_with_bulge(
        tc, size_y, sh[kRoundness], sh[kDilate], sh[kScale], sh[kOnion],
        sh[kBulgeSign], sh[kTheta], sh[kSinTheta], sh[kCosTheta], sh[kCotTheta]);

    float coord_z[3] = {tc[1], tc[2], tc[0]};
    float size_z[3] = {sh[kSize + 1], sh[kSize + 2], sh[kSize + 0]};
    const float sdf_z = sf_part2_with_bulge(
        coord_z, size_z, sh[kRoundness], sh[kDilate], sh[kScale], sh[kOnion],
        sh[kBulgeSign], sh[kTheta], sh[kSinTheta], sh[kCosTheta], sh[kCotTheta]);

    float coord_x[3] = {tc[2], tc[0], tc[1]};
    float size_x[3] = {sh[kSize + 2], sh[kSize + 0], sh[kSize + 1]};
    const float sdf_x = sf_part2_with_bulge(
        coord_x, size_x, sh[kRoundness], sh[kDilate], sh[kScale], sh[kOnion],
        sh[kBulgeSign], sh[kTheta], sh[kSinTheta], sh[kCosTheta], sh[kCotTheta]);

    return sh[kWeight + 0] * sdf_y + sh[kWeight + 1] * sdf_z + sh[kWeight + 2] * sdf_x;
}

__global__ void varaxis_sf_forward_kernel(const float* __restrict__ coords,
                                          const float* __restrict__ params,
                                          const float* __restrict__ gumbel,
                                          float temperature,
                                          int coord_B,
                                          int param_B,
                                          int M,
                                          float* __restrict__ out) {
    __shared__ float sh[27];

    const int b = blockIdx.y;
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    if (b >= param_B) {
        return;
    }
    if (tid == 0) {
        constexpr int kTranslate = 0;
        constexpr int kSize = 3;
        constexpr int kRoundness = 6;
        constexpr int kDilate = 7;
        constexpr int kScale = 8;
        constexpr int kOnion = 9;
        constexpr int kR = 10;
        constexpr int kWeight = 19;
        constexpr int kBulgeSign = 22;
        constexpr int kTheta = 23;
        constexpr int kSinTheta = 24;
        constexpr int kCosTheta = 25;
        constexpr int kCotTheta = 26;

        const int param_base = b * kNumParams;
        sh[kTranslate + 0] = params[param_base + 0];
        sh[kTranslate + 1] = params[param_base + 1];
        sh[kTranslate + 2] = params[param_base + 2];
        sh[kSize + 0] = params[param_base + 3];
        sh[kSize + 1] = params[param_base + 4];
        sh[kSize + 2] = params[param_base + 5];
        sh[kRoundness] = params[param_base + 6];
        sh[kDilate] = params[param_base + 7];
        sh[kScale] = params[param_base + 8];
        sh[kOnion] = params[param_base + 10];

        float rotate[3] = {
            params[param_base + 14],
            params[param_base + 15],
            params[param_base + 16],
        };
        float R[9];
        axis_angle_to_rotation_matrix(rotate, R);
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            sh[kR + i] = R[i];
        }

        const float a0 = (params[param_base + 11] + gumbel[b * 3 + 0]) / temperature;
        const float a1 = (params[param_base + 12] + gumbel[b * 3 + 1]) / temperature;
        const float a2 = (params[param_base + 13] + gumbel[b * 3 + 2]) / temperature;
        const float max_a = fmaxf(fmaxf(a0, a1), a2);
        const float e0 = expf(a0 - max_a);
        const float e1 = expf(a1 - max_a);
        const float e2 = expf(a2 - max_a);
        const float inv_denom = 1.0f / (e0 + e1 + e2);
        sh[kWeight + 0] = e0 * inv_denom;
        sh[kWeight + 1] = e1 * inv_denom;
        sh[kWeight + 2] = e2 * inv_denom;

        float theta_top;
        float sin_theta;
        float cos_theta;
        float cot_theta;
        float bulge_sign;
        precompute_bulge_terms(
            params[param_base + 9],
            bulge_sign,
            theta_top,
            sin_theta,
            cos_theta,
            cot_theta);
        sh[kBulgeSign] = bulge_sign;
        sh[kTheta] = theta_top;
        sh[kSinTheta] = sin_theta;
        sh[kCosTheta] = cos_theta;
        sh[kCotTheta] = cot_theta;
    }
    __syncthreads();

    if (m >= M) {
        return;
    }
    const int coord_b = coord_B == 1 ? 0 : b;
    const int coord_base = (coord_b * M + m) * 3;
    float c[3] = {
        coords[coord_base + 0],
        coords[coord_base + 1],
        coords[coord_base + 2],
    };
    out[b * M + m] = eval_varaxis_sf_cached_forward(c, sh);
}

__global__ void varaxis_sf_backward_kernel(const float* __restrict__ grad_out,
                                           const float* __restrict__ coords,
                                           const float* __restrict__ params,
                                           const float* __restrict__ gumbel,
                                           float temperature,
                                           int B,
                                           int M,
                                           float* __restrict__ grad_coords,
                                           float* __restrict__ grad_params) {
    __shared__ float sh_param[kBlockSize * kNumParams];

    const int b = blockIdx.y;
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

#pragma unroll
    for (int i = 0; i < kNumParams; ++i) {
        sh_param[tid * kNumParams + i] = 0.0f;
    }

    if (b < B && m < M) {
        const int coord_base = (b * M + m) * 3;
        const int param_base = b * kNumParams;

        Dual<kNumVars> c[3] = {
            Dual<kNumVars>::variable(coords[coord_base + 0], 0),
            Dual<kNumVars>::variable(coords[coord_base + 1], 1),
            Dual<kNumVars>::variable(coords[coord_base + 2], 2),
        };

        Dual<kNumVars> p[kNumParams];
#pragma unroll
        for (int i = 0; i < kNumParams; ++i) {
            p[i] = Dual<kNumVars>::variable(params[param_base + i], 3 + i);
        }

        float g[3] = {
            gumbel[b * 3 + 0],
            gumbel[b * 3 + 1],
            gumbel[b * 3 + 2],
        };
        const Dual<kNumVars> out = eval_varaxis_sf(c, p, g, temperature);
        const float upstream = grad_out[b * M + m];

#pragma unroll
        for (int i = 0; i < 3; ++i) {
            grad_coords[coord_base + i] = upstream * out.d[i];
        }
#pragma unroll
        for (int i = 0; i < kNumParams; ++i) {
            sh_param[tid * kNumParams + i] = upstream * out.d[3 + i];
        }
    }

    __syncthreads();

    if (tid == 0 && b < B) {
        const int param_base = b * kNumParams;
#pragma unroll
        for (int p = 0; p < kNumParams; ++p) {
            float acc = 0.0f;
#pragma unroll
            for (int t = 0; t < kBlockSize; ++t) {
                acc += sh_param[t * kNumParams + p];
            }
            atomicAdd(grad_params + param_base + p, acc);
        }
    }
}

__global__ void varaxis_sf_backward_params_kernel(const float* __restrict__ grad_out,
                                                  const float* __restrict__ coords,
                                                  const float* __restrict__ params,
                                                  const float* __restrict__ gumbel,
                                                  float temperature,
                                                  int coord_B,
                                                  int param_B,
                                                  int M,
                                                  float* __restrict__ grad_params) {
    constexpr int kParamVars = 17;
    __shared__ float sh_param[kBlockSize * kNumParams];

    const int b = blockIdx.y;
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

#pragma unroll
    for (int i = 0; i < kNumParams; ++i) {
        sh_param[tid * kNumParams + i] = 0.0f;
    }

    if (b < param_B && m < M) {
        const int coord_b = coord_B == 1 ? 0 : b;
        const int coord_base = (coord_b * M + m) * 3;
        const int param_base = b * kNumParams;

        Dual<kParamVars> c[3] = {
            Dual<kParamVars>(coords[coord_base + 0]),
            Dual<kParamVars>(coords[coord_base + 1]),
            Dual<kParamVars>(coords[coord_base + 2]),
        };

        Dual<kParamVars> p[kNumParams];
#pragma unroll
        for (int i = 0; i < kNumParams; ++i) {
            p[i] = Dual<kParamVars>::variable(params[param_base + i], i);
        }

        float g[3] = {
            gumbel[b * 3 + 0],
            gumbel[b * 3 + 1],
            gumbel[b * 3 + 2],
        };
        const Dual<kParamVars> out = eval_varaxis_sf(c, p, g, temperature);
        const float upstream = grad_out[b * M + m];

#pragma unroll
        for (int i = 0; i < kNumParams; ++i) {
            sh_param[tid * kNumParams + i] = upstream * out.d[i];
        }
    }

    __syncthreads();

    if (tid == 0 && b < param_B) {
        const int param_base = b * kNumParams;
#pragma unroll
        for (int p = 0; p < kNumParams; ++p) {
            float acc = 0.0f;
#pragma unroll
            for (int t = 0; t < kBlockSize; ++t) {
                acc += sh_param[t * kNumParams + p];
            }
            atomicAdd(grad_params + param_base + p, acc);
        }
    }
}

__global__ void varaxis_sf_backward_params_partials_vjp_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ coords,
    const float* __restrict__ params,
    const float* __restrict__ gumbel,
    float temperature,
    int coord_B,
    int param_B,
    int M,
    int num_tiles,
    float* __restrict__ partials) {
    __shared__ float sh_param[kBlockSize * kNumParams];

    const int tile = blockIdx.x;
    const int b = blockIdx.y;
    const int m = tile * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

#pragma unroll
    for (int i = 0; i < kNumParams; ++i) {
        sh_param[tid * kNumParams + i] = 0.0f;
    }

    if (b < param_B && m < M) {
        const int coord_b = coord_B == 1 ? 0 : b;
        const int coord_base = (coord_b * M + m) * 3;
        const int param_base = b * kNumParams;

        const float c[3] = {
            coords[coord_base + 0],
            coords[coord_base + 1],
            coords[coord_base + 2],
        };
        float g[3] = {
            gumbel[b * 3 + 0],
            gumbel[b * 3 + 1],
            gumbel[b * 3 + 2],
        };
        const VarAxisParamVjp eval =
            eval_varaxis_sf_param_vjp_fast(c, params + param_base, g, temperature);
        const float upstream = grad_out[b * M + m];

#pragma unroll
        for (int i = 0; i < kNumParams; ++i) {
            sh_param[tid * kNumParams + i] = upstream * eval.grad[i];
        }
    }

    __syncthreads();

    if (tid == 0 && b < param_B) {
        const int partial_base = (b * num_tiles + tile) * kNumParams;
#pragma unroll
        for (int p = 0; p < kNumParams; ++p) {
            float acc = 0.0f;
#pragma unroll
            for (int t = 0; t < kBlockSize; ++t) {
                acc += sh_param[t * kNumParams + p];
            }
            partials[partial_base + p] = acc;
        }
    }
}

__global__ void varaxis_sf_reduce_param_partials_kernel(
    const float* __restrict__ partials,
    int param_B,
    int num_tiles,
    float* __restrict__ grad_params) {
    __shared__ float sh[kReduceBlockSize];

    const int p = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;

    float acc = 0.0f;
    if (b < param_B && p < kNumParams) {
        for (int tile = tid; tile < num_tiles; tile += blockDim.x) {
            acc += partials[(b * num_tiles + tile) * kNumParams + p];
        }
    }
    sh[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh[tid] += sh[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && b < param_B && p < kNumParams) {
        grad_params[b * kNumParams + p] = sh[0];
    }
}

__device__ inline void outer_weights(const float* __restrict__ logits,
                                     const float* __restrict__ outer_gumbel,
                                     float temperature,
                                     int b,
                                     float& w0,
                                     float& w1) {
    const float a0 = (logits[b * 2 + 0] + outer_gumbel[b * 2 + 0]) / temperature;
    const float a1 = (logits[b * 2 + 1] + outer_gumbel[b * 2 + 1]) / temperature;
    const float max_a = fmaxf(a0, a1);
    const float e0 = expf(a0 - max_a);
    const float e1 = expf(a1 - max_a);
    const float inv = 1.0f / (e0 + e1);
    w0 = e0 * inv;
    w1 = e1 * inv;
}

__device__ inline float smooth_union_pair_value(float a, float b, float k) {
    const float denom = k + kSmoothEps;
    const float h = fminf(fmaxf(0.5f + 0.5f * (b - a) / denom, 0.0f), 1.0f);
    return b + (a - b) * h - k * h * (1.0f - h);
}

__device__ inline void smooth_union_pair_derivs(float a,
                                                float b,
                                                float k,
                                                float& da,
                                                float& db,
                                                float& dk) {
    const float denom = k + kSmoothEps;
    const float unclamped_h = 0.5f + 0.5f * (b - a) / denom;
    const bool active = unclamped_h >= 0.0f && unclamped_h <= 1.0f;
    const float h = fminf(fmaxf(unclamped_h, 0.0f), 1.0f);
    const float d_out_dh = a - b - k + 2.0f * k * h;
    if (active) {
        const float inv = 1.0f / denom;
        da = h - 0.5f * d_out_dh * inv;
        db = (1.0f - h) + 0.5f * d_out_dh * inv;
        dk = -h * (1.0f - h) - 0.5f * d_out_dh * (b - a) * inv * inv;
    } else {
        da = h;
        db = 1.0f - h;
        dk = 0.0f;
    }
}

__device__ inline float assembly_prefix_value(const float* __restrict__ primitive_sdfs,
                                              const float* __restrict__ su_vals,
                                              int K,
                                              int M,
                                              int m,
                                              int end_exclusive) {
    float out = primitive_sdfs[m];
    for (int b = 1; b < end_exclusive; ++b) {
        out = smooth_union_pair_value(out, primitive_sdfs[b * M + m], su_vals[b - 1]);
    }
    return out;
}

__global__ void varaxis_sf_assembly_forward_kernel(float* __restrict__ primitive_sdfs,
                                                   const float* __restrict__ su_vals,
                                                   const float* __restrict__ logits,
                                                   const float* __restrict__ outer_gumbel,
                                                   float temperature,
                                                   int K,
                                                   int M,
                                                   float* __restrict__ out) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) {
        return;
    }

    float acc = 0.0f;
    for (int b = 0; b < K; ++b) {
        float w0;
        float w1;
        outer_weights(logits, outer_gumbel, temperature, b, w0, w1);
        const int idx = b * M + m;
        const float primitive = primitive_sdfs[idx] * w0 + w1;
        primitive_sdfs[idx] = primitive;
        acc = b == 0 ? primitive : smooth_union_pair_value(acc, primitive, su_vals[b - 1]);
    }
    out[m] = acc;
}

__global__ void varaxis_sf_assembly_backward_prep_kernel(const float* __restrict__ grad_primitive,
                                                         const float* __restrict__ grad_out,
                                                         const float* __restrict__ primitive_sdfs,
                                                         const float* __restrict__ su_vals,
                                                         const float* __restrict__ logits,
                                                         const float* __restrict__ outer_gumbel,
                                                         float temperature,
                                                         int K,
                                                         int M,
                                                         float* __restrict__ grad_raw,
                                                         float* __restrict__ grad_su,
                                                         float* __restrict__ grad_logits) {
    __shared__ float sh_su[kMaxAssemblyPrims];
    __shared__ float sh_logits[kMaxAssemblyPrims * 2];

    const int tid = threadIdx.x;
    const int m = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < K - 1; i += blockDim.x) {
        sh_su[i] = 0.0f;
    }
    for (int i = tid; i < K * 2; i += blockDim.x) {
        sh_logits[i] = 0.0f;
    }
    __syncthreads();

    if (m < M) {
        float carry = grad_out[m];
        for (int b = K - 1; b >= 1; --b) {
            const float a = assembly_prefix_value(primitive_sdfs, su_vals, K, M, m, b);
            const float cur = primitive_sdfs[b * M + m];
            float da;
            float db;
            float dk;
            smooth_union_pair_derivs(a, cur, su_vals[b - 1], da, db, dk);

            const float grad_p = grad_primitive[b * M + m] + carry * db;
            float w0;
            float w1;
            outer_weights(logits, outer_gumbel, temperature, b, w0, w1);
            const float raw = (cur - w1) / fmaxf(w0, 1.0e-20f);
            grad_raw[b * M + m] = grad_p * w0;
            const float dlog0 = grad_p * w0 * w1 * (raw - 1.0f) / temperature;
            atomicAdd(sh_logits + b * 2 + 0, dlog0);
            atomicAdd(sh_logits + b * 2 + 1, -dlog0);
            atomicAdd(sh_su + b - 1, carry * dk);

            carry *= da;
        }

        const float cur = primitive_sdfs[m];
        const float grad_p = grad_primitive[m] + carry;
        float w0;
        float w1;
        outer_weights(logits, outer_gumbel, temperature, 0, w0, w1);
        const float raw = (cur - w1) / fmaxf(w0, 1.0e-20f);
        grad_raw[m] = grad_p * w0;
        const float dlog0 = grad_p * w0 * w1 * (raw - 1.0f) / temperature;
        atomicAdd(sh_logits + 0, dlog0);
        atomicAdd(sh_logits + 1, -dlog0);
    }

    __syncthreads();

    for (int i = tid; i < K - 1; i += blockDim.x) {
        atomicAdd(grad_su + i, sh_su[i]);
    }
    for (int i = tid; i < K * 2; i += blockDim.x) {
        atomicAdd(grad_logits + i, sh_logits[i]);
    }
}

__global__ void varaxis_sf_assembly_backward_prep_smallk_kernel(
    const float* __restrict__ grad_primitive,
    const float* __restrict__ grad_out,
    const float* __restrict__ primitive_sdfs,
    const float* __restrict__ su_vals,
    const float* __restrict__ logits,
    const float* __restrict__ outer_gumbel,
    float temperature,
    int K,
    int M,
    float* __restrict__ grad_raw,
    float* __restrict__ grad_su,
    float* __restrict__ grad_logits) {
    __shared__ float sh_su[kFastAssemblyPrims];
    __shared__ float sh_logits[kFastAssemblyPrims * 2];

    const int tid = threadIdx.x;
    const int m = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < K - 1; i += blockDim.x) {
        sh_su[i] = 0.0f;
    }
    for (int i = tid; i < K * 2; i += blockDim.x) {
        sh_logits[i] = 0.0f;
    }
    __syncthreads();

    if (m < M) {
        float prefix[kFastAssemblyPrims];
        prefix[0] = primitive_sdfs[m];
#pragma unroll
        for (int b = 1; b < kFastAssemblyPrims; ++b) {
            if (b < K) {
                prefix[b] = smooth_union_pair_value(prefix[b - 1], primitive_sdfs[b * M + m], su_vals[b - 1]);
            }
        }

        float carry = grad_out[m];
        for (int b = K - 1; b >= 1; --b) {
            const float cur = primitive_sdfs[b * M + m];
            float da;
            float db;
            float dk;
            smooth_union_pair_derivs(prefix[b - 1], cur, su_vals[b - 1], da, db, dk);

            const float grad_p = grad_primitive[b * M + m] + carry * db;
            float w0;
            float w1;
            outer_weights(logits, outer_gumbel, temperature, b, w0, w1);
            const float raw = (cur - w1) / fmaxf(w0, 1.0e-20f);
            grad_raw[b * M + m] = grad_p * w0;
            const float dlog0 = grad_p * w0 * w1 * (raw - 1.0f) / temperature;
            atomicAdd(sh_logits + b * 2 + 0, dlog0);
            atomicAdd(sh_logits + b * 2 + 1, -dlog0);
            atomicAdd(sh_su + b - 1, carry * dk);

            carry *= da;
        }

        const float cur = primitive_sdfs[m];
        const float grad_p = grad_primitive[m] + carry;
        float w0;
        float w1;
        outer_weights(logits, outer_gumbel, temperature, 0, w0, w1);
        const float raw = (cur - w1) / fmaxf(w0, 1.0e-20f);
        grad_raw[m] = grad_p * w0;
        const float dlog0 = grad_p * w0 * w1 * (raw - 1.0f) / temperature;
        atomicAdd(sh_logits + 0, dlog0);
        atomicAdd(sh_logits + 1, -dlog0);
    }

    __syncthreads();

    for (int i = tid; i < K - 1; i += blockDim.x) {
        atomicAdd(grad_su + i, sh_su[i]);
    }
    for (int i = tid; i < K * 2; i += blockDim.x) {
        atomicAdd(grad_logits + i, sh_logits[i]);
    }
}

__global__ void varaxis_sf_assembly_backward_prep_smallk64_kernel(
    const float* __restrict__ grad_primitive,
    const float* __restrict__ grad_out,
    const float* __restrict__ primitive_sdfs,
    const float* __restrict__ su_vals,
    const float* __restrict__ logits,
    const float* __restrict__ outer_gumbel,
    float temperature,
    int K,
    int M,
    float* __restrict__ grad_raw,
    float* __restrict__ grad_su,
    float* __restrict__ grad_logits) {
    __shared__ float sh_su[kFastAssemblyPrims64];
    __shared__ float sh_logits[kFastAssemblyPrims64 * 2];

    const int tid = threadIdx.x;
    const int m = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < K - 1; i += blockDim.x) {
        sh_su[i] = 0.0f;
    }
    for (int i = tid; i < K * 2; i += blockDim.x) {
        sh_logits[i] = 0.0f;
    }
    __syncthreads();

    if (m < M) {
        float prefix[kFastAssemblyPrims64];
        prefix[0] = primitive_sdfs[m];
#pragma unroll
        for (int b = 1; b < kFastAssemblyPrims64; ++b) {
            if (b < K) {
                prefix[b] = smooth_union_pair_value(prefix[b - 1], primitive_sdfs[b * M + m], su_vals[b - 1]);
            }
        }

        float carry = grad_out[m];
        for (int b = K - 1; b >= 1; --b) {
            const float cur = primitive_sdfs[b * M + m];
            float da;
            float db;
            float dk;
            smooth_union_pair_derivs(prefix[b - 1], cur, su_vals[b - 1], da, db, dk);

            const float grad_p = grad_primitive[b * M + m] + carry * db;
            float w0;
            float w1;
            outer_weights(logits, outer_gumbel, temperature, b, w0, w1);
            const float raw = (cur - w1) / fmaxf(w0, 1.0e-20f);
            grad_raw[b * M + m] = grad_p * w0;
            const float dlog0 = grad_p * w0 * w1 * (raw - 1.0f) / temperature;
            atomicAdd(sh_logits + b * 2 + 0, dlog0);
            atomicAdd(sh_logits + b * 2 + 1, -dlog0);
            atomicAdd(sh_su + b - 1, carry * dk);

            carry *= da;
        }

        const float cur = primitive_sdfs[m];
        const float grad_p = grad_primitive[m] + carry;
        float w0;
        float w1;
        outer_weights(logits, outer_gumbel, temperature, 0, w0, w1);
        const float raw = (cur - w1) / fmaxf(w0, 1.0e-20f);
        grad_raw[m] = grad_p * w0;
        const float dlog0 = grad_p * w0 * w1 * (raw - 1.0f) / temperature;
        atomicAdd(sh_logits + 0, dlog0);
        atomicAdd(sh_logits + 1, -dlog0);
    }

    __syncthreads();

    for (int i = tid; i < K - 1; i += blockDim.x) {
        atomicAdd(grad_su + i, sh_su[i]);
    }
    for (int i = tid; i < K * 2; i += blockDim.x) {
        atomicAdd(grad_logits + i, sh_logits[i]);
    }
}

}  // namespace

void launch_varaxis_sf_forward(
    const float* coords,
    const float* params,
    const float* gumbel,
    float temperature,
    int coord_B,
    int param_B,
    int M,
    float* out) {
    if (param_B == 0 || M == 0) {
        return;
    }
    const dim3 block(kForwardBlockSize);
    const dim3 grid((M + kForwardBlockSize - 1) / kForwardBlockSize, param_B);
    varaxis_sf_forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        coords, params, gumbel, temperature, coord_B, param_B, M, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_varaxis_sf_backward(
    const float* grad_out,
    const float* coords,
    const float* params,
    const float* gumbel,
    float temperature,
    int B,
    int M,
    float* grad_coords,
    float* grad_params) {
    if (B == 0 || M == 0) {
        return;
    }
    const dim3 block(kBlockSize);
    const dim3 grid((M + kBlockSize - 1) / kBlockSize, B);
    varaxis_sf_backward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_out, coords, params, gumbel, temperature, B, M, grad_coords, grad_params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_varaxis_sf_backward_params(
    const float* grad_out,
    const float* coords,
    const float* params,
    const float* gumbel,
    float temperature,
    int coord_B,
    int param_B,
    int M,
    float* grad_params) {
    if (param_B == 0 || M == 0) {
        return;
    }
    const dim3 block(kBlockSize);
    const dim3 grid((M + kBlockSize - 1) / kBlockSize, param_B);
    varaxis_sf_backward_params_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_out, coords, params, gumbel, temperature, coord_B, param_B, M, grad_params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_varaxis_sf_backward_params_partials(
    const float* grad_out,
    const float* coords,
    const float* params,
    const float* gumbel,
    float temperature,
    int coord_B,
    int param_B,
    int M,
    int num_tiles,
    float* partials) {
    if (param_B == 0 || M == 0) {
        return;
    }
    const dim3 partial_block(kBlockSize);
    const dim3 partial_grid(num_tiles, param_B);
    varaxis_sf_backward_params_partials_vjp_kernel<<<
        partial_grid, partial_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_out, coords, params, gumbel, temperature, coord_B, param_B, M, num_tiles, partials);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_varaxis_sf_reduce_param_partials(
    const float* partials,
    int param_B,
    int num_tiles,
    float* grad_params) {
    if (param_B == 0 || num_tiles == 0) {
        return;
    }
    const dim3 reduce_block(kReduceBlockSize);
    const dim3 reduce_grid(kNumParams, param_B);
    varaxis_sf_reduce_param_partials_kernel<<<
        reduce_grid, reduce_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        partials, param_B, num_tiles, grad_params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_varaxis_sf_backward_params_reduced(
    const float* grad_out,
    const float* coords,
    const float* params,
    const float* gumbel,
    float temperature,
    int coord_B,
    int param_B,
    int M,
    int num_tiles,
    float* partials,
    float* grad_params) {
    launch_varaxis_sf_backward_params_partials(
        grad_out, coords, params, gumbel, temperature, coord_B, param_B, M, num_tiles, partials);
    launch_varaxis_sf_reduce_param_partials(partials, param_B, num_tiles, grad_params);
}

void launch_varaxis_sf_assembly_forward(
    float* primitive_sdfs,
    const float* su_vals,
    const float* logits,
    const float* outer_gumbel,
    float temperature,
    int K,
    int M,
    float* out) {
    if (K == 0 || M == 0) {
        return;
    }
    const dim3 block(kForwardBlockSize);
    const dim3 grid((M + kForwardBlockSize - 1) / kForwardBlockSize);
    varaxis_sf_assembly_forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        primitive_sdfs, su_vals, logits, outer_gumbel, temperature, K, M, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_varaxis_sf_assembly_backward_prep(
    const float* grad_primitive,
    const float* grad_out,
    const float* primitive_sdfs,
    const float* su_vals,
    const float* logits,
    const float* outer_gumbel,
    float temperature,
    int K,
    int M,
    float* grad_raw,
    float* grad_su,
    float* grad_logits) {
    if (K == 0 || M == 0) {
        return;
    }
    const dim3 block(kBlockSize);
    const dim3 grid((M + kBlockSize - 1) / kBlockSize);
    if (K <= kFastAssemblyPrims) {
        varaxis_sf_assembly_backward_prep_smallk_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_primitive, grad_out, primitive_sdfs, su_vals, logits, outer_gumbel,
            temperature, K, M, grad_raw, grad_su, grad_logits);
    } else if (K <= kFastAssemblyPrims64) {
        varaxis_sf_assembly_backward_prep_smallk64_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_primitive, grad_out, primitive_sdfs, su_vals, logits, outer_gumbel,
            temperature, K, M, grad_raw, grad_su, grad_logits);
    } else {
        varaxis_sf_assembly_backward_prep_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_primitive, grad_out, primitive_sdfs, su_vals, logits, outer_gumbel,
            temperature, K, M, grad_raw, grad_su, grad_logits);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
