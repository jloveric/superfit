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

Old primitives explored during designing SuperFrustum.
These do not have the torch compute implementations provided here (too much messy code to clean).
"""
import numpy as np
from sysl.shader.shader_module import register_shader_module
from sysl.shader.shader_templates.common import CONSTANTS


SPTaperedWrongV1Shader = register_shader_module("""
@name SPTaperedWrongV1
@inputs pos, radius
@outputs dist
@dependencies SPBase
@vardeps 
float SPTaperedWrongV1(vec3 p, vec3 size, float roundness, float dilate_3d, float scaling)
{
    vec2 uv = p.xy;
    float t = p.z / size.z;
    float t_act = clamp(t, 0.0, 1.0);
    float cur_scale = mix(1.0, scaling, t_act);
    vec2 cur_size = size.xy * cur_scale;

    float min_size = min(cur_size.x, cur_size.y)  * 0.5;
    float r = roundness * min_size;
    // float onion_amount = (1.0 - onion_2d) * min_size;

    float sdf2d  = HalfRoundedRectangle2D(uv, cur_size, r);
    // sdf2d = (sdf2d < 0.0)? abs(sdf2d) - onion_amount : sdf2d;
    vec2 d = vec2(sdf2d, max( p.z - size.z, -p.z));
    float sdf3d = min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
    return sdf3d - dilate_3d;
}""")


SPTaperedWrongV2Shader = register_shader_module("""
@name SPTaperedWrongV2
@inputs pos, radius
@outputs dist
@dependencies SPBase
@vardeps 
float SPTaperedWrongV2(vec3 p, vec3 size, vec2 roundness, float onion_2d, float dilate_3d, vec2 scaling)
{
    // Fold to first quadrant
    float H = size.z;
    p.x = abs(p.x);
    p.y = abs(p.y);
    vec2 uv = p.xy;
    vec2 size_bottom = size.xy;
    vec2 size_top    = size_bottom * scaling;
    float r0 = roundness.x * min(size_bottom.x, size_bottom.y) * 0.5;
    float r1 = roundness.y * min(size_top.x, size_top.y) * 0.5;
    float t = p.z / size.z;
    float t_act = clamp(t, 0.0, 1.0);
    vec2 cur_scale = mix(vec2(1.0, 1.0), scaling, t_act);
    vec2 cur_size = mix(size_bottom, size_top, t_act);
    float cur_roundness = mix(roundness.x, roundness.y, t_act);
    float min_size = min(cur_size.x, cur_size.y)  * 0.5;
    float r = cur_roundness * min_size;
    float onion_amount = (1.0 - onion_2d) * min_size;
    float sdf2d  = HalfRoundedRectangle2D(uv, cur_size, r);
    sdf2d = (sdf2d < 0.0)? abs(sdf2d) - onion_amount : sdf2d;
    vec2 d = vec2(sdf2d, max( p.z - size.z, -p.z));
    float sdf3d = min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
    return sdf3d - dilate_3d;
}""")


SPTaperedSupportShader = register_shader_module("""
@name SPTaperedSupport
@inputs pos, radius
@outputs dist
@dependencies SPBase
@vardeps
// --- small utils ---
float dot2(vec2 v){ return dot(v,v); }
float dot2(vec3 v){ return dot(v,v); }
float sq(float x){ return x*x; }
float saturate(float x){ return clamp(x, 0.0, 1.0); }

// Branchless rounded cap, returning **squared** distance
float udCapRoundedSq(vec3 p, float zPlane, vec2 size, float r)
{
    // Signed distance in 2D, positive outside
    float s = RoundedRectangle2D(p.xy, size, r);
    float out2D = max(s, 0.0);
    float dz    = p.z - zPlane;
    return out2D*out2D + dz*dz;
}
// ---- helpers: distance to a 2D segment (squared) ----
float segDist2(vec2 p, vec2 a, vec2 b){
    vec2 ab = b - a;
    float denom = max(dot(ab,ab), 1e-12);
    float t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
    vec2 q = a + t * ab;
    vec2 d = p - q;
    return dot(d,d);
}

// In-plane squared distance to a trapezoid with bottom [ (0,0) -> (L,0) ]
// and top   [ (0,b0) -> (L,b1) ]. Returns 0 inside, exact outside.
float trapezoidInPlaneD2(vec2 uv, float L, float b0, float b1){
    // Early out for degenerate length
    if (L <= 1e-12) {
        // collapses to a vertical segment from (0,b0) to (0,b1)
        float d2v = segDist2(uv, vec2(0.0, b0), vec2(0.0, b1));
        // also compare to bottom segment when L≈0 (both endpoints at u=0)
        float d2b = segDist2(uv, vec2(0.0, 0.0), vec2(0.0, 0.0));
        return min(d2v, d2b);
    }

    // linear top height at u
    float u = uv.x;
    float v = uv.y;
    float t = u / L;
    float vmax = mix(b0, b1, t);

    // inside test (robust, with a tiny epsilon)
    const float EPS = 1e-7;
    bool inside = (u >= -EPS) && (u <= L + EPS) &&
                  (v >= -EPS) && (v <= vmax + EPS);

    if (inside) return 0.0;

    // exact outside distance: min to the four finite edges
    float d2_bottom = segDist2(uv, vec2(0.0, 0.0), vec2(L, 0.0));
    float d2_top    = segDist2(uv, vec2(0.0, b0),  vec2(L, b1));
    float d2_left   = segDist2(uv, vec2(0.0, 0.0), vec2(0.0, b0));
    float d2_right  = segDist2(uv, vec2(L, 0.0),   vec2(L, b1));
    return min(min(d2_bottom, d2_top), min(d2_left, d2_right));
}

// Generic 3D trapezoid prism squared distance:
//  - o  : origin of the bottom-left corner in 3D (u=0, v=0 at z from o)
//  - e1 : in-plane axis along u (z component encodes H): e1 = (Δx, 0, H) or (0, Δy, H)
//  - b0,b1 : top heights at u=0 and u=L respectively (v max at each end)
// NOTE: we orthogonally project p to the plane spanned by <e1, axisK> before computing (u,v).
float trapezoidPrismD2(vec3 p, vec3 o, vec3 e1, float b0, float b1){
    float L2 = dot(e1, e1);
    if (L2 < 1e-18) return 1e30; // degenerate

    float invL = inversesqrt(L2);
    float L    = L2 * invL;
    vec3  uhat = e1 * invL;          // unit along e1

    // choose which “cross” axis lies in the trapezoid plane
    bool useY  = (abs(e1.y) > abs(e1.x));
    vec3  khat = useY ? vec3(1.0, 0.0, 0.0)  // Y-trap: plane spanned by e1 and X
                      : vec3(0.0, 1.0, 0.0); // X-trap: plane spanned by e1 and Y

    // plane normal (unnormalized)
    vec3  nu   = cross(khat, e1);    // guaranteed ⟂ to both uhat and khat
    float n2   = dot(nu, nu);
    if (n2 < 1e-30) return 1e30;     // degenerate

    // project to the plane through 'o' spanned by {uhat, khat}
    vec3  w3   = p - o;
    float wn   = dot(w3, nu);
    vec3  q    = p - (wn / n2) * nu; // orthogonal projection onto the plane

    // in-plane coordinates from q
    float u = dot(q - o, uhat);
    float v = dot(q - o, khat);

    // in-plane distance^2 + perpendicular^2
    float d2uv   = trapezoidInPlaneD2(vec2(u, v), L, b0, b1);
    float dperp2 = (wn * wn) / n2;
    return d2uv + dperp2;
}
float trapezoidInPlaneD2_fast(vec2 uv, float L, float b0, float b1){
    float eps  = 1e-18;
    float invL = 1.0 / max(L, eps);
    float u    = uv.x;
    float v    = uv.y;
    float t    = clamp(u * invL, 0.0, 1.0);
    float vmax = mix(b0, b1, t);

    // Edge distances (segment distances are already branchless)
    float d2_bottom = segDist2(uv, vec2(0.0, 0.0), vec2(L, 0.0));
    float d2_top    = segDist2(uv, vec2(0.0, b0 ), vec2(L, b1));
    float d2_left   = segDist2(uv, vec2(0.0, 0.0), vec2(0.0, b0));
    float d2_right  = segDist2(uv, vec2(L, 0.0),   vec2(L, b1));
    float d2_edges  = min(min(d2_bottom, d2_top), min(d2_left, d2_right));

    // Inside test via half-space products (branchless)
    float inside_u0 = step(0.0, u);
    float inside_u1 = step(u, L);
    float inside_v0 = step(0.0, v);
    float inside_v1 = step(v, vmax + 1e-7);
    float inside    = inside_u0 * inside_u1 * inside_v0 * inside_v1;

    // If inside -> 0, else -> nearest edge distance
    return mix(d2_edges, 0.0, inside);
}

// Prism distance in a plane spanned by {uhat=e1/|e1|, khat}, no branches
float trapezoidPrismD2Axis_fast(vec3 p, vec3 o, vec3 e1, vec3 khat, float b0, float b1){
    float eps = 1e-18;

    float L2   = dot(e1, e1);
    float invL = inversesqrt(max(L2, eps));
    float L    = L2 * invL;
    vec3  uhat = e1 * invL;

    // Normal to the plane {uhat, khat}; guard with eps to avoid branches
    vec3  n  = cross(khat, uhat);
    float n2 = max(dot(n, n), eps);

    // Orthogonal projection of p onto that plane
    vec3  w   = p - o;
    float wn  = dot(w, n);
    vec3  q   = p - (wn / n2) * n;

    float u = dot(q - o, uhat);
    float v = dot(q - o, khat);

    float d2uv   = trapezoidInPlaneD2_fast(vec2(u, v), L, b0, b1);
    float dperp2 = (wn * wn) / n2;
    return d2uv + dperp2;
}

// ---- public wrappers (unchanged signatures) ----
float udTrapezoidXSq(vec3 p, float xb, float xt, float y0, float y1, float H)
{
    vec3 o  = vec3(xb, 0.0, 0.0);
    vec3 e1 = vec3(xt - xb, 0.0, H);
    // X-trap: plane spanned by {e1, Y}
    return trapezoidPrismD2Axis_fast(p, o, e1, vec3(0.0, 1.0, 0.0), y0, y1);
}

float udTrapezoidYSq(vec3 p, float yb, float yt, float x0, float x1, float H)
{
    vec3 o  = vec3(0.0, yb, 0.0);
    vec3 e1 = vec3(0.0, yt - yb, H);
    // Y-trap: plane spanned by {e1, X}
    return trapezoidPrismD2Axis_fast(p, o, e1, vec3(1.0, 0.0, 0.0), x0, x1);
}

float udLateralUnionFastSq(vec3 p,
                           vec2 size_bottom, vec2 size_top,
                           float r0, float r1, float H)
{
    float xb = size_bottom.x, xt = size_top.x;
    float y0 = max(0.0, size_bottom.y - r0);
    float y1 = max(0.0, size_top.y    - r1);
    float yb = size_bottom.y , yt = size_top.y;
    float x0 = max(0.0, size_bottom.x - r0);
    float x1 = max(0.0, size_top.x    - r1);
    float d2X = udTrapezoidXSq(p, xb, xt, y0, y1, H);
    float d2Y = udTrapezoidYSq(p, yb, yt, x0, x1, H);
    return min(d2X, d2Y);
}""")

SPNewtonSolverSingleStepShader = register_shader_module("""
@name SPNewtonSolverSingleStep
@inputs pos, radius
@outputs dist
@dependencies SPTaperedSupport
@vardeps
float udCanalCornerNewtonSingleStepSq(vec3 p,
                                      vec2 c0, vec2 c1,
                                      float r0, float r1,
                                      float H)
{
    // Precompute linear model along z
    float invH = 1.0 / max(H, 1e-12);
    vec2  A    = (c1 - c0) * invH;   // dc/dz
    float beta = (r1 - r0) * invH;   // dr/dz
    float B    = dot(A, A);

    // Start from clamped z
    float z = clamp(p.z, 0.0, H);

    // ----- One (unrolled) Newton-like step, branchless -----
    vec2  c    = c0 + A * z;
    vec2  w    = p.xy - c;
    float r    = r0 + beta * z;

    float rho2 = dot(w, w);
    float invR = inversesqrt(max(rho2, 1e-14));
    float rho  = rho2 * invR;                     // sqrt(rho2) without extra sqrt
    float wc   = dot(w, A);

    float g     = rho - r;                        // in-plane gap
    float rhop  = -wc * invR;                     // d rho / dz
    float invR3 = invR * invR * invR;
    float rhopp = B * invR - (wc * wc) * invR3;   // d^2 rho / dz^2

    float num   = g * (rhop - beta) - (p.z - z);
    float denom = (rhop - beta)*(rhop - beta) + g * rhopp + 1.0;
    float stepz = num / max(denom, 1e-8);

    // Conservative step clamp to keep stability
    float stepBound = 0.25 * H;
    stepz = clamp(stepz, -stepBound, stepBound);

    // New iterate
    z = clamp(z - stepz, 0.0, H);

    // ----- Evaluate 3 candidates (z, 0, H) branchlessly and pick best -----
    // phi(z)
    vec2  cZ   = c0 + A * z;
    float rZ   = r0 + beta * z;
    float rhoZ = length(p.xy - cZ);
    float phiZ = (rhoZ - rZ)*(rhoZ - rZ) + (p.z - z)*(p.z - z);

    // phi(0)
    float rho0 = length(p.xy - c0);
    float phi0 = (rho0 - r0)*(rho0 - r0) + p.z*p.z;

    // phi(H)
    vec2  cH   = c1;
    float rhoH = length(p.xy - cH);
    float phiH = (rhoH - r1)*(rhoH - r1) + (p.z - H)*(p.z - H);

    // Pick best z (branchless)
    float pickH = step(phiH, min(phiZ, phi0));         // 1 if phiH is the smallest
    float pick0 = step(phi0, min(phiZ, phiH)) * (1.0 - pickH);
    float pickZ = 1.0 - (pick0 + pickH);

    float zb = pickZ * z + pick0 * 0.0 + pickH * H;

    // Return squared distance to circle at zb WITHOUT building a direction
    vec2  cB   = c0 + A * zb;
    float rB   = r0 + beta * zb;
    float rhoB = length(p.xy - cB);
    float dxy  = (rhoB - rB);
    float dz   = (p.z - zb);
    return dxy*dxy + dz*dz;
}""")

SPNewtonSolverMultiStepShader = register_shader_module("""
@name SPNewtonSolverMultiStep
@inputs pos, radius
@outputs dist
@dependencies SPTaperedSupport
@vardeps
const int NEWTON_STEPS = 3;        // tweak if you want more accuracy
float udCanalCornerNewtonMultiStepSq(vec3 p,
                                     vec2 c0, vec2 c1,
                                     float r0, float r1,
                                     float H)
{
    // Linear model along z
    float invH = 1.0 / max(H, 1e-12);
    vec2  A    = (c1 - c0) * invH;    // dc/dz
    float beta = (r1 - r0) * invH;    // dr/dz
    float B    = dot(A, A);

    // Init z on the valid slab
    float z = clamp(p.z, 0.0, H);

    // Multi-step Newton with masked updates (keeps warps coherent)
    float alive = 1.0;                 // 1 = still iterating, 0 = converged
    float tol   = 1e-6 * (1.0 + H);
    float stepClamp = 0.25 * H;

    for (int it = 0; it < NEWTON_STEPS; ++it) {
        // State at z
        vec2  c    = c0 + A * z;
        float r    = r0 + beta * z;
        vec2  w    = p.xy - c;

        // sqrt-free rho
        float rho2 = dot(w, w);
        float invR = inversesqrt(max(rho2, 1e-14));
        float rho  = rho2 * invR;

        float wc    = dot(w, A);
        float g     = rho - r;                          // in-plane gap
        float rhop  = -wc * invR;                       // d rho / dz
        float invR3 = invR * invR * invR;
        float rhopp = B * invR - (wc * wc) * invR3;     // d^2 rho / dz^2

        float num   = g * (rhop - beta) - (p.z - z);
        float denom = (rhop - beta)*(rhop - beta) + g * rhopp + 1.0;
        float stepz = num / max(denom, 1e-8);
        stepz = clamp(stepz, -stepClamp, stepClamp);

        float zNew = clamp(z - stepz, 0.0, H);
        float dz   = abs(zNew - z);

        // Apply update only while alive; once converged, freeze z.
        z     = mix(z, zNew, alive);
        float conv  = 1.0 - step(tol, dz);  // 1 if dz < tol, else 0
        alive *= (1.0 - conv);
    }

    // Evaluate candidates at {z, 0, H} (still squared, sqrt-free for rho)
    vec2  cZ   = c0 + A * z;    float rZ = r0 + beta * z;
    vec2  dZ   = p.xy - cZ;     float dZ2 = dot(dZ, dZ);
    float rhoZ = dZ2 * inversesqrt(max(dZ2, 1e-14));
    float phiZ = (rhoZ - rZ)*(rhoZ - rZ) + (p.z - z)*(p.z - z);

    vec2  d0   = p.xy - c0;     float d0_2 = dot(d0, d0);
    float rho0 = d0_2 * inversesqrt(max(d0_2, 1e-14));
    float phi0 = (rho0 - r0)*(rho0 - r0) + p.z*p.z;

    vec2  cH   = c1;            vec2  dH   = p.xy - cH;
    float dH_2 = dot(dH, dH);   float rhoH = dH_2 * inversesqrt(max(dH_2, 1e-14));
    float phiH = (rhoH - r1)*(rhoH - r1) + (p.z - H)*(p.z - H);

    // Branchless selection of best z ∈ {z, 0, H}
    float minZ0   = min(phiZ, phi0);
    float pick0   = step(phi0, phiZ);                 // 1 if phi0 <= phiZ
    float zbZ0    = mix(z, 0.0, pick0);               // z or 0
    float pickH   = step(phiH, minZ0);                // 1 if phiH <= min(phiZ,phi0)
    float zb      = mix(zbZ0, H, pickH);

    // Final squared distance to the circle locus at zb, no direction build
    vec2  cB   = c0 + A * zb;
    float rB   = r0 + beta * zb;
    vec2  wB   = p.xy - cB;      float wB2 = dot(wB, wB);
    float rhoB = wB2 * inversesqrt(max(wB2, 1e-14));
    float dxy  = (rhoB - rB);
    float dz   = (p.z - zb);
    return dxy*dxy + dz*dz;
}""")

SPTaperedNewtonV1Shader = register_shader_module("""
@name SPTaperedNewtonV1
@inputs pos, radius
@outputs dist
@dependencies SPTaperedSupport, SPNewtonSolverMultiStep
@vardeps 
float SPTaperedNewtonV1(vec3 p, vec3 size, float roundness, float dilate_3d, float scale_opp)
{
    // Fold to first quadrant
    size.xy *= 0.5;
    float H = size.z;

    p.x = abs(p.x);
    p.y = abs(p.y);

    // Bottom/top half-extents and radii
    vec2 size_bottom = size.xy;
    vec2 size_top    = size_bottom * scale_opp;

    float minb = min(size_bottom.x, size_bottom.y);
    float r0   = roundness * minb;
    float r1   = r0 * scale_opp;

    // Centers of the round corner arcs
    vec2 c0 = vec2(size_bottom.x - r0, size_bottom.y - r0);
    vec2 c1 = vec2(size_top.x    - r1, size_top.y    - r1);

    // --- Cheap upper bounds (squared)
    float d2Lat = udLateralUnionFastSq(p, size_bottom, size_top, r0, r1, H);
    float d2Top = udCapRoundedSq(p, H,   size_top,    r1);
    float d2Bot = udCapRoundedSq(p, 0.0, size_bottom, r0);
    float best2 = min(min(d2Lat, d2Top), d2Bot);

    // --- Canal (squared)
    float d2Can = udCanalCornerNewtonMultiStepSq(p, c0, c1, r0, r1, H);

    // Global unsigned: 1 sqrt total
    float d_unsigned = sqrt(min(best2, d2Can));

    // --- Sign (branchless)
    float invH = 1.0 / max(H, 1e-12);
    float t_hat = clamp(p.z * invH, 0.0, 1.0);
    vec2  half_t = mix(size_bottom, size_top, t_hat);
    float r_t    = mix(r0, r1, t_hat);

    // s<0 --> inside slice
    float s_slice = RoundedRectangle2D(p.xy, half_t, r_t);
    float inside_slice = step(s_slice, 0.0);            // 1 if inside

    // 0 < p.z < H
    float inside_z = step(0.0, p.z) * step(p.z, H);

    float inside = inside_slice * inside_z;             // 1 inside, 0 outside
    float signed_d = mix(d_unsigned, -d_unsigned, inside);

    return signed_d - dilate_3d;
}
""")

SPTaperedNewtonV2Shader = register_shader_module("""
@name SPTaperedNewtonV2
@inputs pos, radius
@outputs dist
@dependencies SPTaperedSupport, SPNewtonSolverMultiStep
@vardeps 
float SPTaperedNewtonV2(vec3 p, vec3 size, vec2 roundness, float dilate_3d, vec2 scale_opp)
{
    // Fold to first quadrant
    size.xy *= 0.5;
    float H = size.z;
    p.x = abs(p.x);
    p.y = abs(p.y);

    // Bottom/top half-extents and radii
    vec2 size_bottom = size.xy;
    vec2 size_top    = size_bottom * scale_opp;

    float r0 = roundness.x * min(size_bottom.x, size_bottom.y);
    float r1 = roundness.y * min(size_top.x, size_top.y);

    // Centers of the round corner arcs
    vec2 c0 = vec2(size_bottom.x - r0, size_bottom.y - r0) ;
    vec2 c1 = vec2(size_top.x    - r1, size_top.y    - r1) ;

    // --- Cheap upper bounds (squared)
    float d2Lat = udLateralUnionFastSq(p, size_bottom, size_top, r0, r1, H);
    float d2Top = udCapRoundedSq(p, H,   size_top,    r1);
    float d2Bot = udCapRoundedSq(p, 0.0, size_bottom, r0);
    float best2 = min(min(d2Lat, d2Top), d2Bot);

    // --- Canal (squared)
    float d2Can = udCanalCornerNewtonMultiStepSq(p, c0, c1, r0, r1, H);

    // Global unsigned: 1 sqrt total
    float d_unsigned = sqrt(min(best2, d2Can));

    // --- Sign (branchless)
    float invH = 1.0 / max(H, 1e-12);
    float t_hat = clamp(p.z * invH, 0.0, 1.0);
    vec2  half_t = mix(size_bottom, size_top, t_hat);
    float r_t    = mix(r0, r1, t_hat);

    // s<0 --> inside slice
    float s_slice = RoundedRectangle2D(p.xy, half_t, r_t);
    float inside_slice = step(s_slice, 0.0);            // 1 if inside

    // 0 < p.z < H
    float inside_z = step(0.0, p.z) * step(p.z, H);

    float inside = inside_slice * inside_z;             // 1 inside, 0 outside
    float signed_d = mix(d_unsigned, -d_unsigned, inside);

    return signed_d - dilate_3d;
}""")



SPQuarticSolverShader = register_shader_module("""
@name SPQuarticSolver
@inputs pos, radius
@outputs dist
@dependencies SPTaperedSupport
@vardeps 
// ---------- helpers (same reduced constants) ----------
// Compute reduced constants A,B,lambda,mu,R0 from p,c0,c1,r0,r1,H.
// ====================== Reduced constants ======================
void cc_reduced_constants(in vec3 p, in vec2 c0, in vec2 c1,
                          in float r0, in float r1, in float H,
                          out float A, out float B, out float lambda, out float mu, out float R0)
{
    vec2 s = c1 - c0;
    float S = length(s);
    vec2 e = (S > 0.0) ? s / S : vec2(1.0, 0.0);

    vec2 a = p.xy - c0;
    float alpha = dot(a, e);
    float beta  = length(a - alpha * e);

    lambda = (H > 0.0) ? (S / H) : 0.0;          // center drift per unit z
    mu     = (H > 0.0) ? ((r1 - r0) / H) : 0.0;  // radius drift per unit z

    A  = alpha - lambda * p.z;
    B  = beta;
    R0 = r0 + mu * p.z;
}

// ================== F(y) and g(y) (squared) in one pass ==================
void cc_eval_F_g(float A, float B, float lambda, float mu, float R0, float y,
                 out float F, out float g)
{
    // t = A - lambda*y
    float t   = A - lambda * y;
    float T   = t*t + B*B;                   // = d^2
    float inv = inversesqrt(max(T, 1e-14));  // 1/sqrt(T) (stable)
    float d   = T * inv;                     // = sqrt(T) w/o a standalone sqrt
    float r   = R0 + mu * y;

    // d'(y) = -lambda * t / d
    float dprime_minus_mu = -lambda * t * inv - mu;
    float q = d - r;

    F = q * dprime_minus_mu + y;             // stationarity equation
    g = q*q + y*y;                           // squared objective
}

// =================== Main: squared distance to canal =====================
float udCanalCornerQuarticSq(vec3 p, vec2 c0, vec2 c1, float r0, float r1, float H)
{
    // Degenerate height -> circle in z=0 plane (squared 3D distance to circle)
    if (H <= 1e-6) {
        float q = length(p.xy - c0) - r0;
        return q*q + p.z*p.z;
    }

    float A,B,lambda,mu,R0;
    cc_reduced_constants(p, c0, c1, r0, r1, H, A,B,lambda,mu,R0);

    // y = z* - p.z lives in [ymin, ymax]
    float ymin = -p.z;
    float ymax = H - p.z;

    // ---------- coarse scan: up to 2 brackets + best g ----------
    const int   M      = 12;       // samples
    float ybest        = ymin;
    float gbest        = 1e30;

    float yprev        = ymin;
    float Fprev, gprev;
    cc_eval_F_g(A,B,lambda,mu,R0, yprev, Fprev, gprev);
    gbest = gprev;

    // store up to two brackets
    float br0_lo = 0.0, br0_hi = 0.0;
    float br1_lo = 0.0, br1_hi = 0.0;
    int   nb = 0;

    for (int i=1; i<=M; ++i) {
        float y = mix(ymin, ymax, float(i)/float(M));
        float Fcur, gcur;
        cc_eval_F_g(A,B,lambda,mu,R0, y, Fcur, gcur);

        if (gcur < gbest) { gbest = gcur; ybest = y; }

        // sign change → bracket
        if (nb < 2) {
            if ((Fprev == 0.0) || (Fcur == 0.0) || (Fprev * Fcur < 0.0)) {
                if (nb == 0) { br0_lo = yprev; br0_hi = y; }
                else         { br1_lo = yprev; br1_hi = y; }
                nb++;
            }
        }

        yprev = y;
        Fprev = Fcur;
    }

    // ---------- endpoints (explicitly) ----------
    {
        float F0, g0; cc_eval_F_g(A,B,lambda,mu,R0, ymin, F0, g0);
        if (g0 < gbest) { gbest = g0; ybest = ymin; }
        float F1, g1; cc_eval_F_g(A,B,lambda,mu,R0, ymax, F1, g1);
        if (g1 < gbest) { gbest = g1; ybest = ymax; }
    }

    // ---------- refine each bracket by bisection ----------
    for (int k=0; k<2; ++k) {
        if (k >= nb) break;

        float lo = (k==0) ? br0_lo : br1_lo;
        float hi = (k==0) ? br0_hi : br1_hi;
        float Flo, glo; cc_eval_F_g(A,B,lambda,mu,R0, lo, Flo, glo);
        float Fhi, ghi; cc_eval_F_g(A,B,lambda,mu,R0, hi, Fhi, ghi);

        // Guard invalid bracket (shouldn't happen, but be safe)
        if (!(Flo * Fhi <= 0.0)) {
            continue;
        }

        for (int it=0; it<22; ++it) {
            float mid = 0.5*(lo + hi);
            float Fm, gm; cc_eval_F_g(A,B,lambda,mu,R0, mid, Fm, gm);

            if (abs(Fm) < 1e-10 || abs(hi - lo) < 1e-6*(1.0 + abs(mid))) {
                // mid is our root approximation; evaluate g and exit
                if (gm < gbest) { gbest = gm; ybest = mid; }
                break;
            }

            if (Flo * Fm < 0.0) { hi = mid; Fhi = Fm; }
            else                { lo = mid; Flo = Fm; }
        }

        // also check the final midpoint just in case loop exited by width
        float y = 0.5*(lo + hi);
        float Fy, gy; cc_eval_F_g(A,B,lambda,mu,R0, y, Fy, gy);
        if (gy < gbest) { gbest = gy; ybest = y; }
    }

    // ---------- if no bracket, stick with best coarse sample ----------
    // (For most configurations, true minimizer has F=0; otherwise it's at an endpoint already handled.)

    return max(gbest, 0.0);
}""")

SPTaperedQuarticV1Shader= register_shader_module("""
@name SPTaperedQuarticV1
@inputs pos, radius
@outputs dist
@dependencies SPTaperedSupport, SPQuarticSolver
@vardeps 
float SPTaperedQuarticV1(vec3 p, vec3 size, float roundness, float dilate_3d, float scale_opp)
{
    // Fold to first quadrant
    size.xy *= 0.5;
    float H = size.z;
    p.x = abs(p.x);
    p.y = abs(p.y);

    // Bottom/top half-extents and radii
    vec2 size_bottom = size.xy;
    vec2 size_top    = size_bottom * scale_opp;

    float r0 = roundness * min(size_bottom.x, size_bottom.y);
    float r1 = r0 * scale_opp;
    //r0 = min(r0, min(size_bottom.x, size_bottom.y));
    //r1 = min(r1, min(size_top.x,    size_top.y));

    // Centers of the round corner arcs
    vec2 c0 = vec2(size_bottom.x - r0, size_bottom.y - r0) ;
    vec2 c1 = vec2(size_top.x    - r1, size_top.y    - r1) ;

    // Evaluate CHEAP pieces first (squared), to get a good upper bound.
    float d2Lat = udLateralUnionFastSq(p, size_bottom, size_top, r0, r1, H);
    float d2Top = udCapRoundedSq(p, H,   size_top,    r1);
    float d2Bot = udCapRoundedSq(p, 0.0, size_bottom, r0);

    float best2 = min(d2Lat, min(d2Top, d2Bot));  // running upper bound (squared)

    // Canal (squared). If you modify your solver, pass best2 to prune internally.
    float d2Can = udCanalCornerQuarticSq(p, c0, c1, r0, r1, H);

    // Global unsigned (one sqrt total)
    float d_unsigned = sqrt(min(best2, d2Can));

    // Sign from slice AND z-range (closed solid)
    float t_hat  = (H > 0.0) ? clamp(p.z / H, 0.0, 1.0) : 0.0;
    vec2  half_t = mix(size_bottom, size_top, t_hat);
    float r_t    = mix(r0, r1, t_hat);
    bool  inside_slice = (RoundedRectangle2D(p.xy, half_t, r_t) < 0.0);
    bool  inside_z     = (p.z > 0.0 && p.z < H);
    float sdf = (inside_slice && inside_z) ? -d_unsigned : d_unsigned;
    return sdf - dilate_3d;
}""")

SPTaperedApproxHelpers = register_shader_module("""
@name SPTaperedApproxHelpers
@inputs pos, radius
@outputs dist
@dependencies SPTaperedSupport, SPNewtonSolverSingleStep
@vardeps 

bool canalBand(vec3 p, vec2 hb, vec2 ht, float r0, float r1, float H)
{
    if (p.z < 0.0 || p.z > H) return false;
    float xMin = min(hb.x - r0, ht.x - r1);
    float yMin = min(hb.y - r0, ht.y - r1);
    return (p.x >= xMin - 1e-6) && (p.y >= yMin - 1e-6);
}

float udLateralOneSideSq(vec3 p,
                         vec2 size_bottom, vec2 size_top,
                         float r0, float r1, float H)
{
    // HALF extents + flat spans in HALF units
    float xb = size_bottom.x, xt = size_top.x;
    float yb = size_bottom.y, yt = size_top.y;
    float x0 = max(0.0, xb - r0),    x1 = max(0.0, xt - r1);
    float y0 = max(0.0, yb - r0),    y1 = max(0.0, yt - r1);

    // Decide side via top diagonal: y : x compared to (yt : xt)
    bool useY = (p.y * xt > p.x * yt); // above diagonal => Y-side

    float d2 = useY
             ? udTrapezoidYSq(p, yb, yt, x0, x1, H)
             : udTrapezoidXSq(p, xb, xt, y0, y1, H);

    // Optional safety (rarely triggers, keeps one-side fast but guards near bisector)
    // If you want it, uncomment:
    float d2_other_lb;
    if (useY) {
        vec3 nu = vec3(-H, 0.0, xt - xb); // X-side plane unnormalized normal
        float n2 = dot(nu,nu);
        float dperp2 = (n2>0.0) ? sq(dot(p - vec3(xb,0,0), nu))/n2 : 0.0;
        d2_other_lb = dperp2;
        if (d2_other_lb < d2) d2 = min(d2, udTrapezoidXSq(p, xb, xt, y0, y1, H));
    } else {
        vec3 nu = vec3(0.0, -H, yt - yb);
        float n2 = dot(nu,nu);
        float dperp2 = (n2>0.0) ? sq(dot(p - vec3(0,yb,0), nu))/n2 : 0.0;
        d2_other_lb = dperp2;
        if (d2_other_lb < d2) d2 = min(d2, udTrapezoidYSq(p, yb, yt, x0, x1, H));
    }

    return d2;
}
// Only pay the RoundedRectangle2D cost when it can still win
float udCapRoundedSqPruned(vec3 p, float zPlane, vec2 sizeFull, float r, float best2)
{
    float dz  = p.z - zPlane;
    float dz2 = dz*dz;

    // Lower bound: cap distance >= |dz|
    if (dz2 >= best2) return 1e30;  // cannot beat current best

    // Need the true outside-in-plane part
    float s2   = RoundedRectangle2D(p.xy, sizeFull, r); // signed
    float out2 = max(s2, 0.0);
    return out2*out2 + dz2;
}
bool insideRoundedRect_NoSqrt(vec2 p, vec2 sizeFull, float r)
{
    // Reflect to first quadrant
    vec2 pa = abs(p);
    vec2 h  = sizeFull;        // half extents
    vec2 m  = h - vec2(r);           // start of corner quarter-circle

    // Inside the middle rectangle?
    if (pa.x <= m.x && pa.y <= m.y) return true;

    // Outside the AABB?
    if (pa.x >= h.x || pa.y >= h.y) return false;

    // Corner test (circle of radius r): compare squared dists, no sqrt
    vec2 d = max(pa - m, vec2(0.0)); // only positive in the corner wedge
    return dot(d, d) <= r*r + 1e-12; // tiny bias for robust boundary ownership
}""")

SPTaperedApproxV1Shader = register_shader_module("""
@name SPTaperedApproxV1
@inputs pos, radius
@outputs dist
@dependencies SPTaperedSupport, SPNewtonSolverSingleStep, SPTaperedApproxHelpers
@vardeps 
float SPTaperedApproxV1(vec3 p, vec3 size, float roundness, float dilate_3d, float scale_opp)
{
    // Fold to first quadrant
    size.xy *= 0.5;
    float H = size.z;
    p.x = abs(p.x);
    p.y = abs(p.y);

    vec2 size_bottom = size.xy;                 // FULL extents
    vec2 size_top    = size_bottom * scale_opp;

    float r0 = roundness * min(size_bottom.x, size_bottom.y);
    float r1 = r0 * scale_opp;

    // Canal centers (HALF extents - r)
    vec2 c0 = vec2(size_bottom.x - r0, size_bottom.y - r0);
    vec2 c1 = vec2(size_top.x    - r1, size_top.y    - r1);

    // --- Cheap upper bounds (squared)
    float d2Lat = udLateralUnionFastSq(p, size_bottom, size_top, r0, r1, H);
    float d2Top = udCapRoundedSq(p, H,   size_top,    r1);
    float d2Bot = udCapRoundedSq(p, 0.0, size_bottom, r0);
    float best2 = min(min(d2Lat, d2Top), d2Bot);

    // --- Canal (squared)
    float d2Can = 1e30;
    if (canalBand(p, size_bottom, size_top, r0, r1, H) && best2 > 1e-8) {
        d2Can = udCanalCornerNewtonSingleStepSq(p, c0, c1, r0, r1, H);
        // Optional: tiny safety for sphere tracing blends
        // d2Can *= 0.98;
    }

    // Global unsigned: 1 sqrt total
    float d_unsigned = sqrt(min(best2, d2Can));

    // --- Sign (branchless)
    float invH = 1.0 / max(H, 1e-12);
    float t_hat = clamp(p.z * invH, 0.0, 1.0);
    vec2  half_t = mix(size_bottom, size_top, t_hat);
    float r_t    = mix(r0, r1, t_hat);

    // s<0 --> inside slice
    float s_slice = RoundedRectangle2D(p.xy, half_t, r_t);
    float inside_slice = step(s_slice, 0.0);            // 1 if inside

    // 0 < p.z < H
    float inside_z = step(0.0, p.z) * step(p.z, H);

    float inside = inside_slice * inside_z;             // 1 inside, 0 outside
    float signed_d = mix(d_unsigned, -d_unsigned, inside);

    return signed_d - dilate_3d;
}""")

SPTaperedApproxV2Shader = register_shader_module("""
@name SPTaperedApproxV2
@inputs pos, radius
@outputs dist
@dependencies SPTaperedSupport, SPNewtonSolverSingleStep, SPTaperedApproxHelpers
@vardeps 
float SPTaperedApproxV2(vec3 p, vec3 size, vec2 roundness, float dilate_3d, vec2 scale_opp)
{
    // Fold to first quadrant
    size.xy *= 0.5;
    float H = size.z;
    p.x = abs(p.x);
    p.y = abs(p.y);

    // Bottom/top half-extents and radii
    vec2 size_bottom = size.xy;
    vec2 size_top    = size_bottom * scale_opp;

    float r0 = roundness.x * min(size_bottom.x, size_bottom.y);
    float r1 = roundness.y * min(size_top.x, size_top.y);

    // Centers of the round corner arcs
    vec2 c0 = vec2(size_bottom.x - r0, size_bottom.y - r0) ;
    vec2 c1 = vec2(size_top.x    - r1, size_top.y    - r1) ;

    // --- Cheap upper bounds (squared)
    float d2Lat = udLateralUnionFastSq(p, size_bottom, size_top, r0, r1, H);
    float d2Top = udCapRoundedSq(p, H,   size_top,    r1);
    float d2Bot = udCapRoundedSq(p, 0.0, size_bottom, r0);
    float best2 = min(min(d2Lat, d2Top), d2Bot);

    // --- Canal (squared)
    float d2Can = 1e30;
    if (canalBand(p, size_bottom, size_top, r0, r1, H) && best2 > 1e-8) {
        d2Can = udCanalCornerNewtonSingleStepSq(p, c0, c1, r0, r1, H);
        // Optional: tiny safety for sphere tracing blends
        // d2Can *= 0.98;
    }

    // Global unsigned: 1 sqrt total
    float d_unsigned = sqrt(min(best2, d2Can));

    // --- Sign (branchless)
    float invH = 1.0 / max(H, 1e-12);
    float t_hat = clamp(p.z * invH, 0.0, 1.0);
    vec2  half_t = mix(size_bottom, size_top, t_hat);
    float r_t    = mix(r0, r1, t_hat);

    // s<0 --> inside slice
    float s_slice = RoundedRectangle2D(p.xy, half_t, r_t);
    float inside_slice = step(s_slice, 0.0);            // 1 if inside

    // 0 < p.z < H
    float inside_z = step(0.0, p.z) * step(p.z, H);

    float inside = inside_slice * inside_z;             // 1 inside, 0 outside
    float signed_d = mix(d_unsigned, -d_unsigned, inside);

    return signed_d - dilate_3d;
}
""")


SPChamferedV1Shader = register_shader_module("""
@name SPChamferedV1
@inputs pos, radius
@outputs dist
@dependencies SPBase
@vardeps 
float sdOctahedron( in vec3 p, in float s)// by Iq 2019
{
    p = abs(p);
    float m = p.x+p.y+p.z-s;
    vec3 q;
         if( 3.0*p.x < m ) q = p.xyz;
    else if( 3.0*p.y < m ) q = p.yzx;
    else if( 3.0*p.z < m ) q = p.zxy;
    else return m*0.57735027;
    
    float k = clamp(0.5*(q.z-q.y+s),0.0,s); 
    return length(vec3(q.x,q.y-s+k,q.z-k)); 
}

float SPChamferedV1(vec3 op, vec3 size, float ch, float dilate_3d) {
    vec3 p = abs(op) + vec3(ch);
    p = max(vec3(0), p - size); 
    float d = sdOctahedron(p, ch);
    return d - dilate_3d;
}""")

SPChamferedV2Shader = register_shader_module("""
@name SPChamferedV2
@inputs pos, radius
@outputs dist
@dependencies SPBase
@vardeps 

// ---------- small utils ----------
float dot2(vec2 v){ return dot(v,v); }
float dot2(vec3 v){ return dot(v,v); }
float sq(float x){ return x*x; }
const float EPS = 1e-6;

// ---------- 2D chamfered box SDF (expects FULL extents 'size') ----------
float sdChamferBox( in vec2 p, in vec2 size, in float chamfer )
{
    vec2 b = size * 0.5;
    p = abs(p) - b;
    p = (p.y > p.x) ? p.yx : p.xy;
    p.y += chamfer;
    const float k = 1.0 - sqrt(2.0);
    if (p.y < 0.0 && p.y + p.x * k < 0.0) return p.x;
    if (p.x < p.y) return (p.x + p.y) * sqrt(0.5);
    return length(p);
}

// ---------- cap distance (squared), branchless ----------
float udCapChamferedSq(vec3 p, float zPlane, vec2 sizeFull, float chamfer)
{
    float s2 = sdChamferBox(p.xy, sizeFull, chamfer); // signed 2D
    float out2D = max(s2, 0.0);
    float dz    = p.z - zPlane;
    return out2D*out2D + dz*dz;
}

// ---------- 2D segment distance (squared) ----------
float dist2Segment2(vec2 p, vec2 a, vec2 b) {
    vec2 ab = b - a;
    float t = clamp(dot(p - a, ab) / max(dot(ab,ab), 1e-12), 0.0, 1.0);
    vec2 q = a + t * ab;
    return dot2(p - q);
}

// ---------- LATERALS (planar trapezoids), HALF-extents inside ----------
// X-side: (xb,0,0)->(xb,y0,0)->(xt,y1,H)->(xt,0,H)
float udTrapezoidXSq_chamf(vec3 p, float xb, float xt, float y0, float y1, float H)
{
    if (max(y0, y1) <= EPS) return 1e30;

    vec3 o  = vec3(xb, 0.0, 0.0);
    vec3 e1 = vec3(xt - xb, 0.0, H);
    float L2 = dot(e1,e1);
    if (L2 < 1e-18) return 1e30;
    float invL = inversesqrt(L2);
    float L    = L2 * invL;
    vec3 b1    = e1 * invL;
    vec3 nu    = vec3(-H, 0.0, xt - xb); // unnormalized normal

    vec3 w     = p - o;
    float n2   = dot(nu,nu);
    float dperp2 = (n2 > 0.0) ? sq(dot(w,nu))/n2 : 0.0;

    float u = dot(w, b1);
    float v = p.y;

    float t  = clamp(u / L, 0.0, 1.0);
    float vmax = y0 + (y1 - y0) * t;

    bool inside = (u >= -EPS) && (u <= L + EPS) && (v >= -EPS) && (v <= vmax + EPS);
    if (inside) return dperp2;

    vec2 P = vec2(u, v);
    float d2 = dist2Segment2(P, vec2(0.0,0.0), vec2(L,0.0));
    d2 = min(d2, dist2Segment2(P, vec2(0.0,0.0), vec2(0.0,y0)));
    d2 = min(d2, dist2Segment2(P, vec2(L,0.0), vec2(L,y1)));
    d2 = min(d2, dist2Segment2(P, vec2(0.0,y0), vec2(L,y1)));

    return d2 + dperp2;
}

// Y-side: (0,yb,0)->(x0,yb,0)->(x1,yt,H)->(0,yt,H)
float udTrapezoidYSq_chamf(vec3 p, float yb, float yt, float x0, float x1, float H)
{
    if (max(x0, x1) <= EPS) return 1e30;

    vec3 o  = vec3(0.0, yb, 0.0);
    vec3 e1 = vec3(0.0, yt - yb, H);
    float L2 = dot(e1,e1);
    if (L2 < 1e-18) return 1e30;
    float invL = inversesqrt(L2);
    float L    = L2 * invL;
    vec3 b1    = e1 * invL;
    vec3 nu    = vec3(0.0, -H, yt - yb);

    vec3 w     = p - o;
    float n2   = dot(nu,nu);
    float dperp2 = (n2 > 0.0) ? sq(dot(w,nu))/n2 : 0.0;

    float u = dot(w, b1);
    float v = p.x - o.x;

    float t  = clamp(u / L, 0.0, 1.0);
    float vmax = x0 + (x1 - x0) * t;

    bool inside = (u >= -EPS) && (u <= L + EPS) && (v >= -EPS) && (v <= vmax + EPS);
    if (inside) return dperp2;

    vec2 P = vec2(u, v);
    float d2 = dist2Segment2(P, vec2(0.0,0.0), vec2(L,0.0));
    d2 = min(d2, dist2Segment2(P, vec2(0.0,0.0), vec2(0.0,x0)));
    d2 = min(d2, dist2Segment2(P, vec2(L,0.0), vec2(L,x1)));
    d2 = min(d2, dist2Segment2(P, vec2(0.0,x0), vec2(L,x1)));

    return d2 + dperp2;
}

// Union of laterals for CHAMFER (you pass FULL extents; we convert to half)
float udLateralUnionFastSqChamfer(vec3 p,
                                  vec2 size_bottom_full, vec2 size_top_full,
                                  float c0, float c1, float H)
{
    float xb = 0.5 * size_bottom_full.x, xt = 0.5 * size_top_full.x;
    float y0 = max(0.0, 0.5 * size_bottom_full.y - c0);
    float y1 = max(0.0, 0.5 * size_top_full.y    - c1);

    float yb = 0.5 * size_bottom_full.y, yt = 0.5 * size_top_full.y;
    float x0 = max(0.0, 0.5 * size_bottom_full.x - c0);
    float x1 = max(0.0, 0.5 * size_top_full.x    - c1);

    float d2X = udTrapezoidXSq_chamf(p, xb, xt, y0, y1, H);
    float d2Y = udTrapezoidYSq_chamf(p, yb, yt, x0, x1, H);
    return min(d2X, d2Y);
}

// ---------- CHAMFER CORNER (planar trapezoid) distance (squared) ----------
// Vertices (first quadrant):
//   B1 = (hb.x, hb.y - c0, 0),  B2 = (hb.x - c0, hb.y, 0)
//   T1 = (ht.x, ht.y - c1, H),  T2 = (ht.x - c1, ht.y, H)
float udChamferCornerSq(vec3 p, vec2 halfB, vec2 halfT, float c0, float c1, float H)
{
    if (max(c0, c1) <= EPS) return 1e30;

    vec3 B1 = vec3(halfB.x,           halfB.y - c0, 0.0);
    vec3 B2 = vec3(halfB.x - c0,      halfB.y,      0.0);
    vec3 T1 = vec3(halfT.x,           halfT.y - c1, H);
    vec3 T2 = vec3(halfT.x - c1,      halfT.y,      H);

    // Plane basis: b1 along E1 = T1 - B1; b2 orthonormalized from E2 = B2 - B1
    vec3 o  = B1;
    vec3 E1 = T1 - B1;
    vec3 E2b= B2 - B1;                // bottom top-edge direction
    vec3 nU = cross(E1, E2b);         // UNnormalized plane normal

    float L1_2 = dot(E1,E1);
    if (L1_2 < 1e-18) return 1e30;    // degenerate
    float invL1 = inversesqrt(L1_2);
    float L1    = L1_2 * invL1;
    vec3  b1    = E1 * invL1;

    // Orthonormal second axis in plane
    vec3 E2b_proj = E2b - b1 * dot(E2b, b1);
    float v0 = length(E2b_proj);
    if (v0 < 1e-18) return 1e30;
    vec3  b2 = E2b_proj / v0;

    // Top span in same b2 direction
    vec3 E2t     = T2 - T1;
    float v1     = length(E2t - b1 * dot(E2t, b1));  // projected length

    // Signed perpendicular distance to the plane (squared, unnormalized normal)
    vec3 w   = p - o;
    float n2 = dot(nU,nU);
    float dperp2 = (n2 > 0.0) ? sq(dot(w, nU)) / n2 : 0.0;

    // In-plane coordinates
    float u = dot(w, b1);   // along E1
    float v = dot(w, b2);   // along chamfer direction (in-plane)

    float t  = clamp(u / L1, 0.0, 1.0);
    float vmax = v0 + (v1 - v0) * t;

    bool inside = (u >= -EPS) && (u <= L1 + EPS) && (v >= -EPS) && (v <= vmax + EPS);
    if (inside) return dperp2;

    // 2D distance to trapezoid edges in (u,v)
    vec2 P = vec2(u, v);
    float d2 = dist2Segment2(P, vec2(0.0,0.0), vec2(L1,0.0));   // longitudinal bottom edge (B1->T1)
    d2 = min(d2, dist2Segment2(P, vec2(0.0,0.0), vec2(0.0,v0))); // left (B1->B2)
    d2 = min(d2, dist2Segment2(P, vec2(L1,0.0), vec2(L1,v1)));   // right (T1->T2)
    d2 = min(d2, dist2Segment2(P, vec2(0.0,v0), vec2(L1,v1)));   // slanted top (B2->T2)

    return d2 + dperp2;
}

// ---------- Main: chamfered variant (no canal/Newton) ----------
float SPChamferedV2(vec3 p, vec3 size, vec2 roundness, float dilate_3d, vec2 scale_opp)
{
    // Fold to first quadrant
    float H = size.z;
    p.x = abs(p.x);
    p.y = abs(p.y);

    // FULL extents at bottom/top
    vec2 size_bottom = size.xy;
    vec2 size_top    = size_bottom * scale_opp;

    // Chamfer lengths (world units). Same mapping as radius: fraction of HALF min-dim.
    float c0 = roundness.x * 0.5 * min(size_bottom.x, size_bottom.y);
    float c1 = roundness.y * 0.5 * min(size_top.x,    size_top.y);

    // HALF extents (for corner/planes)
    vec2 halfB = 0.5 * size_bottom;
    vec2 halfT = 0.5 * size_top;

    // 1) Laterals (squared)
    float d2Lat = udLateralUnionFastSqChamfer(p, size_bottom, size_top, c0, c1, H);

    // 2) Caps (squared)
    float d2Top = udCapChamferedSq(p, H,   size_top,    c1);
    float d2Bot = udCapChamferedSq(p, 0.0, size_bottom, c0);

    // 3) Chamfer corner (squared)
    float d2Can = udChamferCornerSq(p, halfB, halfT, c0, c1, H);

    // Global unsigned (one sqrt)
    float d_unsigned = sqrt(min(d2Lat, min(d2Top, min(d2Bot, d2Can))));

    // Sign from slice SDF of CHAMFER (consistent!)
    float t_hat  = (H > 0.0) ? clamp(p.z / H, 0.0, 1.0) : 0.0;
    vec2  size_t = mix(size_bottom, size_top, t_hat);  // FULL extents
    float c_t    = mix(c0, c1, t_hat);
    bool  inside_slice = (sdChamferBox(p.xy, size_t, c_t) < 0.0);
    bool  inside_z     = (p.z > 0.0 && p.z < H);

    float sdf = (inside_slice && inside_z) ? -d_unsigned : d_unsigned;
    return sdf - dilate_3d;
}""")



