"""
Basic Supported Primitives.
"""
import numpy as np
from sysl.shader.shader_module import register_shader_module
from sysl.shader.shader_templates.common import CONSTANTS

CONSTANTS.update({
    "PI": ("float", np.pi),
})

SPBaseShader = register_shader_module("""
@name SPBase
@inputs none
@outputs none
@dependencies 
float HalfRoundedRectangle2D( in vec2 p, in vec2 b, in float r )
{
    vec2 size = b / 2.0;
    vec2 q = abs(p) - size +r;
    return min(max(q.x,q.y),0.0) + length(max(q,0.0)) - r;
}
float HalfRoundedRectangle2D( in vec2 p, in vec2 b, in vec4 r )
{
    vec2 size = b / 2.0;
    r.xy = (p.x>0.0)?r.xy : r.zw;
    r.x  = (p.y>0.0)?r.x  : r.y;
    vec2 q = abs(p) - size + r.x;
    return min(max(q.x,q.y),0.0) + length(max(q,0.0)) - r.x;
}
float RoundedRectangle2D( in vec2 p, in vec2 size, in float r )
{
    vec2 q = abs(p) - size +r;
    return min(max(q.x,q.y),0.0) + length(max(q,0.0)) - r;
}
float RoundedRectangle2D( in vec2 p, in vec2 size, in vec4 r )
{
    r.xy = (p.x>0.0)?r.xy : r.zw;
    r.x  = (p.y>0.0)?r.x  : r.y;
    vec2 q = abs(p) - size + r.x;
    return min(max(q.x,q.y),0.0) + length(max(q,0.0)) - r.x;
}
""")

SPTaperedOnionShader = register_shader_module("""
@name SPTaperedOnion
@inputs p, size, roundness, dilate_3d, scale, onion_ratio
@outputs sd
@dependencies SPBase
@vardeps

/* 2d determinant (aka cross2d) */
float det( in vec2 a, in vec2 b ) { return a.x*b.y - a.y*b.x; }

/* Modified from iq's: https://www.shadertoy.com/view/3tdSDj
   Returns signed distance to segment and also the side (left/right = neg/pos) */
vec2 sdSegmentWithSign( in vec2 p, in vec2 a, in vec2 b )
{
	vec2 pa = p - a;
	vec2 ba = b - a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return vec2(length( pa - ba*h ), det(pa, ba));
}

/* Signed distance to non-intersecting, convex quad. */
float sdQuad(in vec2 pos, in vec2 p0, in vec2 p1, in vec2 p2, in vec2 p3)
{
    vec2 sd0 = sdSegmentWithSign(pos, p0, p1);
    vec2 sd1 = sdSegmentWithSign(pos, p1, p2);
    vec2 sd2 = sdSegmentWithSign(pos, p2, p3);
    vec2 sd3 = sdSegmentWithSign(pos, p3, p0);
    float sd = min(sd0.x, min(sd1.x, min(sd2.x, sd3.x)));
    
    /* Point tests to the left of all segments. */
    /* Can probably do something more clever here :) */
    if (sd0.y < 0.0f && sd1.y < 0.0f && sd2.y < 0.0f && sd3.y < 0.0f)
      sd = -sd;
    
    return sd;
}
float cross2D(vec2 a, vec2 b) { return a.x*b.y - a.y*b.x; }
float sdTaperTrapezoidOnionExact(vec2 p, float inner, float h, float x3, float onion_ratio)
{
    // Geometry
    float xL = -inner * (1.0 - onion_ratio);
    float xTL = -inner + (x3 + inner) * onion_ratio;
    float yB = -h, yT =  h;
    vec2  Lb = vec2(xL, yB), Lt = vec2(xTL, yT);
    vec2  Rb = vec2(0.0, yB), Rt = vec2(x3,  yT);
    vec2  eS = Rt - Rb;                     // slanted edge vector = (x3, 2h)
    vec2  eS_L = Lt - Lb;                     // slanted edge vector = (x3, 2h)
    float inv_e2 = 1.0 / dot(eS, eS);       // precompute once
    float inv_e2_L = 1.0 / dot(eS_L, eS_L);

    // --- squared distances to the 4 segments (valid both inside and outside)
    // Left vertical

    vec2  pa_L        = p - Lb;
    float t_L         = clamp(dot(pa_L, eS_L) * inv_e2_L, 0.0, 1.0);
    vec2  q_slant_L   = Lb + eS_L * t_L;
    float d2_slant_L  = dot(p - q_slant_L, p - q_slant_L);

    // Bottom horizontal
    float x_cl_bot  = clamp(p.x, xL, 0.0);
    vec2  q_bot     = vec2(x_cl_bot, yB);
    float d2_bottom = dot(p - q_bot, p - q_bot);

    // Top horizontal  (NOTE: xL <= x3; use min/max to be robust if not)
    float x_min_top = xTL;
    float x_max_top = x3;
    float x_cl_top  = clamp(p.x, x_min_top, x_max_top);
    vec2  q_top     = vec2(x_cl_top, yT);
    float d2_top    = dot(p - q_top, p - q_top);

    // Slanted segment
    vec2  pa        = p - Rb;
    float t         = clamp(dot(pa, eS) * inv_e2, 0.0, 1.0);
    vec2  q_slant   = Rb + eS * t;
    float d2_slant  = dot(p - q_slant, p - q_slant);

    float d2 = min(min(d2_slant_L, d2_bottom), min(d2_top, d2_slant));

    // --- sign from half-planes (<= 0 means inside)
    float cL_L = eS_L.y*(p.x - Lb.x) - eS_L.x*(p.y - Lb.y); // det(p-Lb, eS_L)
    cL_L = - cL_L;
    float cB = yB - p.y;                 // bottom
    float cT = p.y - yT;                 // top
    float cS = eS.y*(p.x - Rb.x) - eS.x*(p.y - Rb.y); // det(p-Rb, eS)

    // For this geometry, interior satisfies cL<=0, cB<=0, cT<=0, cS<=0
    bool inside = max(max(cL_L, cB), max(cT, cS)) <= 0.0;

    float d = sqrt(d2);
    return inside ? -d : d;
}

float SPTaperedOnion(vec3 p, vec3 size, float roundness, float dilate_3d, float scale, float onion_ratio)
{
    // first get 2D sdf w.r.t. rounded box
    vec2 uv = p.xy;
    float min_size = min(size.x, size.y)  * 0.5;
    float r = roundness * min_size;
    float sdf2d  = HalfRoundedRectangle2D(uv, size.xy, r);
    // float onion_amount = (1.0 - onion_ratio) * min_size;
    // sdf2d = (sdf2d < 0.0)? abs(sdf2d) - onion_amount : sdf2d;
    // now get quad w.r.t. 
    vec2 pos_2d = vec2(sdf2d, p.z);
    
    // float inner_deep  = 0.5 * min(size.x, size.y); // x extent of left edge
    float half_height = 0.5 * size.z;              // y extent

    float x3     = -(1.0 - scale) * min_size;
    float sd = sdTaperTrapezoidOnionExact(pos_2d, min_size, half_height, x3, onion_ratio);
    return sd - dilate_3d;
}""")

CONSTANTS.update({
    "BULGE_EPS": ("float", 1e-5),
})

SuperFrustumShader = register_shader_module("""
@name SuperFrustum
@inputs pos, radius
@outputs dist
@dependencies SPTaperedOnion
@vardeps PI, BULGE_EPS
// --- constants ---

// rotate 90° CCW
vec2 rot90(vec2 v){ return vec2(-v.y, v.x); }

vec2 mapArcBulge(vec2 p, float z, float bulge)
{
    float half_z = 0.5 * z;
    p.x = p.x * sign(bulge);
    float theta_top = max(abs(bulge) * (PI * 0.5), BULGE_EPS);
    //float theta_top = bulge * (PI * 0.5);

    float center_pos = half_z / tan(theta_top);
    vec2  center = vec2(center_pos, 0.0);
    float radius = sqrt(center_pos * center_pos + half_z * half_z);

    // angle wrt arc center
    float point_angle = atan(p.y, center_pos - p.x);

    // inside the arc span
    float new_y = clamp(point_angle / theta_top, -1.0, 1.0) * half_z;
    float new_x = length(p - center) - radius;
    vec2  inside_point = vec2(new_x, new_y);

    // endpoints and correct tangents/normals
    float s = sin(theta_top), c = cos(theta_top);

    // top ( +z/2 )
    vec2 t_top = normalize(vec2(s,  c));
    vec2 n_top = vec2(-t_top.y, t_top.x);
    vec2 end_top = vec2(0.0,  half_z);
    float along_top = dot(p - end_top, t_top);
    float perp_top  = dot(p - end_top, n_top);
    vec2 above_point = vec2(perp_top, half_z + along_top);

    // bottom ( -z/2 )  — note t_bot != -t_top
    vec2 t_bot = normalize(vec2(-s,  c));
    vec2 n_bot = vec2(-t_bot.y, t_bot.x);
    vec2 end_bot = vec2(0.0, -half_z);
    float along_bot = dot(p - end_bot, t_bot);
    float perp_bot  = dot(p - end_bot, n_bot);
    vec2 below_point = vec2(perp_bot, -half_z + along_bot);

    // region masks
    float mask_above = step(theta_top, point_angle);     // angle >  +theta
    float mask_below = step(point_angle, -theta_top);    // angle <  -theta

    vec2 new_point = mix(inside_point, above_point, mask_above);
    new_point = mix(new_point, below_point, mask_below);
    new_point.x = new_point.x * sign(bulge);
    return new_point;
}

float SuperFrustum(vec3 p, vec3 size, float roundness, float dilate_3d, float scale, float bulge, float onion_ratio)
{   
    vec3 new_p = p;
    if (bulge != 0.0)
    {
        vec2 new_xz = mapArcBulge(p.xz, size.z, bulge);
        new_p = vec3(new_xz.x, p.y, new_xz.y);
    }
    float sd = SPTaperedOnion(new_p, size, roundness, dilate_3d, scale, onion_ratio);
    return sd;
}""")

SolidSFShader = register_shader_module("""
@name SolidSF
@inputs pos, size, params, onion_ratio
@outputs dist
@dependencies SuperFrustum
@vardeps

float SolidSF(vec3 p, vec3 size, float roundness, float dilate_3d, float scale, float bulge, float onion_ratio, vec4 prob)
{   
    float w_cube = prob.x;
    float w_sphere = prob.y;
    float w_cylinder = prob.z;
    float w_cone = prob.w;

    float size_xy = (size.x + size.y)/2.0;
    vec3 size_cyl = vec3(size_xy, size_xy, size.z);
    vec3 new_size = (w_cube) * size +  (w_cylinder + w_cone) * size_cyl;
    vec3 new_roundness = (w_cylinder + w_cone);
    float new_dilate_3d = w_sphere * dilate_3d;
    float new_scale = (w_cube + w_sphere + w_cylinder);
    float new_bulge = 0 * bulge;
    float new_onion_ratio = (w_cube + w_cylinder + w_cone) * onion_ratio;
    return SuperFrustum(p, new_size, new_roundness, new_dilate_3d, new_scale, new_bulge, new_onion_ratio);
}""")

CONSTANTS.update({
    "ON_EPS": ("float", 1e-8),
})


SPProtoShader = register_shader_module("""
@name SPProto
@inputs pos, size, roundness, dilate_3d, onion_ratio, extrussion
@outputs dist
@dependencies SPBase
@vardeps ON_EPS

float SPProto(vec3 p, vec3 size, vec4 roundness, float dilate_3d, float onion_ratio, vec2 extrussion)
{
    // Match original effective mapping:
    // 2D in (x,y), extrusion along z
    size = size / 2.0;
    vec2 q2 = p.xy;
    float z = p.z;

    // Common scales
    float min_size = min(size.x, size.y);
    float halfZ    = size.z;

    vec4 r4 = roundness * min_size;
    float exScale = min(min_size, halfZ);
    vec2 ex = extrussion * exScale;

    float onion_amount = onion_ratio * min_size;

    // ---- 2D rounded box (per-corner radius) ----
    // Corner pick matching original logic:
    // rx = (x>0)? r.xy : r.zw; rc = (y>0)? rx.x : rx.y
    vec2 rx = (q2.x > 0.0) ? r4.xy : r4.zw;
    float rc = (q2.y > 0.0) ? rx.x : rx.y;

    vec2 a = abs(q2) - size.xy + vec2(rc);
    vec2 m = max(a, 0.0);
    float d = min(max(a.x, a.y), 0.0) + length(m) - rc;

    // Pre-extrude inset/outset transform
    float th = 0.5 * max(ex.x, ex.y) + min_size - onion_amount;
    d = abs(d + th) - th;

    // Asymmetric extrusion rounding by z sign
    float er = (z < 0.0) ? ex.x : ex.y;
    float h  = halfZ - er;

    // ---- Rounded extrusion ----
    float qx = d + er;
    float qy = abs(z) - h;

    // Equivalent to:
    // min(max(qx,qy),0) + length(max(vec2(qx,qy),0)) - er
    // with fewer temporaries
    float i = min(max(qx, qy), 0.0);
    vec2 o = max(vec2(qx, qy), 0.0);
    d = i + length(o) - er;

    // Onion (optional)
    if (onion_amount > ON_EPS) {
        d = abs(d + onion_amount) - onion_amount;
    }

    // Final dilate
    return d - dilate_3d;
}""")


CONSTANTS.update({
    "MAX_INNER_BULGE": ("float", 0.9),
    "SIZE_EPS": ("float", 1e-5),
})

SuperGeonShader = register_shader_module("""
@name SuperGeon
@inputs pos, size, roundness, dilate_3d, scale, bulge, onion_ratio, trapezoider
@outputs dist
@dependencies SPBase, dot2, SuperFrustum, Trapezoid2D, EulerRotate2D
@vardeps MAX_INNER_BULGE, BULGE_EPS, SIZE_EPS    

// ---------- Fast helpers ----------
float saturate(float x) { return clamp(x, 0.0, 1.0); }

// Squared distance point-segment, with precomputed inverse length^2 when available.
float sd2PointSegment(vec2 p, vec2 a, vec2 e, float inv_e2)
{
    vec2 pa = p - a;
    float t = saturate(dot(pa, e) * inv_e2);
    vec2 d = pa - e * t;
    return dot(d, d);
}

// ---------- 1) Faster right bulged edge ----------
vec2 sdBulgedRightEdgeFast(vec2 p, vec2 A, vec2 B, float bulge)
{
    vec2 e = B - A;
    float e2 = dot(e, e);
    if (e2 < SIZE_EPS) {
        vec2 d = p - A;
        return vec2(length(d), 1.0);
    }

    float L = sqrt(e2);
    float invL = 1.0 / L;

    // tangent + inside-normal
    vec2 t   = e * invL;
    vec2 nIn = vec2(e.y, -e.x) * invL;

    vec2 q = p - A;
    float u = dot(q, nIn);
    float v = dot(q, t);

    // local (x,z), z centered
    vec2 local = vec2(u, v - 0.5 * L);

    // Straight edge fast path (very common during optimization near zero bulge)
    if (abs(bulge) < BULGE_EPS) {
        float zc = clamp(local.y, -0.5 * L, 0.5 * L);
        vec2 dvec = vec2(local.x, local.y - zc);
        return vec2(length(dvec), local.x);
    }

    // Curved map
    vec2 m = mapArcBulge(local, L, bulge);

    // Distance to x=0, clamped in z
    float zc = clamp(m.y, -0.5 * L, 0.5 * L);
    vec2 dvec = vec2(m.x, m.y - zc);
    float d = length(dvec);

    // Keep your sign proxy convention
    float c_like = -m.x;
    return vec2(d, c_like);
}

// ---------- 2) Faster tapered trapezoid onion+bulge ----------
float sdTaperTrapezoidOnionBulgeFast(
    vec2 p,
    float inner,
    float h,
    float x3,
    float onion_ratio,
    float bulge // [-1,1]
){
    // Geometry
    float oneMinusOnion = 1.0 - onion_ratio;
    float xL  = -inner * oneMinusOnion;
    float xTL = -inner + (x3 + inner) * onion_ratio;
    float yB  = -h, yT = h;

    vec2 Lb = vec2(xL,  yB);
    vec2 Lt = vec2(xTL, yT);
    vec2 eL = Lt - Lb;

    float e2L = dot(eL, eL);
    float inv_e2L = 1.0 / max(e2L, 1e-12);

    // ---- squared distances to boundaries (keep squared until final sqrt)

    // Left segment
    float d2_left = sd2PointSegment(p, Lb, eL, inv_e2L);

    // Bottom segment [xL, 0] at yB
    float xb = clamp(p.x, xL, 0.0);
    float dyb = p.y - yB;
    float d2_bottom = (p.x - xb) * (p.x - xb) + dyb * dyb;

    // Top segment [xTL, x3] at yT
    float xt = clamp(p.x, xTL, x3);
    float dyt = p.y - yT;
    float d2_top = (p.x - xt) * (p.x - xt) + dyt * dyt;

    // Right bulged edge
    vec2 rightRes  = sdBulgedRightEdgeFast(p, vec2(0.0, yB), vec2(x3, yT), bulge);
    float d2_right = rightRes.x * rightRes.x;

    float d2 = min(min(d2_left, d2_bottom), min(d2_top, d2_right));

    // ---- inside test (same convention)
    // Left half-plane sign
    float cL = eL.y * (p.x - Lb.x) - eL.x * (p.y - Lb.y);
    cL = -cL;

    float cB = yB - p.y;
    float cT = p.y - yT;
    float cS = rightRes.y;

    // branchless-ish inside mask
    float m = max(max(cL, cB), max(cT, cS));
    float d = sqrt(d2);
    return (m <= 0.0) ? -d : d;
}

// ---------- 3) Faster InnerSuperGeon ----------
float InnerSuperGeonFast(
    vec3 p, vec3 size,
    float roundness,
    float onion_ratio,
    float trapezoider,
    float taper,
    float taper_bulge,
    float dilate_3d
){
    // Half extents once
    size *= 0.5;

    // Swap without branch (mix) based on trapezoider sign
    float sw = step(0.0, trapezoider); // 1 when >=0 => swap
    vec2 pxyA = p.xy;
    vec2 pxyB = p.yx;
    p.xy = mix(pxyA, pxyB, sw);

    vec2 sxyA = size.xy;
    vec2 sxyB = size.yx;
    size.xy = mix(sxyA, sxyB, sw);

    float min_size = min(size.x, size.y);
    float trap_amount = 1.0 - abs(trapezoider);

    float r = roundness * min_size;
    vec2 sxy_r = size.xy - vec2(r);

    // 2D trapezoid sdf then corner roundness offset
    float sdf2d = Trapezoid2D(p.xy, sxy_r.x, sxy_r.x * trap_amount, sxy_r.y) - r;

    vec2 pos_2d = vec2(sdf2d, p.z);

    // Params for bulged onion trapezoid in sdf-space
    float half_height = size.z;
    float x3 = -(1.0 - taper) * min_size;

    // Bulge scaling
    float inv_h = 1.0 / max(half_height, 1e-8);
    float max_bulge = min_size * (1.0 - onion_ratio) * inv_h * min(1.0, taper);
    max_bulge = min(MAX_INNER_BULGE, max_bulge);
    float min_bulge = 0.5;

    // preserve your piecewise choice
    float bulge_scale = (taper_bulge > 0.0) ? max_bulge : min_bulge;
    float bulge = clamp(taper_bulge, -1.0, 1.0) * bulge_scale;

    float sd = sdTaperTrapezoidOnionBulgeFast(
        pos_2d, min_size, half_height, x3, onion_ratio, bulge
    );

    return sd - dilate_3d;
}
float SuperGeon(vec3 p, vec3 size, float roundness, float dilate_3d, float taper, float bulge, float onion_ratio, float trapeze, float taper_bulge, float rot2d)
{   
    vec3 new_p = p;
    if (bulge != 0.0)
    {
        vec2 new_xz = mapArcBulge(p.xz, size.z, bulge);
        new_p = vec3(new_xz.x, p.y, new_xz.y);
    }else{
        new_p.x = -new_p.x;
    }
    new_p.xy = EulerRotate2D(new_p.xy, rot2d);

    float sd = InnerSuperGeonFast(new_p, size, roundness, onion_ratio, trapeze, taper, taper_bulge, dilate_3d);
    return sd;
}""")

