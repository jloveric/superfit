import torch as th
import geolipi.symbolic as gls
from geolipi.torch_compute.sketcher import Sketcher
from typing import Optional, List, Tuple
from geolipi.symbolic.registry import register_symbol
from sysl.shader.shader_module import register_shader_module


SPBaseShader = register_shader_module("""
@name SPBase
@inputs pos, radius
@outputs dist
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
@inputs pos, radius
@outputs dist
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

SuperFrustumShader = register_shader_module("""
@name SuperFrustum
@inputs pos, radius
@outputs dist
@dependencies SPTaperedOnion
@vardeps
// --- constants ---
#define PI 3.14159265359

// rotate 90° CCW
vec2 rot90(vec2 v){ return vec2(-v.y, v.x); }

vec2 mapArcBulge(vec2 p, float z, float bulge)
{
    float eps = 1e-5;
    float half_z = 0.5 * z;
    float theta_top = max(bulge * (PI * 0.5), eps);

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
    return new_point;
}

float SuperFrustum(vec3 p, vec3 size, float roundness, float dilate_3d, float scale, float bulge_ratio, float onion_ratio)
{   
    vec3 new_p = p;
    float bulge = bulge_ratio;
    if (bulge > 0.0)
    {
        vec2 new_xz = mapArcBulge(p.xz, size.z, bulge);
        new_p = vec3(new_xz.x, p.y, new_xz.y);
    }
    float sd = SPTaperedOnion(new_p, size, roundness, dilate_3d, scale, onion_ratio);
    return sd;
}""")