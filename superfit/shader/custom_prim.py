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

Custom primitives to speed up SuperGeon.
Did not really speed up things :(.
Uncomment the SMMap["SuperGeon"] = CustomSuperGeon line to use this.
"""
import numpy as np
from sysl.shader.shader_module import register_shader_module
from sysl.shader.shader_templates.common import CONSTANTS
from sysl.shader.shader_module import register_shader_module, SMMap
from sysl.shader.shader_mod_ext import CustomFunctionShaderModule
from string import Template


ShaderGeonStructShader = register_shader_module("""
@name SGStruct
@inputs 
@outputs dist
@dependencies SPBase, dot2, SuperFrustum, Trapezoid2D, EulerRotate2D
@vardeps MAX_INNER_BULGE    
struct SuperGeonPC {
    vec4 outerArc; // center radius s c
    vec4 innerRightArc; // center radius s c
    float inner_bulge;
};
""")

SGStructGenShader = register_shader_module("""
@name SGStructGen
@inputs pos, size, roundness, dilate_3d, scale, bulge, onion_ratio, trapeze
@outputs dist
@dependencies SGStruct
@vardeps MAX_INNER_BULGE, BULGE_EPS, SIZE_EPS
SuperGeonPC makeSuperGeonPC(vec3 size, float roundness, float dilate_3d, float taper, float bulge, float onion_ratio, float trapeze, float taper_bulge, float rot2d)
{
    SuperGeonPC sg_struct;

    size *= 0.5;
    float min_size = min(size.x, size.y);
    float half_height = size.z;

    float inv_h = 1.0 / max(half_height, 1e-8);
    float max_bulge = min_size * (1.0 - onion_ratio) * inv_h * min(1.0, taper);
    max_bulge = min(MAX_INNER_BULGE, max_bulge);
    float min_bulge = 0.5;

    // preserve your piecewise choice
    float bulge_scale = (taper_bulge > 0.0) ? max_bulge : min_bulge;
    float inner_bulge = clamp(taper_bulge, -1.0, 1.0) * bulge_scale;
    sg_struct.inner_bulge = inner_bulge;
    
    float theta_top = max(abs(bulge) * (PI * 0.5), BULGE_EPS);
    float center_pos = half_height / tan(theta_top);
    float radius = sqrt(center_pos * center_pos + half_height * half_height);
    float s = sin(theta_top), c = cos(theta_top);



    sg_struct.outerArc = vec4(center_pos, radius, s, c);

    theta_top = max(abs(inner_bulge) * (PI * 0.5), BULGE_EPS);
    center_pos = half_height / tan(theta_top);
    radius = sqrt(center_pos * center_pos + half_height * half_height);
    s = sin(theta_top), c = cos(theta_top);

    sg_struct.innerRightArc = vec4(center_pos, radius, s, c);

    
    return sg_struct;
}
""")

SGFromStructShader = register_shader_module("""
@name SGFromStruct
@inputs pos, 
@outputs dist
@dependencies SGStruct, SGStructGen
@vardeps MAX_INNER_BULGE

vec2 mapArcBulgeStruct(vec2 p, float z, float bulge, vec4 arcSpec)
{
    float half_z = 0.5 * z;
    p.x = p.x * sign(bulge);
    float theta_top = max(abs(bulge) * (PI * 0.5), BULGE_EPS);
    //float theta_top = bulge * (PI * 0.5);

    float center_pos = arcSpec.x;
    vec2  center = vec2(center_pos, 0.0);
    float radius = arcSpec.y;

    // angle wrt arc center
    float point_angle = atan(p.y, center_pos - p.x);

    // inside the arc span
    float new_y = clamp(point_angle / theta_top, -1.0, 1.0) * half_z;
    float new_x = length(p - center) - radius;
    vec2  inside_point = vec2(new_x, new_y);

    // endpoints and correct tangents/normals
    float s = arcSpec.z, c = arcSpec.w;

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
vec2 sdBulgedRightEdgeFastStruct(vec2 p, vec2 A, vec2 B, float bulge, vec4 arcSpec)
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
    vec2 m = mapArcBulgeStruct(local, L, bulge, arcSpec);

    // Distance to x=0, clamped in z
    float zc = clamp(m.y, -0.5 * L, 0.5 * L);
    vec2 dvec = vec2(m.x, m.y - zc);
    float d = length(dvec);

    // Keep your sign proxy convention
    float c_like = -m.x;
    return vec2(d, c_like);
}

// ---------- 2) Faster tapered trapezoid onion+bulge ----------
float sdTaperTrapezoidOnionBulgeFastStruct(
    vec2 p,
    float inner,
    float h,
    float x3,
    float onion_ratio,
    float bulge, // [-1,1]
    vec4 arcSpec
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
    vec2 rightRes  = sdBulgedRightEdgeFastStruct(p, vec2(0.0, yB), vec2(x3, yT), bulge,arcSpec);
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
    float trapeze,
    float taper,
    float inner_bulge,
    float dilate_3d,
    vec4 arcSpec
){
    // Half extents once
    size *= 0.5;

    // Swap without branch (mix) based on trapeze sign
    float sw = step(0.0, trapeze); // 1 when >=0 => swap
    vec2 pxyA = p.xy;
    vec2 pxyB = p.yx;
    p.xy = mix(pxyA, pxyB, sw);

    vec2 sxyA = size.xy;
    vec2 sxyB = size.yx;
    size.xy = mix(sxyA, sxyB, sw);

    float min_size = min(size.x, size.y);
    float trap_amount = 1.0 - abs(trapeze);

    float r = roundness * min_size;
    vec2 sxy_r = size.xy - vec2(r);

    // 2D trapezoid sdf then corner roundness offset
    float sdf2d = Trapezoid2D(p.xy, sxy_r.x, sxy_r.x * trap_amount, sxy_r.y) - r;

    vec2 pos_2d = vec2(sdf2d, p.z);

    // Params for bulged onion trapezoid in sdf-space
    float half_height = size.z;
    float x3 = -(1.0 - taper) * min_size;


    float sd = sdTaperTrapezoidOnionBulgeFastStruct(
        pos_2d, min_size, half_height, x3, onion_ratio, inner_bulge, arcSpec
    );

    return sd - dilate_3d;
}
float SuperGeon(vec3 p, vec3 size, float roundness, float dilate_3d, float taper, float bulge, float onion_ratio, float trapeze, float taper_bulge, float rot2d, SuperGeonPC sg_struct)
{   
    vec3 new_p = p;
    if (bulge != 0.0)
    {
        vec2 new_xz = mapArcBulgeStruct(p.xz, size.z, bulge, sg_struct.outerArc);
        new_p = vec3(new_xz.x, p.y, new_xz.y);
    }else{
        new_p.x = -new_p.x;
    }
    new_p.xy = EulerRotate2D(new_p.xy, rot2d);

    float sd = InnerSuperGeonFast(new_p, size, roundness, onion_ratio, trapeze, taper, sg_struct.inner_bulge, dilate_3d, sg_struct.innerRightArc);
    return sd;
}

""")

# Build the custom builder. 
SGStuctInit = Template("""
SuperGeonPC sg_struct_list[${n_structs}];
void init_sg_struct_list() {
    ${code}
}
""")

class CustomSuperGeon(CustomFunctionShaderModule):
    def __init__(self, name=None,template=None, *args, **kwargs):
        if template is None:
            template = SGStuctInit
        if name is None:
            name = "SuperGeon"

        super().__init__(name, template, *args, **kwargs)
        self.dependencies = ["SGFromStruct",]
        # self.dependencies = ["SGFromStructMini"]
        self.has_precompute = True
        self.code_lines = []
        
    def precompute_call(self):
        return "init_sg_struct_list();"

    def get_updated_param(self, primitive_param):
        new_params = f"{primitive_param}, {self.get_latest_struct()}"
        return new_params

    def register_hit(self, *args, **kwargs):
        primitive_param = kwargs.get("primitive_param", None)
        assert primitive_param is not None, "primitive_param are required"
        code_line = f"sg_struct_list[{self.hit_count}] = makeSuperGeonPC({primitive_param});"
        self.code_lines.append(code_line)
        self.hit_count += 1
    
    def get_latest_struct(self):
        return f"sg_struct_list[{self.hit_count-1}]"

    def generate_code(self):
        inner_code_lines = "\n".join(self.code_lines)
        code = self.template.substitute(code=inner_code_lines, n_structs=self.hit_count)
        self.code = code.strip()

    def emit_code(self):
        if self.code is None:
            self.generate_code()
        return self.code


# SMMap["SuperGeon"] = CustomSuperGeon

# Modify Eval:
