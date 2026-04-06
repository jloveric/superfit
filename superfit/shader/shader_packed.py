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

Packed Variant used in the editing shader pass. 
"""
import numpy as np
from sysl.shader.shader_module import register_shader_module
from sysl.shader.shader_templates.common import CONSTANTS

SFSPShader = register_shader_module("""
@name SFSP
@inputs pos, size, params, onion_ratio
@outputs dist
@dependencies SuperFrustum
@vardeps 

float SFSP(vec3 p, vec3 size, vec4 params, float onion_ratio)
{   
    float roundness = params.x;
    float dilate_3d = params.y;
    float scale = params.z;
    float bulge_ratio = params.w;
    return SuperFrustum(p, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio);
}""")

SPPSPShader = register_shader_module("""
@name SPPSP
@inputs pos, size, params, onion_ratio
@outputs dist
@dependencies SPProto
@vardeps 

float SPPSP(vec3 p, vec3 size, vec4 roundess, vec4 doe)
{
    return SPProto(p, size, roundess, doe.x, doe.y, doe.zw);
}""")

SGSPShader = register_shader_module("""
@name SGSP
@inputs pos, size, params, onion_ratio
@outputs dist
@dependencies SuperGeon
@vardeps 

float SGSP(vec3 p, vec3 size, vec4 params, vec4 params2)
{  
    float roundness = params.x;
    float dilate_3d = params.y;
    float taper = params.z;
    float bulge = params.w;
    float onion_ratio = params2.x;
    float trapeze = params2.y;
    float taper_bulge = params2.z;
    float rot2d = params2.w;
    float sdf = SuperGeon(p, size, roundness, dilate_3d, taper, bulge, onion_ratio, trapeze, taper_bulge, rot2d);
    return sdf;
}""")
