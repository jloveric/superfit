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

Shaders for the variable axis primitives.
"""
import numpy as np
from sysl.shader.shader_module import register_shader_module
from sysl.shader.shader_templates.common import CONSTANTS

# ================================ Y Axis ================================


SuperFrustumYShader = register_shader_module("""
@name SuperFrustumY
@inputs pos, radius
@outputs dist
@dependencies SuperFrustum
@vardeps
float SuperFrustumY(vec3 p, vec3 size, float roundness, float dilate_3d, float scale, float bulge_ratio, float onion_ratio)
{   
    return SuperFrustum(p, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio);
}""")

SPProtoYShader = register_shader_module("""
@name SPProtoY
@inputs pos, size, roundness, dilate_3d, onion_ratio, extrussion
@outputs dist
@dependencies SPProto
@vardeps ON_EPS
float SPProtoY(vec3 p, vec3 size, vec4 roundness, float dilate_3d, float onion_ratio, vec2 extrussion)
{
    return SPProto(p, size, roundness, dilate_3d, onion_ratio, extrussion);
}""")

SuperGeonYShader = register_shader_module("""
@name SuperGeonY
@inputs pos, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, trapeze, taper_bulge, rot2d
@outputs dist
@dependencies SuperGeon
@vardeps
float SuperGeonY(vec3 p, vec3 size, float roundness, float dilate_3d, float scale, float bulge_ratio, float onion_ratio, float trapeze, float taper_bulge, float rot2d)
{
    return SuperGeon(p, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, trapeze, taper_bulge, rot2d);
}""")

SFSPYShader = register_shader_module("""
@name SFSPY
@inputs pos, size, params, onion_ratio
@outputs dist
@dependencies SFSP
@vardeps 

float SFSPY(vec3 p, vec3 size, vec4 params, float onion_ratio)
{
    return SFSP(p, size, params, onion_ratio);
}""")       

SPPSPYShader = register_shader_module("""
@name SPPSPY
@inputs pos, size, roundess, doe
@outputs dist
@dependencies SPPSP
@vardeps 

float SPPSPY(vec3 p, vec3 size, vec4 roundess, vec4 doe)
{
    return SPPSP(p, size, roundess, doe);
}""")

SGSPYShader = register_shader_module("""
@name SGSPY
@inputs pos, size, params, params_2
@outputs dist
@dependencies SGSP
@vardeps 

float SGSPY(vec3 p, vec3 size, vec4 params, vec4 params_2)
{
    return SGSP(p, size, params, params_2);
}""")  

# ================================ Z Axis ================================


SuperFrustumZShader = register_shader_module("""
@name SuperFrustumZ
@inputs pos, radius
@outputs dist
@dependencies SuperFrustum
@vardeps
float SuperFrustumZ(vec3 p, vec3 size, float roundness, float dilate_3d, float scale, float bulge_ratio, float onion_ratio)
{   
    p = p.yzx;
    size = size.yzx;
    return SuperFrustum(p, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio);
}""")

SPProtoZShader = register_shader_module("""
@name SPProtoZ
@inputs pos, size, roundness, dilate_3d, onion_ratio, extrussion
@outputs dist
@dependencies SPProto
@vardeps ON_EPS
float SPProtoZ(vec3 p, vec3 size, vec4 roundness, float dilate_3d, float onion_ratio, vec2 extrussion)
{
    p = p.yzx;
    size = size.yzx;    
    return SPProto(p, size, roundness, dilate_3d, onion_ratio, extrussion);
}""")

SuperGeonZShader = register_shader_module("""
@name SuperGeonZ
@inputs pos, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, trapeze, taper_bulge, rot2d
@outputs dist
@dependencies SuperGeon
@vardeps
float SuperGeonZ(vec3 p, vec3 size, float roundness, float dilate_3d, float scale, float bulge_ratio, float onion_ratio, float trapeze, float taper_bulge, float rot2d)
{
    p = p.yzx;
    size = size.yzx;
    return SuperGeon(p, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, trapeze, taper_bulge, rot2d);
}""")

SFSPZShader = register_shader_module("""
@name SFSPZ
@inputs pos, size, params, onion_ratio
@outputs dist
@dependencies SFSP
@vardeps 

float SFSPZ(vec3 p, vec3 size, vec4 params, float onion_ratio)
{
    p = p.yzx;
    size = size.yzx;
    return SFSP(p, size, params, onion_ratio);
}""")

SPPSPZShader = register_shader_module("""
@name SPPSPZ
@inputs pos, size, roundess, doe
@outputs dist
@dependencies SPPSP
@vardeps 

float SPPSPZ(vec3 p, vec3 size, vec4 roundess, vec4 doe)
{
    return SPPSP(p, size, roundess, doe);
}""")

SGSPZShader = register_shader_module("""
@name SGSPZ
@inputs pos, size, params, params_2
@outputs dist
@dependencies SGSP
@vardeps 

float SGSPZ(vec3 p, vec3 size, vec4 params, vec4 params_2)
{
    p = p.yzx;
    size = size.yzx;
    return SGSP(p, size, params, params_2);
}""")


# ================================ X Axis ================================


SuperFrustumXShader = register_shader_module("""
@name SuperFrustumX
@inputs pos, radius
@outputs dist
@dependencies SuperFrustum
@vardeps
float SuperFrustumX(vec3 p, vec3 size, float roundness, float dilate_3d, float scale, float bulge_ratio, float onion_ratio)
{      
    p = p.zxy;
    size = size.zxy;
    return SuperFrustum(p, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio);
}""")

SPProtoXShader = register_shader_module("""
@name SPProtoX
@inputs pos, size, roundness, dilate_3d, onion_ratio, extrussion
@outputs dist
@dependencies SPProto
@vardeps ON_EPS
float SPProtoX(vec3 p, vec3 size, vec4 roundness, float dilate_3d, float onion_ratio, vec2 extrussion)
{
    p = p.zxy;
    size = size.zxy;
    return SPProto(p, size, roundness, dilate_3d, onion_ratio, extrussion);
}""")

SuperGeonXShader = register_shader_module("""
@name SuperGeonX
@inputs pos, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, trapeze, taper_bulge, rot2d
@outputs dist
@dependencies SuperGeon
@vardeps
float SuperGeonX(vec3 p, vec3 size, float roundness, float dilate_3d, float scale, float bulge_ratio, float onion_ratio, float trapeze, float taper_bulge, float rot2d)
{
    p = p.zxy;
    size = size.zxy;
    return SuperGeon(p, size, roundness, dilate_3d, scale, bulge_ratio, onion_ratio, trapeze, taper_bulge, rot2d);
}""")

SFSPXShader = register_shader_module("""
@name SFSPX
@inputs pos, size, params, onion_ratio
@outputs dist
@dependencies SFSP
@vardeps 

float SFSPX(vec3 p, vec3 size, vec4 params, float onion_ratio)
{
    p = p.zxy;
    size = size.zxy;
    return SFSP(p, size, params, onion_ratio);
}""")

SPPSPXShader = register_shader_module("""
@name SPPSPX
@inputs pos, size, roundess, doe
@outputs dist
@dependencies SPPSP
@vardeps 

float SPPSPX(vec3 p, vec3 size, vec4 roundess, vec4 doe)
{
    return SPPSP(p, size, roundess, doe);
}""")

SGSPXShader = register_shader_module("""
@name SGSPX
@inputs pos, size, params, onion_ratio
@outputs dist
@dependencies SGSP
@vardeps 

float SGSPX(vec3 p, vec3 size, vec4 params, vec4 params_2)
{
    p = p.zxy;
    size = size.zxy;
    return SGSP(p, size, params, params_2);
}""")