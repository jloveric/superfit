"""
Shaders for the variable axis primitives.
"""
import numpy as np
from sysl.shader.shader_module import register_shader_module
from sysl.shader.shader_templates.common import CONSTANTS

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
