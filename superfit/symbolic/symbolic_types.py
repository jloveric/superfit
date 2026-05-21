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
"""
import superfit.symbolic as sps

VALID_PACKED_CLASSES = (
    sps.CuboidPacked,
    sps.SQPacked,
    sps.SuperFrustumPacked, 
    sps.SPProtoPacked,
    sps.SuperGeonPacked,
    sps.SolidSFPacked,
    sps.VarAxisSQPacked,
    sps.VarAxisSFPacked,
    sps.VarAxisSGPacked,
    sps.VarAxisSPPPacked,
)

VARAXIS_CLASSES = (
    sps.VarAxisSQ,
    sps.VarAxisSF,
    sps.VarAxisSG,
    sps.VarAxisSPP,
)
VARAXIS_EXECUTED_CLASSES = (
    sps.SuperFrustumX,
    sps.SuperFrustumY,
    sps.SuperFrustumZ,
    sps.SuperQuadricX,
    sps.SuperQuadricY,
    sps.SuperQuadricZ,
    sps.SuperGeonX,
    sps.SuperGeonY,
    sps.SuperGeonZ,
    sps.SPProtoX,
    sps.SPProtoY,
    sps.SPProtoZ,
)
VALID_BATCHED_SU_CLASSES = (
    sps.CuboidPackedBatchedSU,
    sps.SQPackedBatchedSU,
    sps.SuperFrustumPackedBatchedSU,
    sps.SPProtoPackedBatchedSU,
    sps.SGPackedBatchedSU,
    sps.SolidSFPackedBatchedSU,
    sps.VarAxisSQPackedBatchedSU,
    sps.VarAxisSFPackedBatchedSU,
    sps.VarAxisSPPPackedBatchedSU,
    sps.VarAxisSGPackedBatchedSU,
)
VALID_BATCHED_STOCHASTIC_SU_CLASSES = (
    sps.CuboidPackedBatchedStochasticSU,
    sps.SQPackedBatchedStochasticSU,
    sps.SuperFrustumPackedBatchedStochasticSU,
    sps.SPProtoPackedBatchedStochasticSU,
    sps.SGPackedBatchedStochasticSU,
    sps.SolidSFPackedBatchedStochasticSU,
    sps.VarAxisSQPackedBatchedStochasticSU,
    sps.VarAxisSFPackedBatchedStochasticSU,
    sps.CustomVASF,
    sps.VarAxisSPPPackedBatchedStochasticSU,
    sps.VarAxisSGPackedBatchedStochasticSU,
)

SFSP_CLASSES = (
    sps.SFSPX,
    sps.SFSPY,
    sps.SFSPZ,
    sps.SFSP,
    sps.SPPSPX,
    sps.SPPSPY,
    sps.SPPSPZ,
    sps.SPPSP,
    sps.SGSPX,
    sps.SGSPY,
    sps.SGSPZ,
    sps.SGSP,
)
SFSP_UNRAVEL_CLASSES = (
    sps.SuperFrustumX,
    sps.SuperFrustumY,
    sps.SuperFrustumZ,
    sps.SuperFrustum,
    sps.SuperGeonX,
    sps.SuperGeonY,
    sps.SuperGeonZ,
    sps.SuperGeon,
    sps.SPProtoX,
    sps.SPProtoY,
    sps.SPProtoZ,
    sps.SPProto,
)
