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
    sps.VarAxisSPPPackedBatchedStochasticSU,
    sps.VarAxisSGPackedBatchedStochasticSU,
)

SFSP_CLASSES = (
    sps.SFSPX,
    sps.SFSPY,
    sps.SFSPZ,
    sps.SFSP,
)
SFSP_UNRAVEL_CLASSES = (
    sps.SuperFrustumX,
    sps.SuperFrustumY,
    sps.SuperFrustumZ,
    sps.SuperFrustum,
)