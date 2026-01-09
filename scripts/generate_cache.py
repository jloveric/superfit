
import os
import time
import torch as th
import superfit.symbolic as sps
import torch._dynamo as dynamo
from superfit.utils.config import AlgorithmConfig as AlgConf
# from .fast_opt import run_optimization_loop
from geolipi.torch_compute.unroll_expression import unroll_expression
# from ..torch_compute.triton_convert import batched_sf_packed_stochastic_eval, _sdf_smooth_union_pair
from superfit.torch_compute.batched_primitives import batched_sf_packed_stochastic_eval, _sdf_smooth_union_pair, batched_sf_packed_stochastic_su_eval


def compile_program():
    # prim_function = map_to_prim_inner_fn[in_program.__class__]
    # return batched_sf_packed_stochastic_su_eval
    dtype = AlgConf.OPT_DTYPE
    device = "cuda"
    for BATCH_SIZE in range(2, 100):
        print("================================================")
        print(f"Generating Cache for Batch size: {BATCH_SIZE}")
        print("================================================")
        
        prim_function = batched_sf_packed_stochastic_eval
        PC_SIZE = 200_000 + 128 ** 3
        temperature = 1.0
        artifact_file = AlgConf.AOT_ARTIFACT_FILE
        _coords = th.randn(1, PC_SIZE, 3, dtype=dtype, device=device).clone().detach().requires_grad_(False)
        _params = th.randn(BATCH_SIZE, 14, dtype=dtype, device=device).clone().detach().requires_grad_(True)
        _su_vals = th.randn(BATCH_SIZE-1, 1, dtype=dtype, device=device).clone().detach().requires_grad_(True)
        _logits = th.randn(BATCH_SIZE, 2, dtype=dtype, device=device).clone().detach().requires_grad_(True)
        _temperature = th.randn(1, dtype=dtype, device=device).clone().detach().requires_grad_(False)
        


        comp_func = th.compile(batched_sf_packed_stochastic_su_eval, 
            backend="inductor",
            mode="max-autotune",
            # mode="max-autotune-no-cudagraphs",
            dynamic=False,
            fullgraph=True,
            # options={"triton.cudagraphs": False},
        )
        
        # dynamo.mark_dynamic(_coords, 1)
        # dynamo.mark_dynamic(_params, 0, min=2, max=100)
        # dynamo.mark_dynamic(_logits, 0, min=2, max=100)
        # dynamo.mark_dynamic(_su_vals, 0, min=1, max=99)

        start_time = time.time()
        res1, res2 = comp_func(_coords, _params, _su_vals, _logits, _temperature)
        loss = res1.sum() + res2.sum()
        loss.backward()
        real_artifact_file = artifact_file.replace(".pt", f"_{BATCH_SIZE}.pt")
        print(f"Saving artifacts to {real_artifact_file}")
        artifacts = th.compiler.save_cache_artifacts()
        assert artifacts is not None
        artifact_bytes, cache_info = artifacts
        th.save(artifact_bytes, real_artifact_file)
        th.cuda.empty_cache()
        th._dynamo.reset()

if __name__ == "__main__":
    compile_program()