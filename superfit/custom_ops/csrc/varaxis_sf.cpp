// SPDX-FileCopyrightText: 2026 Aditya Ganeshan
// SPDX-License-Identifier: MIT

#include <torch/extension.h>

#include <cstdlib>
#include <limits>
#include <vector>

void launch_varaxis_sf_forward(
    const float* coords,
    const float* params,
    const float* gumbel,
    float temperature,
    int coord_B,
    int param_B,
    int M,
    float* out);

void launch_varaxis_sf_backward(
    const float* grad_out,
    const float* coords,
    const float* params,
    const float* gumbel,
    float temperature,
    int B,
    int M,
    float* grad_coords,
    float* grad_params);

void launch_varaxis_sf_backward_params(
    const float* grad_out,
    const float* coords,
    const float* params,
    const float* gumbel,
    float temperature,
    int coord_B,
    int param_B,
    int M,
    float* grad_params);

void launch_varaxis_sf_backward_params_reduced(
    const float* grad_out,
    const float* coords,
    const float* params,
    const float* gumbel,
    float temperature,
    int coord_B,
    int param_B,
    int M,
    int num_tiles,
    float* partials,
    float* grad_params);

void launch_varaxis_sf_backward_params_partials(
    const float* grad_out,
    const float* coords,
    const float* params,
    const float* gumbel,
    float temperature,
    int coord_B,
    int param_B,
    int M,
    int num_tiles,
    float* partials);

void launch_varaxis_sf_reduce_param_partials(
    const float* partials,
    int param_B,
    int num_tiles,
    float* grad_params);

void launch_varaxis_sf_assembly_forward(
    float* primitive_sdfs,
    const float* su_vals,
    const float* logits,
    const float* outer_gumbel,
    float temperature,
    int K,
    int M,
    float* out);

void launch_varaxis_sf_assembly_backward_prep(
    const float* grad_primitive,
    const float* grad_out,
    const float* primitive_sdfs,
    const float* su_vals,
    const float* logits,
    const float* outer_gumbel,
    float temperature,
    int K,
    int M,
    float* grad_raw,
    float* grad_su,
    float* grad_logits);

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_FLOAT(x);    \
    CHECK_CONTIGUOUS(x)

static int reduced_backward_min_k() {
    const char* value = std::getenv("SUPERFIT_CUSTOM_VASF_REDUCED_BACKWARD_MIN_K");
    if (value == nullptr || value[0] == '\0') {
        return 2;
    }

    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 1 || parsed > 129) {
        return 2;
    }
    return static_cast<int>(parsed);
}

static void check_common(
    const torch::Tensor& coords,
    const torch::Tensor& params,
    const torch::Tensor& gumbel,
    double temperature) {
    CHECK_INPUT(coords);
    CHECK_INPUT(params);
    CHECK_INPUT(gumbel);
    TORCH_CHECK(coords.dim() == 3 && coords.size(2) == 3, "coords must have shape (B, M, 3)");
    TORCH_CHECK(params.dim() == 2 && params.size(1) == 17, "params must have shape (B, 17)");
    TORCH_CHECK(gumbel.dim() == 2 && gumbel.size(1) == 3, "gumbel must have shape (B, 3)");
    TORCH_CHECK(
        coords.size(0) == 1 || coords.size(0) == params.size(0),
        "coords B dimension must be 1 or match params B");
    TORCH_CHECK(gumbel.size(0) == params.size(0), "gumbel B dimension must match params B");
    TORCH_CHECK(temperature > 0.0, "temperature must be positive");
    TORCH_CHECK(coords.size(0) <= std::numeric_limits<int>::max(), "coords B is too large");
    TORCH_CHECK(params.size(0) <= std::numeric_limits<int>::max(), "params B is too large");
    TORCH_CHECK(coords.size(1) <= std::numeric_limits<int>::max(), "M is too large");
}

static void check_assembly(
    const torch::Tensor& coords,
    const torch::Tensor& params,
    const torch::Tensor& su_vals,
    const torch::Tensor& logits,
    const torch::Tensor& inner_gumbel,
    const torch::Tensor& outer_gumbel,
    double temperature) {
    check_common(coords, params, inner_gumbel, temperature);
    CHECK_INPUT(su_vals);
    CHECK_INPUT(logits);
    CHECK_INPUT(outer_gumbel);
    TORCH_CHECK(su_vals.dim() == 2 && su_vals.size(0) == params.size(0) - 1 && su_vals.size(1) == 1,
                "su_vals must have shape (params.shape[0] - 1, 1)");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) == params.size(0) && logits.size(1) == 2,
                "logits must have shape (params.shape[0], 2)");
    TORCH_CHECK(outer_gumbel.dim() == 2 && outer_gumbel.size(0) == params.size(0) && outer_gumbel.size(1) == 2,
                "outer_gumbel must have shape (params.shape[0], 2)");
    TORCH_CHECK(params.size(0) >= 1, "assembly requires at least one primitive");
    TORCH_CHECK(params.size(0) <= 128, "custom assembly CUDA path supports at most 128 primitives");
}

torch::Tensor varaxis_sf_forward(
    torch::Tensor coords,
    torch::Tensor params,
    double temperature,
    torch::Tensor gumbel) {
    check_common(coords, params, gumbel, temperature);
    auto out = torch::empty({params.size(0), coords.size(1)}, coords.options());
    launch_varaxis_sf_forward(
        coords.data_ptr<float>(),
        params.data_ptr<float>(),
        gumbel.data_ptr<float>(),
        static_cast<float>(temperature),
        static_cast<int>(coords.size(0)),
        static_cast<int>(params.size(0)),
        static_cast<int>(coords.size(1)),
        out.data_ptr<float>());
    return out;
}

std::vector<torch::Tensor> varaxis_sf_backward(
    torch::Tensor grad_out,
    torch::Tensor coords,
    torch::Tensor params,
    double temperature,
    torch::Tensor gumbel) {
    CHECK_INPUT(grad_out);
    check_common(coords, params, gumbel, temperature);
    TORCH_CHECK(coords.size(0) == params.size(0), "full backward requires coords B to match params B");
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must have shape (B, M)");
    TORCH_CHECK(grad_out.size(0) == params.size(0), "grad_out B dimension must match params");
    TORCH_CHECK(grad_out.size(1) == coords.size(1), "grad_out M dimension must match coords");

    auto grad_coords = torch::zeros_like(coords);
    auto grad_params = torch::zeros_like(params);
    launch_varaxis_sf_backward(
        grad_out.data_ptr<float>(),
        coords.data_ptr<float>(),
        params.data_ptr<float>(),
        gumbel.data_ptr<float>(),
        static_cast<float>(temperature),
        static_cast<int>(coords.size(0)),
        static_cast<int>(coords.size(1)),
        grad_coords.data_ptr<float>(),
        grad_params.data_ptr<float>());
    return {grad_coords, grad_params};
}

torch::Tensor varaxis_sf_backward_params(
    torch::Tensor grad_out,
    torch::Tensor coords,
    torch::Tensor params,
    double temperature,
    torch::Tensor gumbel) {
    CHECK_INPUT(grad_out);
    check_common(coords, params, gumbel, temperature);
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must have shape (B, M)");
    TORCH_CHECK(grad_out.size(0) == params.size(0), "grad_out B dimension must match params");
    TORCH_CHECK(grad_out.size(1) == coords.size(1), "grad_out M dimension must match coords");

    auto grad_params = torch::zeros_like(params);
    const int param_B = static_cast<int>(params.size(0));
    const int M = static_cast<int>(coords.size(1));
    if (param_B < reduced_backward_min_k()) {
        launch_varaxis_sf_backward_params(
            grad_out.data_ptr<float>(),
            coords.data_ptr<float>(),
            params.data_ptr<float>(),
            gumbel.data_ptr<float>(),
            static_cast<float>(temperature),
            static_cast<int>(coords.size(0)),
            param_B,
            M,
            grad_params.data_ptr<float>());
    } else {
        constexpr int kBlockSize = 128;
        constexpr int kNumParams = 17;
        const int num_tiles = (M + kBlockSize - 1) / kBlockSize;
        auto partials = torch::empty({param_B, num_tiles, kNumParams}, params.options());
        launch_varaxis_sf_backward_params_reduced(
            grad_out.data_ptr<float>(),
            coords.data_ptr<float>(),
            params.data_ptr<float>(),
            gumbel.data_ptr<float>(),
            static_cast<float>(temperature),
            static_cast<int>(coords.size(0)),
            param_B,
            M,
            num_tiles,
            partials.data_ptr<float>(),
            grad_params.data_ptr<float>());
    }
    return grad_params;
}

std::vector<torch::Tensor> varaxis_sf_assembly_backward_prep_only(
    torch::Tensor grad_primitive,
    torch::Tensor grad_out,
    torch::Tensor primitive_sdfs,
    torch::Tensor params,
    torch::Tensor su_vals,
    torch::Tensor logits,
    double temperature,
    torch::Tensor outer_gumbel) {
    CHECK_INPUT(grad_primitive);
    CHECK_INPUT(grad_out);
    CHECK_INPUT(primitive_sdfs);
    CHECK_INPUT(params);
    CHECK_INPUT(su_vals);
    CHECK_INPUT(logits);
    CHECK_INPUT(outer_gumbel);
    TORCH_CHECK(params.dim() == 2 && params.size(1) == 17, "params must have shape (K, 17)");
    TORCH_CHECK(grad_primitive.dim() == 2, "grad_primitive must have shape (K, M)");
    TORCH_CHECK(primitive_sdfs.dim() == 2, "primitive_sdfs must have shape (K, M)");
    TORCH_CHECK(grad_primitive.sizes() == primitive_sdfs.sizes(),
                "grad_primitive and primitive_sdfs shapes must match");
    TORCH_CHECK(grad_primitive.size(0) == params.size(0), "K dimension must match params");
    TORCH_CHECK(grad_out.dim() == 1 && grad_out.size(0) == primitive_sdfs.size(1),
                "grad_out must have shape (M,)");
    TORCH_CHECK(su_vals.dim() == 2 && su_vals.size(0) == params.size(0) - 1 && su_vals.size(1) == 1,
                "su_vals must have shape (K - 1, 1)");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) == params.size(0) && logits.size(1) == 2,
                "logits must have shape (K, 2)");
    TORCH_CHECK(outer_gumbel.dim() == 2 && outer_gumbel.size(0) == params.size(0) && outer_gumbel.size(1) == 2,
                "outer_gumbel must have shape (K, 2)");
    TORCH_CHECK(temperature > 0.0, "temperature must be positive");

    auto grad_raw = torch::empty_like(primitive_sdfs);
    auto grad_su = torch::zeros_like(su_vals);
    auto grad_logits = torch::zeros_like(logits);
    launch_varaxis_sf_assembly_backward_prep(
        grad_primitive.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        primitive_sdfs.data_ptr<float>(),
        su_vals.data_ptr<float>(),
        logits.data_ptr<float>(),
        outer_gumbel.data_ptr<float>(),
        static_cast<float>(temperature),
        static_cast<int>(params.size(0)),
        static_cast<int>(primitive_sdfs.size(1)),
        grad_raw.data_ptr<float>(),
        grad_su.data_ptr<float>(),
        grad_logits.data_ptr<float>());
    return {grad_raw, grad_su, grad_logits};
}

torch::Tensor varaxis_sf_backward_params_partials(
    torch::Tensor grad_out,
    torch::Tensor coords,
    torch::Tensor params,
    double temperature,
    torch::Tensor gumbel) {
    CHECK_INPUT(grad_out);
    check_common(coords, params, gumbel, temperature);
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must have shape (B, M)");
    TORCH_CHECK(grad_out.size(0) == params.size(0), "grad_out B dimension must match params");
    TORCH_CHECK(grad_out.size(1) == coords.size(1), "grad_out M dimension must match coords");

    constexpr int kBlockSize = 128;
    constexpr int kNumParams = 17;
    const int param_B = static_cast<int>(params.size(0));
    const int M = static_cast<int>(coords.size(1));
    const int num_tiles = (M + kBlockSize - 1) / kBlockSize;
    auto partials = torch::empty({param_B, num_tiles, kNumParams}, params.options());
    launch_varaxis_sf_backward_params_partials(
        grad_out.data_ptr<float>(),
        coords.data_ptr<float>(),
        params.data_ptr<float>(),
        gumbel.data_ptr<float>(),
        static_cast<float>(temperature),
        static_cast<int>(coords.size(0)),
        param_B,
        M,
        num_tiles,
        partials.data_ptr<float>());
    return partials;
}

torch::Tensor varaxis_sf_reduce_param_partials(torch::Tensor partials) {
    CHECK_INPUT(partials);
    TORCH_CHECK(partials.dim() == 3 && partials.size(2) == 17,
                "partials must have shape (B, num_tiles, 17)");
    auto grad_params = torch::empty({partials.size(0), partials.size(2)}, partials.options());
    launch_varaxis_sf_reduce_param_partials(
        partials.data_ptr<float>(),
        static_cast<int>(partials.size(0)),
        static_cast<int>(partials.size(1)),
        grad_params.data_ptr<float>());
    return grad_params;
}

std::vector<torch::Tensor> varaxis_sf_assembly_forward(
    torch::Tensor coords,
    torch::Tensor params,
    torch::Tensor su_vals,
    torch::Tensor logits,
    double temperature,
    torch::Tensor inner_gumbel,
    torch::Tensor outer_gumbel) {
    check_assembly(coords, params, su_vals, logits, inner_gumbel, outer_gumbel, temperature);

    auto primitive_sdfs = torch::empty({params.size(0), coords.size(1)}, coords.options());
    auto out = torch::empty({coords.size(1)}, coords.options());
    launch_varaxis_sf_forward(
        coords.data_ptr<float>(),
        params.data_ptr<float>(),
        inner_gumbel.data_ptr<float>(),
        static_cast<float>(temperature),
        static_cast<int>(coords.size(0)),
        static_cast<int>(params.size(0)),
        static_cast<int>(coords.size(1)),
        primitive_sdfs.data_ptr<float>());
    launch_varaxis_sf_assembly_forward(
        primitive_sdfs.data_ptr<float>(),
        su_vals.data_ptr<float>(),
        logits.data_ptr<float>(),
        outer_gumbel.data_ptr<float>(),
        static_cast<float>(temperature),
        static_cast<int>(params.size(0)),
        static_cast<int>(coords.size(1)),
        out.data_ptr<float>());
    return {primitive_sdfs, out};
}

std::vector<torch::Tensor> varaxis_sf_assembly_backward_params(
    torch::Tensor grad_primitive,
    torch::Tensor grad_out,
    torch::Tensor primitive_sdfs,
    torch::Tensor coords,
    torch::Tensor params,
    torch::Tensor su_vals,
    torch::Tensor logits,
    double temperature,
    torch::Tensor inner_gumbel,
    torch::Tensor outer_gumbel) {
    CHECK_INPUT(grad_primitive);
    CHECK_INPUT(grad_out);
    CHECK_INPUT(primitive_sdfs);
    check_assembly(coords, params, su_vals, logits, inner_gumbel, outer_gumbel, temperature);
    TORCH_CHECK(grad_primitive.dim() == 2, "grad_primitive must have shape (K, M)");
    TORCH_CHECK(grad_primitive.size(0) == params.size(0), "grad_primitive K dimension must match params");
    TORCH_CHECK(grad_primitive.size(1) == coords.size(1), "grad_primitive M dimension must match coords");
    TORCH_CHECK(primitive_sdfs.dim() == 2, "primitive_sdfs must have shape (K, M)");
    TORCH_CHECK(primitive_sdfs.size(0) == params.size(0), "primitive_sdfs K dimension must match params");
    TORCH_CHECK(primitive_sdfs.size(1) == coords.size(1), "primitive_sdfs M dimension must match coords");
    TORCH_CHECK(grad_out.dim() == 1 && grad_out.size(0) == coords.size(1),
                "grad_out must have shape (M,)");

    auto grad_raw = torch::empty_like(primitive_sdfs);
    auto grad_params = torch::zeros_like(params);
    auto grad_su = torch::zeros_like(su_vals);
    auto grad_logits = torch::zeros_like(logits);
    launch_varaxis_sf_assembly_backward_prep(
        grad_primitive.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        primitive_sdfs.data_ptr<float>(),
        su_vals.data_ptr<float>(),
        logits.data_ptr<float>(),
        outer_gumbel.data_ptr<float>(),
        static_cast<float>(temperature),
        static_cast<int>(params.size(0)),
        static_cast<int>(coords.size(1)),
        grad_raw.data_ptr<float>(),
        grad_su.data_ptr<float>(),
        grad_logits.data_ptr<float>());
    const int param_B = static_cast<int>(params.size(0));
    const int M = static_cast<int>(coords.size(1));
    if (param_B < reduced_backward_min_k()) {
        launch_varaxis_sf_backward_params(
            grad_raw.data_ptr<float>(),
            coords.data_ptr<float>(),
            params.data_ptr<float>(),
            inner_gumbel.data_ptr<float>(),
            static_cast<float>(temperature),
            static_cast<int>(coords.size(0)),
            param_B,
            M,
            grad_params.data_ptr<float>());
    } else {
        constexpr int kBlockSize = 128;
        constexpr int kNumParams = 17;
        const int num_tiles = (M + kBlockSize - 1) / kBlockSize;
        auto partials = torch::empty({param_B, num_tiles, kNumParams}, params.options());
        launch_varaxis_sf_backward_params_reduced(
            grad_raw.data_ptr<float>(),
            coords.data_ptr<float>(),
            params.data_ptr<float>(),
            inner_gumbel.data_ptr<float>(),
            static_cast<float>(temperature),
            static_cast<int>(coords.size(0)),
            param_B,
            M,
            num_tiles,
            partials.data_ptr<float>(),
            grad_params.data_ptr<float>());
    }
    return {grad_params, grad_su, grad_logits};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &varaxis_sf_forward, "Fused VarAxisSF forward (CUDA)");
    m.def("backward", &varaxis_sf_backward, "Fused VarAxisSF backward (CUDA)");
    m.def("backward_params", &varaxis_sf_backward_params, "Fused VarAxisSF params-only backward (CUDA)");
    m.def("backward_params_partials", &varaxis_sf_backward_params_partials,
          "Diagnostic VarAxisSF params-only backward partials (CUDA)");
    m.def("reduce_param_partials", &varaxis_sf_reduce_param_partials,
          "Diagnostic VarAxisSF params-only partial-gradient reduction (CUDA)");
    m.def("assembly_forward", &varaxis_sf_assembly_forward, "Fused CustomVASF assembly forward (CUDA)");
    m.def("assembly_backward_prep", &varaxis_sf_assembly_backward_prep_only,
          "Diagnostic CustomVASF assembly backward prep (CUDA)");
    m.def("assembly_backward_params", &varaxis_sf_assembly_backward_params,
          "Fused CustomVASF assembly params-only backward prep plus VarAxisSF backward (CUDA)");
}
