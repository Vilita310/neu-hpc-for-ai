#include <torch/extension.h>

torch::Tensor tk_forward(torch::Tensor A, torch::Tensor B) {
    return torch::matmul(A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tk_forward, "TK MoE forward");
}
