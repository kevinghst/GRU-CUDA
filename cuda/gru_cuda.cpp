#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> gru_cuda_forward(
    torch::Tensor input,
    torch::Tensor x2h_w,
    torch::Tensor h2h_w,
    torch::Tensor x2h_b,
    torch::Tensor h2h_b,
    torch::Tensor old_h);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> gru_forward(
    torch::Tensor input,
    torch::Tensor x2h_w,
    torch::Tensor h2h_w,
    torch::Tensor x2h_b,
    torch::Tensor h2h_b,
    torch::Tensor old_h) {
  CHECK_INPUT(input);
  CHECK_INPUT(x2h_w);
  CHECK_INPUT(h2h_w);
  CHECK_INPUT(x2h_b);
  CHECK_INPUT(h2h_b);
  CHECK_INPUT(old_h);

  return gru_cuda_forward(input, x2h_w, h2h_w, x2h_b, h2h_b, old_h);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gru_forward, "GRU forward (CUDA)");
}