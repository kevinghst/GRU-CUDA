#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}
  
template <typename scalar_t>
__global__ void gru_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gate_x,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gate_h,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> old_h,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> resetgate,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> inputgate,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> newgate,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_h) {
      //batch index
      const int n = blockIdx.y;
      // column index
      const int c = blockIdx.x * blockDim.x + threadIdx.x;

      if (c < gate_x.size(2) && c < gate_h.size(2)) {
        resetgate[n][c] = sigmoid(gate_x[n][0][c] + gate_h[n][0][c]);
        inputgate[n][c] = sigmoid(gate_x[n][1][c] + gate_h[n][1][c]);
        newgate[n][c] = tanh(gate_x[n][2][c] + resetgate[n][c] * gate_h[n][2][c]);
        new_h[n][c] = newgate[n][c] + inputgate[n][c] * (old_h[n][c] - newgate[n][c]);
      }
}

template <typename scalar_t>
__global__ void gru_cuda_backward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_hx,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_hidden_gates,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_input_gates,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_hy,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> ig,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> hx,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> ng,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> rg,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gate_h) {
      //batch index
      const int n = blockIdx.y;
      //column index
      const int c = blockIdx.x * blockDim.x + threadIdx.x;

      if (c < d_hidden_gates.size(2)) {
        const auto gig = d_sigmoid(ig[n][c]) * grad_hy[n][c] * (hx[n][c] - ng[n][c]);
        const auto gin = d_tanh(ng[n][c]) * grad_hy[n][c] * (1 - ig[n][c]);
        const auto ghn = gin * rg[n][c];
        const auto grg = d_sigmoid(rg[n][c]) * gin * gate_h[n][2][c];

        d_hidden_gates[n][0][c] = grg;
        d_hidden_gates[n][1][c] = gig;
        d_hidden_gates[n][2][c] = ghn;

        d_input_gates[n][0][c] = grg;
        d_input_gates[n][1][c] = gig;
        d_input_gates[n][2][c] = gin;
      }
}

} // namespace

std::vector<torch::Tensor> gru_cuda_forward(
    torch::Tensor input,
    torch::Tensor x2h_w,
    torch::Tensor h2h_w,
    torch::Tensor x2h_b,
    torch::Tensor h2h_b,
    torch::Tensor old_h) {
        auto gate_x_weights = torch::addmm(x2h_b, input, x2h_w.transpose(0, 1));
        auto gate_h_weights = torch::addmm(h2h_b, old_h, h2h_w.transpose(0, 1));

        const auto batch_size = old_h.size(0);
        const auto state_size = old_h.size(1);

        auto gate_x = gate_x_weights.reshape({batch_size, 3, state_size});
        auto gate_h = gate_h_weights.reshape({batch_size, 3, state_size});

        auto resetgate = torch::zeros_like(old_h);
        auto inputgate = torch::zeros_like(old_h);
        auto newgate = torch::zeros_like(old_h);
        auto new_h = torch::zeros_like(old_h);

        const int threads = 1024;
        const dim3 blocks((state_size + threads - 1) / threads, batch_size);

        AT_DISPATCH_FLOATING_TYPES(gate_x.type(), "gru_forward_cuda", ([&] {
            gru_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                gate_x.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                gate_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                old_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                resetgate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                inputgate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                newgate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                new_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
          }));
        

        return {
          new_h, resetgate, inputgate, newgate, gate_h
        };
}

std::vector<torch::Tensor> gru_cuda_backward(
  torch::Tensor grad_hy,
  torch::Tensor rg,
  torch::Tensor ig,
  torch::Tensor ng,
  torch::Tensor gate_h,
  torch::Tensor hx,
  torch::Tensor x,
  torch::Tensor x2h_w,
  torch::Tensor h2h_w) {

    // gates.shape = {batch_size, 3, state_size}
    auto d_hidden_gates = torch::zeros_like(gate_h);
    auto d_input_gates = torch::zeros_like(gate_h);
    auto d_hx = torch::zeros_like(grad_hy);

    const auto batch_size = grad_hy.size(0);
    const auto state_size = grad_hy.size(1);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "gru_forward_cuda", ([&] {
      gru_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
          d_hx.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_hidden_gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
          d_input_gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
          grad_hy.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          ig.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          hx.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          ng.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          rg.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          gate_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
    }));

    auto d_hidden_gates_weights = d_hidden_gates.flatten(1,2);
    auto d_input_gates_weights = d_input_gates.flatten(1,2);

    auto d_hidden_weights = d_hidden_gates_weights.t().mm(hx);
    auto d_input_weights = d_input_gates_weights.t().mm(x);

    auto d_hidden_bias = d_hidden_gates_weights.sum(0, true);
    auto d_input_bias = d_input_gates_weights.sum(0, true);

    auto grad_hx =  grad_hy * ig + d_hidden_gates_weights.mm(h2h_w);

    return {d_input_weights, d_hidden_weights, d_input_bias, d_hidden_bias, grad_hx};
}

