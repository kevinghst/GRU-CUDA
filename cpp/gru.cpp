#include <torch/extension.h>

#include <vector>

// s'(z) = (1 - s(z)) * s(z)
torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

std::vector<torch::Tensor> gru_forward(
    torch::Tensor input,
    torch::Tensor x2h_w,
    torch::Tensor h2h_w,
    torch::Tensor x2h_b,
    torch::Tensor h2h_b,
    torch::Tensor old_h) {
        auto gate_x_weights = torch::addmm(x2h_b, input, x2h_w.transpose(0, 1));
        auto gate_h_weights = torch::addmm(h2h_b, old_h, h2h_w.transpose(0, 1));

        auto gates_x = gate_x_weights.chunk(3, 1);
        auto gates_h = gate_h_weights.chunk(3, 1);

        auto i_r = gates_x[0];
        auto i_i = gates_x[1];
        auto i_n = gates_x[2];

        auto h_r = gates_h[0];
        auto h_i = gates_h[1];
        auto h_n = gates_h[2];

        auto resetgate = torch::sigmoid(i_r + h_r);
        auto inputgate = torch::sigmoid(i_i + h_i);
        auto newgate = torch::tanh(i_n + (resetgate * h_n));

        auto new_h = newgate + inputgate * (old_h - newgate);

        return {
            new_h
        };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gru_forward, "GRU forward");
}


