#include <torch/extension.h>

#include <vector>

// s'(z) = (1 - s(z)) * s(z)
torch::Tensor d_sigmoid(torch::Tensor s) {
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor t) {
  return 1 - (t * t);
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
            new_h, resetgate, inputgate, newgate, gate_h_weights
        };
}

std::vector<torch::Tensor> gru_backward(
    torch::Tensor grad_hy,
    torch::Tensor rg,
    torch::Tensor ig,
    torch::Tensor ng,
    torch::Tensor gate_h_weights,
    torch::Tensor hx,
    torch::Tensor x,
    torch::Tensor x2h_w,
    torch::Tensor h2h_w) {
      auto gate_h = gate_h_weights.chunk(3, 1);

      auto gig = d_sigmoid(ig) * grad_hy * (hx - ng);
      auto gin = d_tanh(ng) * grad_hy * (1 - ig);
      auto ghn = gin * rg;
      auto grg = d_sigmoid(rg) * gin * gate_h[2];

      auto grad_hidden_gates = 
        torch::cat({grg, gig, ghn}, /*dim=*/1);

      auto grad_input_gates =
        torch::cat({grg, gig, gin}, /*dim=*/1);

      auto grad_hidden_weights = grad_hidden_gates.t().mm(hx);
      auto grad_input_weights = grad_input_gates.t().mm(x);

      auto grad_hidden_bias = grad_hidden_gates.sum(/*dim=*/0, /*keepdim*/true);
      auto grad_input_bias = grad_input_gates.sum(/*dim=*/0, /*keepdim*/true);

      auto grad_hx =  grad_hy * ig + grad_hidden_gates.mm(h2h_w);

      return {grad_input_weights, grad_hidden_weights, grad_input_bias, grad_hidden_bias, grad_hx};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gru_forward, "GRU forward");
  m.def("backward", &gru_backward, "GRU backward");
}
