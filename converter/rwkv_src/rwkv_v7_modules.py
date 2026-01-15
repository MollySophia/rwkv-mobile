import torch
import torch.nn as nn
import torch.nn.functional as F
import rwkv_src.elemwise_ops as op

custom_norm_wrapper_src = """
#include <torch/extension.h>
#include <torch/script.h>

torch::Tensor l2norm(torch::Tensor x) {
    return x / (x.norm(2, -1, true) + 1e-6);
}

torch::Tensor reducel2(torch::Tensor x) {
    return x.norm(2, -1, true);
}

TORCH_LIBRARY(customop, m) {
    m.def("l2norm", &l2norm);
    m.def("reducel2", &reducel2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
"""

_ = torch.utils.cpp_extension.load_inline(
    name='extension', cpp_sources=[custom_norm_wrapper_src])

custom_wkv_src = """
#include <torch/extension.h>
#include <torch/script.h>

std::tuple<torch::Tensor, torch::Tensor> wkv7(
    torch::Tensor r, torch::Tensor w, torch::Tensor k,
    torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor state) {

    auto num_head = state.size(0);
    auto head_size = state.size(1);
    int seq_length = k.size(0);

    auto kv = torch::matmul(v, k);
    auto ab = torch::matmul(a, b);
    auto wkv_tensor = torch::zeros({seq_length, num_head, 1, head_size}, kv.options());
    for (int i = 0; i < seq_length; i++) {
        state = w[i] * state + kv[i] + torch::matmul(state, ab[i]);
        wkv_tensor[i] = torch::matmul(state, r[i]).reshape({num_head, 1, head_size});
    }

    return std::make_tuple(wkv_tensor, state);
}

TORCH_LIBRARY(rwkv, m) {
  m.def("wkv7", &wkv7);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}

"""

_ = torch.utils.cpp_extension.load_inline(
    name='extension', cpp_sources=[custom_wkv_src])

def l2norm(x):
    return x / (torch.ops.customop.reducel2(x) + 1e-6)

class Rwkv7SelfAttention(nn.Module):
    def __init__(self, state_dict, hidden_size, head_size, model_args=None, layer_id=0, rescale_layer=0, custom_wkv=False):
        super().__init__()
        prefix = f'blocks.{layer_id}.att.'
        self.layer_id = layer_id
        self.num_heads = hidden_size // head_size
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.custom_wkv = custom_wkv
        self.model_args = model_args

        if layer_id == 0:
            state_dict[prefix + 'v0'] = state_dict[prefix + 'a0']
            state_dict[prefix + 'v1'] = state_dict[prefix + 'a1']
            state_dict[prefix + 'v2'] = state_dict[prefix + 'a2']

        self.D_DECAY_LORA = state_dict[prefix + 'w1'].shape[-1]
        self.D_AAA_LORA = state_dict[prefix + 'a1'].shape[-1]
        self.D_MV_LORA = state_dict[prefix + 'v1'].shape[-1]
        self.D_GATE_LORA = state_dict[prefix + 'g1'].shape[-1]

        self.x_r = nn.Parameter(state_dict[prefix + 'x_r'])
        self.x_w = nn.Parameter(state_dict[prefix + 'x_w'])
        self.x_k = nn.Parameter(state_dict[prefix + 'x_k'])
        self.x_v = nn.Parameter(state_dict[prefix + 'x_v'])
        self.x_a = nn.Parameter(state_dict[prefix + 'x_a'])
        self.x_g = nn.Parameter(state_dict[prefix + 'x_g'])

        self.time_decay = nn.Parameter(state_dict[prefix + 'w0'])

        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance.weight = nn.Parameter(state_dict[prefix + 'receptance.weight'])
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key.weight = nn.Parameter(state_dict[prefix + 'key.weight'])
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'])

        self.matmul_time_decay_w1   = nn.Linear(hidden_size, self.D_DECAY_LORA, bias=False)
        self.matmul_time_decay_w1.weight = nn.Parameter(state_dict[prefix + 'w1'].t())
        self.matmul_time_decay_w2   = nn.Linear(self.D_DECAY_LORA, hidden_size, bias=False)
        self.matmul_time_decay_w2.weight = nn.Parameter(state_dict[prefix + 'w2'].t())

        self.a0 = nn.Parameter(state_dict[prefix + 'a0'])
        self.matmul_a1 = nn.Linear(hidden_size, self.D_AAA_LORA, bias=False)
        self.matmul_a1.weight = nn.Parameter(state_dict[prefix + 'a1'].t())
        self.matmul_a2 = nn.Linear(self.D_AAA_LORA, hidden_size, bias=False)
        self.matmul_a2.weight = nn.Parameter(state_dict[prefix + 'a2'].t())

        self.v0 = nn.Parameter(state_dict[prefix + 'v0'])
        self.matmul_v1 = nn.Linear(hidden_size, self.D_MV_LORA, bias=False)
        self.matmul_v1.weight = nn.Parameter(state_dict[prefix + 'v1'].t())
        self.matmul_v2 = nn.Linear(self.D_MV_LORA, hidden_size, bias=False)
        self.matmul_v2.weight = nn.Parameter(state_dict[prefix + 'v2'].t())

        self.matmul_g1 = nn.Linear(hidden_size, self.D_GATE_LORA, bias=False)
        self.matmul_g1.weight = nn.Parameter(state_dict[prefix + 'g1'].t())
        self.matmul_g2 = nn.Linear(self.D_GATE_LORA, hidden_size, bias=False)
        self.matmul_g2.weight = nn.Parameter(state_dict[prefix + 'g2'].t())

        self.k_k = nn.Parameter(state_dict[prefix + 'k_k'])
        self.k_a = nn.Parameter(state_dict[prefix + 'k_a'])
        self.r_k = nn.Parameter(state_dict[prefix + 'r_k'].flatten())

        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output.weight = nn.Parameter(state_dict[prefix + 'output.weight'])

        self.ln_x = nn.InstanceNorm2d(self.num_heads, eps=64e-5)
        # self.ln_x = nn.LayerNorm(self.head_size, eps=64e-5)

        self.ln_x_w = nn.Parameter(state_dict[prefix + 'ln_x.weight'])
        self.ln_x_b = nn.Parameter(state_dict[prefix + 'ln_x.bias'])
        self.mul_ln_x = op.Multiply()
        self.add_ln_x = op.Add()

        self.ln_1                   = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ln_1.weight            = nn.Parameter(state_dict[f'blocks.{layer_id}.ln1.weight'])
        self.ln_1.bias              = nn.Parameter(state_dict[f'blocks.{layer_id}.ln1.bias'])
        self.add_attention          = op.Add()
        self.mul_attention          = op.Multiply()
        self.sub_shifted            = op.Subtract()
        self.add_time_decay0        = op.Add()

        self.tanh_w                 = op.Tanh()
        self.exp_w                  = op.Exponential()
        self.sigmoid_a              = nn.Sigmoid()
        self.sigmoid_g              = nn.Sigmoid()
        self.sigmoid_v              = nn.Sigmoid()
        self.sigmoid_w              = nn.Sigmoid()

    def forward(self, x, state1, state2, v_first):
        last_x = x
        x = self.ln_1(x)
        batch_size, seq_length, hidden_size = x.size()
        assert batch_size == 1
        if seq_length == 1:
            state1_out = x
            sx = self.sub_shifted(state1, x)
        else:
            past = torch.cat([state1, x[:, :-1, :]], dim=1)
            sx = self.sub_shifted(past, x)
            state1_out = x[:, -1, :] + torch.finfo(torch.float32).smallest_normal

        xr = x + self.x_r * sx
        xw = x + self.x_w * sx
        xk = x + self.x_k * sx
        xv = x + self.x_v * sx
        xa = x + self.x_a * sx
        xg = x + self.x_g * sx

        receptance = self.receptance(xr)
        key = self.key(xk)
        value = self.value(xv)
        gate = self.matmul_g2(self.sigmoid_g(self.matmul_g1(xg)))
        a = self.sigmoid_a(self.a0 + self.matmul_a2(self.matmul_a1(xa)))
        time_decay = self.matmul_time_decay_w2(self.tanh_w(self.matmul_time_decay_w1(xw)))

        kk = key * self.k_k
        if self.model_args is not None and self.model_args.USE_ONNX_L2NORM:
            kk = torch.ops.customop.l2norm(kk.view(seq_length, self.num_heads, self.head_size)).view(batch_size, seq_length, hidden_size)
        elif self.model_args is not None and self.model_args.USE_ONNX_REDUCE_L2:
            kk = l2norm(kk.view(seq_length, self.num_heads, self.head_size)).view(batch_size, seq_length, hidden_size)
        else:
            kk = torch.nn.functional.normalize(kk.view(seq_length, self.num_heads, self.head_size), dim=-1, p=2.0, eps=1e-6).view(batch_size, seq_length, hidden_size)
        key = key * (1 + (a-1) * self.k_a)

        if self.layer_id == 0:
            v_first = value
        else:
            value = value + (v_first - value) * self.sigmoid_v(self.v0 + self.matmul_v2(self.matmul_v1(xv)))

        time_decay = self.add_time_decay0(self.time_decay, time_decay)
        time_decay = self.exp_w(-0.606531 * self.sigmoid_w(time_decay)).view(seq_length, self.num_heads, 1, self.head_size)

        # kernel
        if self.model_args is not None and self.model_args.USE_CUSTOM_WKV:
            b = kk * a
            a = -kk
            x, state2_out = torch.ops.rwkv.wkv7(receptance.view(seq_length, self.num_heads, self.head_size, 1),
                                                time_decay.view(seq_length, self.num_heads, 1, self.head_size),
                                                key.view(seq_length, self.num_heads, 1, self.head_size),
                                                value.view(seq_length, self.num_heads, self.head_size, 1),
                                                a.view(seq_length, self.num_heads, self.head_size, 1),
                                                b.view(seq_length, self.num_heads, 1, self.head_size),
                                                state2.squeeze())
        else:
            b = (kk * a).view(seq_length, self.num_heads, 1, self.head_size)
            a = (-kk).view(seq_length, self.num_heads, self.head_size, 1)
            v = value.view(seq_length, self.num_heads, self.head_size, 1)
            k = key.view(seq_length, self.num_heads, 1, self.head_size)
            r = receptance.view(seq_length, self.num_heads, self.head_size, 1)
            if seq_length == 1:
                vk = v @ k
                state2_out = state2 * time_decay + state2 @ a @ b + vk
                x = (state2_out @ r).view(seq_length, self.num_heads, 1, self.head_size)
            else:
                x_list = []
                state2_out = state2
                v_list = torch.split(v, 1, dim=0)
                k_list = torch.split(k, 1, dim=0)
                r_list = torch.split(r, 1, dim=0)
                a_list = torch.split(a, 1, dim=0)
                b_list = torch.split(b, 1, dim=0)
                time_decay_list = torch.split(time_decay, 1, dim=0)
                for i in range(seq_length):
                    vk = v_list[i] @ k_list[i]
                    state2_out = state2_out * time_decay_list[i] + state2_out @ a_list[i] @ b_list[i] + vk
                    x_list.append(state2_out @ r_list[i])
                x = torch.cat(x_list, dim=0)

        # group_norm
        x = self.ln_x(x).view(batch_size, seq_length, self.hidden_size)
        x = self.mul_ln_x(x, self.ln_x_w)
        x = self.add_ln_x(x, self.ln_x_b)

        x = x + ((receptance * key * self.r_k).view(seq_length, self.num_heads, self.head_size).sum(dim=-1, keepdim=True) * value.view(seq_length, self.num_heads, self.head_size)).view(seq_length, self.hidden_size)
        x = self.mul_attention(x, gate)
        x = self.output(x)

        return self.add_attention(last_x, x), state1_out, state2_out, v_first

class Rwkv7FeedForward(nn.Module):
    def __init__(self, state_dict, hidden_size, intermediate_size, layer_id=0, num_layers=0):
        super().__init__()
        prefix = f'blocks.{layer_id}.ffn.'
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.x_k = nn.Parameter(state_dict[prefix + 'x_k'])

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.key.weight = nn.Parameter(state_dict[prefix + 'key.weight'])
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'])

        self.ln_2                   = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ln_2.weight            = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.weight'])
        self.ln_2.bias              = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.bias'])

        self.sub_shifted            = op.Subtract()
        self.mul_x_k                = op.Multiply()
        self.add_x_k                = op.Add()
        self.relu                   = nn.ReLU()
        self.pow                    = op.Pow()
        self.add_feed_forward       = op.Add()

    def forward(self, x, state):
        last_x = x
        x = self.ln_2(x)
        batch_size, seq_length, _ = x.size()
        assert batch_size == 1
        if seq_length == 1:
            state_out = x
            sx = self.sub_shifted(state, x)
        else:
            past = torch.cat([state, x[:, :-1, :]], dim=1)
            sx = self.sub_shifted(past, x)
            # mystery trick for coreml
            state_out = x[:, -1, :] + torch.finfo(torch.float32).smallest_normal
            if self.layer_id == self.num_layers - 1:
                sx = sx[:, -1, :]
                x = x[:, -1, :]
                last_x = last_x[:, -1, :]

        xk = self.add_x_k(x, self.mul_x_k(sx, self.x_k))

        key = self.pow(self.relu(self.key(xk)), 2)
        value = self.value(key)

        return self.add_feed_forward(value, last_x), state_out