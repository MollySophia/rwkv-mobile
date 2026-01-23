import torch
import torch.nn.functional as F
import numpy as np

def extract_info_from_model_cfg(model_cfg):
    return model_cfg.n_layer, model_cfg.n_head, model_cfg.n_embd

def get_dummy_state(batch_size, model_cfg, device, merged_states=False):
    if merged_states:
        num_layers, num_heads, embed_dim = extract_info_from_model_cfg(model_cfg)
        head_size = embed_dim // num_heads
        token_shift_state = torch.zeros(batch_size, 2 * num_layers, embed_dim).to(device=device)
        wkv_state = torch.zeros(batch_size * num_layers, num_heads, head_size, head_size).to(device=device)
        # state = torch.zeros(batch_size, num_layers * (2 + head_size), embed_dim).to(device=device)
        if model_cfg.fp16:
            token_shift_state = token_shift_state.half()
            wkv_state = wkv_state.half()
        return [token_shift_state, wkv_state]
    else:
        def _cache(shape, fp16):
            if fp16:
                return torch.zeros(shape, dtype=torch.float16).to(device=device)
            else:
                return torch.zeros(shape).to(device=device)

        num_layers, num_heads, embed_dim = extract_info_from_model_cfg(model_cfg)
        head_size = embed_dim // num_heads

        state_0 = (1, batch_size, embed_dim)
        state_1 = (batch_size, num_heads, head_size, head_size)
        state_2 = (1, batch_size, embed_dim)

        state = []
        for _ in range(0, num_layers):
            state += [_cache(state_0, model_cfg.fp16), _cache(state_1, model_cfg.fp16), _cache(state_2, model_cfg.fp16)]
        return state

def get_dummy_input_for_rwkv_causal_llm(batch_size, input_length, device, model_cfg=None, dict_output=False, merged_states=False):
    input_ids = torch.tensor([[0]*input_length for _ in range(batch_size)], dtype=torch.int32).to(device)
    if dict_output:
        inputs = {'in0': input_ids, 'state': get_dummy_state(batch_size, model_cfg, device, merged_states)}
    else:
        inputs = [input_ids] + get_dummy_state(batch_size, model_cfg, device, merged_states)
    return inputs

def init_inputs_for_rwkv_onnx(n_heads, head_size, n_layers, batch_size, input_length, dtype=np.float16):
    n_embed = n_heads * head_size
    input_ids = np.zeros((batch_size, input_length), dtype=np.int64)
    inputs = {'input_ids': input_ids}
    for i in range(n_layers):
        inputs[f'layer{i}_state0_in'] = np.zeros((batch_size, n_embed), dtype=dtype)
        inputs[f'layer{i}_state1_in'] = np.zeros((batch_size, n_heads, head_size, head_size), dtype=dtype)
        inputs[f'layer{i}_state2_in'] = np.zeros((batch_size, n_embed), dtype=dtype)
    return inputs

def to_device(t, device):
    if isinstance(t, torch.Tensor):
        return t.detach().clone().to(device)
    if isinstance(t, tuple):
        return tuple([to_device(i, device) for i in t])
    if isinstance(t, list):
        return [to_device(i, device) for i in t]
    if isinstance(t, dict):
        return {k:to_device(v, device) for k,v in t.items()}
    return t

def to_cpu(t):
    return to_device(t, torch.device('cpu'))

def get_input_output_names(model_cfg):
    num_layers, _, _ = extract_info_from_model_cfg(model_cfg)
    def _get_state_names(sfx, n_layers):
        all = []
        for i in range(n_layers):
            for j in range(0, 3):
                all.extend([f'layer{i}_state{j}_{sfx}'])
        return all

    input_names = ['input_ids']
    input_names += _get_state_names('in', num_layers)
    output_names = ['logits']
    output_names += _get_state_names('out', num_layers)
    return input_names, output_names

def sample_logits(out, temperature=1.0, top_p=0.8, top_k=128):
    # probs = F.softmax(out, dim=-1).squeeze().cpu().numpy()
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    if type(out) == torch.Tensor:
        out = out.squeeze().cpu().numpy()
    probs = softmax(out)
    if top_k == 0:
        return np.argmax(probs)
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    cutoff = sorted_probs[top_k]
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out