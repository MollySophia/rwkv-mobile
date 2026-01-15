import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import torch.utils.cpp_extension
from rwkv_src.rwkv_v6_modules import Rwkv6SelfAttention, Rwkv6FeedForward
from rwkv_src.rwkv_v5_modules import Rwkv5SelfAttention, Rwkv5FeedForward
from rwkv_src.rwkv_v7_modules import Rwkv7SelfAttention, Rwkv7FeedForward

def check_rwkv_info(state_dict):
    n_layer = 0
    version = 5
    n_head = 0
    for k in state_dict.keys():
        layer_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
        n_layer = max(n_layer, layer_id + 1)
        if 'ln_x' in k:
            version = max(5, version)
        if 'gate.weight' in k:
            version = max(5.1, version)
        if int(version) == 5 and 'att.time_decay' in k:
            n_head = state_dict[k].shape[0]
            if len(state_dict[k].shape) > 1:
                if state_dict[k].shape[1] > 1:
                    version = max(5.2, version)
        if 'time_maa' in k:
            version = max(6, version)
        if 'r_k' in k:
            version = max(7, version)
            n_head, _ = state_dict[k].shape
        if int(version) == 6 and 'time_faaaa' in k:
            n_head = state_dict[k].shape[0]
    return version, n_layer, n_head

class RWKV_Block(nn.Module):
    def __init__(self, state_dict, n_embd, head_size, n_ffn, layer_id, layer_begin, num_layers, rescale_layer=0, version=6.0, custom_wkv=False, model_args=None):
        super().__init__()
        self.version = version
        self.layer_offset = layer_id - layer_begin
        self.model_args = model_args
        if self.version == 7:
            self.att = Rwkv7SelfAttention(state_dict, n_embd, head_size, model_args=model_args, layer_id=layer_id, custom_wkv=custom_wkv)
            self.ffn = Rwkv7FeedForward(state_dict, n_embd, n_ffn, layer_id=layer_id, num_layers=num_layers)
        elif self.version == 6:
            self.att = Rwkv6SelfAttention(state_dict, n_embd, head_size, layer_id=layer_id, rescale_layer=rescale_layer, custom_wkv=custom_wkv)
            self.ffn = Rwkv6FeedForward(state_dict, n_embd, n_ffn, layer_id=layer_id, rescale_layer=rescale_layer)
        else:
            self.att = Rwkv5SelfAttention(state_dict, n_embd, head_size, version=version, layer_id=layer_id, rescale_layer=rescale_layer)
            self.ffn = Rwkv5FeedForward(state_dict, n_embd, n_ffn, layer_id=layer_id, rescale_layer=rescale_layer)

    def forward(self, x, state=None, v_first=None):
        if len(state) == 2:
            token_shift_state, wkv_state = state
            if self.version == 7:
                x, token_shift_state[:, 2*self.layer_offset, :], wkv_state[self.layer_offset, :, :], v_first = self.att(x, token_shift_state[:, 2*self.layer_offset, :], wkv_state[self.layer_offset, :, :], v_first)
                x, token_shift_state[:, 2*self.layer_offset+1, :] = self.ffn(x, token_shift_state[:, 2*self.layer_offset+1, :])
                return x, [token_shift_state, wkv_state], v_first
            else:
                x, token_shift_state[:, 2*self.layer_offset, :], wkv_state[self.layer_offset, :, :] = self.att(x, token_shift_state[:, 2*self.layer_offset, :], wkv_state[self.layer_offset, :, :])
                x, token_shift_state[:, 2*self.layer_offset+1, :] = self.ffn(x, token_shift_state[:, 2*self.layer_offset+1, :])
                return x, [token_shift_state, wkv_state]
        else:
            if self.version == 7:
                x, state[3*self.layer_offset], state[3*self.layer_offset+1], v_first = self.att(x, state[3*self.layer_offset], state[3*self.layer_offset+1], v_first)
                x, state[3*self.layer_offset+2] = self.ffn(x, state[3*self.layer_offset+2])
                return x, state, v_first
            else:
                x, state[3*self.layer_offset], state[3*self.layer_offset+1] = self.att(x, state[3*self.layer_offset], state[3*self.layer_offset+1])
                x, state[3*self.layer_offset+2] = self.ffn(x, state[3*self.layer_offset+2])
                return x, state


class RWKV_RNN(torch.nn.Module):
    def __init__(self, args, chunks=1, chunk_idx=0):
        super().__init__()
        self.args = args
        self.eval()

        if '.pth' in args.MODEL_NAME:
            args.MODEL_NAME = args.MODEL_NAME.replace('.pth', '')
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        self.args.n_embd = w['emb.weight'].shape[1]
        self.args.vocab_size = w['emb.weight'].shape[0]
        self.args.n_att = w['blocks.0.att.key.weight'].shape[0]
        self.args.n_ffn = w['blocks.0.ffn.key.weight'].shape[0]
        self.args.version, self.args.n_layer, self.args.n_head = check_rwkv_info(w)
        self.args.head_size = self.args.n_embd // self.args.n_head

        if self.args.version == 7:
            self.args.RESCALE_LAYER = 0

        if chunk_idx == 0:
            print("Model version:", self.args.version)
            print("n_layer:", self.args.n_layer)
            print("n_embd:", self.args.n_embd)
            print("vocab_size:", self.args.vocab_size)
            print("n_att:", self.args.n_att)
            print("n_ffn:", self.args.n_ffn)

        layers_per_chunk = self.args.n_layer // chunks
        self.layer_begin = chunk_idx * layers_per_chunk
        self.layer_end = min(self.args.n_layer, (chunk_idx + 1) * layers_per_chunk)
        self.chunk_idx = chunk_idx
        self.chunks = chunks
        print(f"Chunk {chunk_idx}: layers {self.layer_begin} to {self.layer_end}")

        self.device = torch.device('cuda') if self.args.USE_CUDA and torch.cuda.is_available() else torch.device('cpu')
        self.gpu = True if self.device is not torch.device('cpu') else False

        for k in w.keys():
            if not self.args.fp16 or 'emb' in k or 'ln0' in k:
                w[k] = w[k].float()
            else:
                w[k] = w[k].half()

        emb_weight = w['emb.weight']
        emb_weight = F.layer_norm(emb_weight, emb_weight.size()[-1:], weight=w['blocks.0.ln0.weight'].flatten(), bias=w['blocks.0.ln0.bias'].flatten())
        if self.args.USE_EMBEDDING:
            self.embedding = torch.nn.Embedding.from_pretrained(emb_weight)
            if self.args.fp16:
                self.embedding.half()
        else:
            if self.args.fp16:
                self.emb_weight = emb_weight.half()
            else:
                self.emb_weight = emb_weight

        self.blocks = nn.ModuleList([RWKV_Block(w, self.args.n_embd, self.args.head_size, self.args.n_ffn,
            layer_id=i,layer_begin=self.layer_begin, num_layers=self.args.n_layer,
            rescale_layer=self.args.RESCALE_LAYER, version=self.args.version,
            custom_wkv=self.args.wkv_customop, model_args=self.args) for i in range(self.layer_begin, self.layer_end)])
        self.ln_out = nn.LayerNorm(self.args.n_embd, eps=1e-5)
        self.ln_out.weight = nn.Parameter(w['ln_out.weight'])
        self.ln_out.bias = nn.Parameter(w['ln_out.bias'])
        self.head = nn.Linear(self.args.n_embd, self.args.vocab_size, bias=False)
        self.head.weight = nn.Parameter(w['head.weight'])

        if self.gpu:
            self.to(self.device)

        if self.args.fp16:
            self.half()

    def forward(self, in0, *args):
        state = [*args]
        if len(state) == 1:
            state = state[0]
        with torch.no_grad():
            if self.args.USE_EMBEDDING and self.chunk_idx == 0:
                x = self.embedding(in0)
            else:
                x = in0
            try:
                batch_size, seq_length, _ = x.size()
            except:
                batch_size, seq_length = 1, 1

            if self.args.version == 7:
                v_first = torch.empty_like(x)

            for i in range(self.layer_begin, self.layer_end):
                if self.args.version == 7:
                    x, state, v_first = self.blocks[i-self.layer_begin](x, state, v_first=v_first)
                else:
                    x, state = self.blocks[i-self.layer_begin](x, state)
                if self.args.RESCALE_LAYER > 0:
                    if (i+1) % self.args.RESCALE_LAYER == 0:
                        x = x / 2

            if self.chunk_idx == self.chunks - 1:
                x = self.ln_out(x)
                x = self.head(x)
            else:
                x = x.view(batch_size, seq_length, self.args.n_embd)

            return x, state

class RWKV_RNN_Stateful(RWKV_RNN):
    def __init__(self, args, chunks=1, chunk_idx=0):
        super().__init__(args, chunks, chunk_idx)
        self.register_buffer('state_tokenshift', torch.zeros(2, self.args.n_layer, self.args.n_embd))
        self.register_buffer('state_wkv', torch.zeros(self.args.n_layer, self.args.n_head, self.args.head_size, self.args.head_size))
        # for i in range(self.args.n_layer):
        #     self.register_buffer(f'state_att_tokenshift_{i}', torch.zeros(1, 1, self.args.n_embd))
        #     self.register_buffer(f'state_wkv_{i}', torch.zeros(1, self.args.n_head, self.args.head_size, self.args.head_size))
        #     self.register_buffer(f'state_ffn_tokenshift_{i}', torch.zeros(1, 1, self.args.n_embd))

    def forward(self, in0):
        states = []
        for i in range(self.args.n_layer):
            states.append(self.state_tokenshift[0:1, i:i+1, :])
            states.append(self.state_wkv[i:i+1, :, :, :])
            states.append(self.state_tokenshift[1:2, i:i+1, :])
            # states.append(self._buffers[f'state_att_tokenshift_{i}'])
            # states.append(self._buffers[f'state_wkv_{i}'])
            # states.append(self._buffers[f'state_ffn_tokenshift_{i}'])
        x, states = super().forward(in0, states)
        for i in range(self.args.n_layer):
            self.state_tokenshift[0:1, i:i+1, :] = states[3*i]
            self.state_wkv[i:i+1, :, :, :] = states[3*i+1]
            self.state_tokenshift[1:2, i:i+1, :] = states[3*i+2]
            # self._buffers[f'state_att_tokenshift_{i}'] = states[3*i]
            # self._buffers[f'state_wkv_{i}'] = states[3*i+1]
            # self._buffers[f'state_ffn_tokenshift_{i}'] = states[3*i+2]
        return x

def make_chunks(chunks, args):
    return [RWKV_RNN(args, chunks=chunks, chunk_idx=i) for i in range(chunks)]
