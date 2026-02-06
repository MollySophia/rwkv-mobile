import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.cpp_extension
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

def _compute_chunk_boundaries(n_layer, chunks):
    assert chunks > 0, "chunks must be >= 1"
    base_layers = n_layer // chunks
    extra_layers = n_layer % chunks
    boundaries = []
    for i in range(chunks):
        if i < extra_layers:
            layers_in_chunk = base_layers + 1
            layer_begin = i * layers_in_chunk
        else:
            layers_in_chunk = base_layers
            layer_begin = extra_layers * (base_layers + 1) + (i - extra_layers) * base_layers
        layer_end = min(n_layer, layer_begin + layers_in_chunk)
        boundaries.append((layer_begin, layer_end))
    return boundaries

def _clone_block_with_offset(block, layer_offset):
    new_block = RWKV_Block.__new__(RWKV_Block)
    nn.Module.__init__(new_block)
    new_block.version = block.version
    new_block.layer_offset = layer_offset
    new_block.model_args = block.model_args
    new_block.att = block.att
    new_block.ffn = block.ffn
    return new_block

class RWKV_Block(nn.Module):
    def __init__(self, state_dict, n_embd, head_size, n_ffn, layer_id, layer_begin, num_layers, version=6.0, model_args=None):
        super().__init__()
        self.version = version
        self.layer_offset = layer_id - layer_begin
        self.model_args = model_args
        if self.version == 7:
            self.att = Rwkv7SelfAttention(state_dict, n_embd, head_size, model_args=model_args, layer_id=layer_id)
            self.ffn = Rwkv7FeedForward(state_dict, n_embd, n_ffn, layer_id=layer_id, num_layers=num_layers)
        else:
            assert False, "Unsupported version"

    def forward(self, x, state=None, v_first=None):
        if len(state) == 2:
            token_shift_state, wkv_state = state
            x, token_shift_state[:, 2*self.layer_offset, :], wkv_state[self.layer_offset, :, :], v_first = self.att(x, token_shift_state[:, 2*self.layer_offset, :], wkv_state[self.layer_offset, :, :], v_first)
            x, token_shift_state[:, 2*self.layer_offset+1, :] = self.ffn(x, token_shift_state[:, 2*self.layer_offset+1, :])
            return x, [token_shift_state, wkv_state], v_first
        else:
            x, state[3*self.layer_offset], state[3*self.layer_offset+1], v_first = self.att(x, state[3*self.layer_offset], state[3*self.layer_offset+1], v_first)
            x, state[3*self.layer_offset+2] = self.ffn(x, state[3*self.layer_offset+2])
            return x, state, v_first


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

        if chunk_idx == 0:
            print("Model version:", self.args.version)
            print("n_layer:", self.args.n_layer)
            print("n_embd:", self.args.n_embd)
            print("vocab_size:", self.args.vocab_size)
            print("n_att:", self.args.n_att)
            print("n_ffn:", self.args.n_ffn)

        assert self.args.version == 7, "Only version 7 is supported"

        boundaries = _compute_chunk_boundaries(self.args.n_layer, chunks)
        self.layer_begin, self.layer_end = boundaries[chunk_idx]
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

        if self.chunk_idx == 0:
            emb_weight = w['emb.weight']
            emb_weight = F.layer_norm(emb_weight, emb_weight.size()[-1:], weight=w['blocks.0.ln0.weight'].flatten(), bias=w['blocks.0.ln0.bias'].flatten()).detach()
            if self.args.USE_EMBEDDING:
                self.embedding = torch.nn.Embedding(emb_weight.shape[0], emb_weight.shape[1])
                self.embedding.weight = nn.Parameter(emb_weight)
                if self.args.fp16:
                    self.embedding.half()
            else:
                if self.args.fp16:
                    self.emb_weight = emb_weight.half()
                else:
                    self.emb_weight = emb_weight

        self.blocks = nn.ModuleList([RWKV_Block(w, self.args.n_embd, self.args.head_size, self.args.n_ffn,
            layer_id=i,layer_begin=self.layer_begin, num_layers=self.args.n_layer,
            version=self.args.version,
            model_args=self.args) for i in range(self.layer_begin, self.layer_end)])

        if self.chunk_idx == chunks - 1 and not self.args.SKIP_LMHEAD:
            self.ln_out = nn.LayerNorm(self.args.n_embd, eps=1e-5)
            self.ln_out.weight = nn.Parameter(w['ln_out.weight'])
            self.ln_out.bias = nn.Parameter(w['ln_out.bias'])
            self.head = nn.Linear(self.args.n_embd, self.args.vocab_size, bias=False)
            self.head.weight = nn.Parameter(w['head.weight'])

        if self.gpu:
            self.to(self.device)

        if self.args.fp16:
            self.half()

    def forward(self, in0, state, v_first=None):
        grad_ctx = torch.enable_grad() if getattr(self.args, "USE_GRAD", False) else torch.no_grad()
        with grad_ctx:
            if self.args.USE_EMBEDDING and self.chunk_idx == 0:
                x = self.embedding(in0)
            else:
                x = in0

            # try:
            #     batch_size, seq_length, _ = x.size()
            # except:
            #     batch_size, seq_length = 1, 1

            for i in range(self.layer_begin, self.layer_end):
                x, state, v_first = self.blocks[i-self.layer_begin](x, state, v_first=v_first)

            if self.chunk_idx == self.chunks - 1:
                if not self.args.SKIP_LMHEAD:
                    x = self.ln_out(x)
                    x = self.head(x)
            # else:
            #     x = x.view(batch_size, seq_length, self.args.n_embd)

            if self.chunk_idx == 0 and self.chunks != 1:
                return x, state, v_first
            else:
                return x, state

class RWKV_RNN_Stateful(RWKV_RNN):
    def __init__(self, args, chunks=1, chunk_idx=0):
        super().__init__(args, chunks, chunk_idx)
        self.layers_this_chunk = self.layer_end - self.layer_begin
        self.register_buffer('state_tokenshift', torch.zeros(2, self.layers_this_chunk, self.args.n_embd))
        self.register_buffer('state_wkv', torch.zeros(self.layers_this_chunk, self.args.n_head, self.args.head_size, self.args.head_size))

    def forward(self, in0, v_first=None):
        states = []
        for i in range(self.layers_this_chunk):
            states.append(self.state_tokenshift[0:1, i:i+1, :])
            states.append(self.state_wkv[i:i+1, :, :, :])
            states.append(self.state_tokenshift[1:2, i:i+1, :])

        outputs = super().forward(in0, states, v_first)
        if len(outputs) == 2:
            x, states = outputs
        else:
            x, states, v_first = outputs
        for i in range(self.layers_this_chunk):
            self.state_tokenshift[0:1, i:i+1, :] = states[3*i] + torch.finfo(torch.float32).smallest_normal
            self.state_wkv[i:i+1, :, :, :] = states[3*i+1] + torch.finfo(torch.float32).smallest_normal
            self.state_tokenshift[1:2, i:i+1, :] = states[3*i+2] + torch.finfo(torch.float32).smallest_normal

        if self.chunk_idx == 0 and self.chunks != 1:
            return x, v_first
        else:
            return x

    @classmethod
    def from_full_model(cls, full_model, chunks=1, chunk_idx=0):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.args = full_model.args
        obj.eval()

        boundaries = _compute_chunk_boundaries(obj.args.n_layer, chunks)
        obj.layer_begin, obj.layer_end = boundaries[chunk_idx]
        obj.layers_this_chunk = obj.layer_end - obj.layer_begin
        obj.chunk_idx = chunk_idx
        obj.chunks = chunks

        obj.device = getattr(full_model, "device", torch.device("cpu"))
        obj.gpu = getattr(full_model, "gpu", obj.device is not torch.device("cpu"))

        if chunk_idx == 0:
            if obj.args.USE_EMBEDDING and hasattr(full_model, "embedding"):
                obj.embedding = full_model.embedding
            elif hasattr(full_model, "emb_weight"):
                obj.emb_weight = full_model.emb_weight

        blocks = []
        for i in range(obj.layer_begin, obj.layer_end):
            blocks.append(_clone_block_with_offset(full_model.blocks[i], i - obj.layer_begin))
        obj.blocks = nn.ModuleList(blocks)

        if chunk_idx == chunks - 1 and not obj.args.SKIP_LMHEAD:
            obj.ln_out = full_model.ln_out
            obj.head = full_model.head

        obj.register_buffer(
            'state_tokenshift',
            torch.zeros(2, obj.layers_this_chunk, obj.args.n_embd, device=obj.device),
        )
        obj.register_buffer(
            'state_wkv',
            torch.zeros(obj.layers_this_chunk, obj.args.n_head, obj.args.head_size, obj.args.head_size, device=obj.device),
        )
        return obj

class RWKV_LMHead(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval()

        if '.pth' in args.MODEL_NAME:
            args.MODEL_NAME = args.MODEL_NAME.replace('.pth', '')
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        self.args.n_embd = w['emb.weight'].shape[1]
        self.args.vocab_size = w['emb.weight'].shape[0]
        self.ln_out = nn.LayerNorm(self.args.n_embd, eps=1e-5)
        self.ln_out.weight = nn.Parameter(w['ln_out.weight'])
        self.ln_out.bias = nn.Parameter(w['ln_out.bias'])
        self.head = nn.Linear(self.args.n_embd, self.args.vocab_size, bias=False)
        self.head.weight = nn.Parameter(w['head.weight'])

        self.device = torch.device('cuda') if self.args.USE_CUDA and torch.cuda.is_available() else torch.device('cpu')
        self.gpu = True if self.device is not torch.device('cpu') else False

        if self.gpu:
            self.to(self.device)

        if self.args.fp16:
            self.half()
        else:
            self.float()
    
    def forward(self, x):
        x = self.ln_out(x)
        x = self.head(x)
        return x

def make_chunks(chunks, args):
    return [RWKV_RNN(args, chunks=chunks, chunk_idx=i) for i in range(chunks)]

def make_chunks_stateful(chunks, args, full_model=None):
    if full_model is None:
        return [RWKV_RNN_Stateful(args, chunks=chunks, chunk_idx=i) for i in range(chunks)]
    return [RWKV_RNN_Stateful.from_full_model(full_model, chunks=chunks, chunk_idx=i) for i in range(chunks)]