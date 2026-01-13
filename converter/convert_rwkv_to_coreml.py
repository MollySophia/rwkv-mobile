from rwkv_src.rwkv_modeling import RWKV_RNN, RWKV_RNN_Stateful
from rwkv_src.model_utils import get_dummy_input_for_rwkv_causal_llm
import coremltools as ct
from coremltools.optimize.torch.quantization import PostTrainingQuantizer, PostTrainingQuantizerConfig
from coremltools.optimize.torch.palettization import PostTrainingPalettizer, PostTrainingPalettizerConfig
from pathlib import Path
import argparse, types, os
import torch
from transformers import AutoTokenizer
import numpy as np

from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb

parser = argparse.ArgumentParser(description='Export coreml model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('--stateful', action='store_true', help='Use stateful model')
parser.add_argument('--customop', action='store_true', help='Use composite custom op for wkv7')
parser.add_argument('--int8', action='store_true', help='Use int8 quantization')
parser.add_argument('--int4', action='store_true', help='Use int4 quantization')
parser.add_argument('--lut8', action='store_true', help='Use lut8 palettization')
parser.add_argument('--lut4', action='store_true', help='Use lut4 palettization')
parser_args = parser.parse_args()

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = False
model_args.wkv_customop = False
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 0
model_args.USE_ONNX_L2NORM = False
model_args.USE_ONNX_REDUCE_L2 = False
model_args.USE_CUSTOM_WKV = parser_args.customop

model_args.MODEL_NAME = str(parser_args.model).replace('.pth', '')
model = RWKV_RNN_Stateful(model_args) if parser_args.stateful else RWKV_RNN(model_args)
args = model.args

merge_states = False

PREFILL_SEQ_LENGTH = 16

# Imports for custom ops (not all may be required)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil import (
    Builder as mb,
    Operation,
    types
)
from coremltools.converters.mil.mil.input_type import (
    InputSpec,
    TensorInputType,
)

from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs

# @register_op(is_custom_op=True)
# class custom_wkv7(Operation):
#     input_spec = InputSpec(
#         r = TensorInputType(type_domain="T"),
#         w = TensorInputType(type_domain="T"),
#         k = TensorInputType(type_domain="T"),
#         v = TensorInputType(type_domain="T"),
#         a = TensorInputType(type_domain="T"),
#         b = TensorInputType(type_domain="T"),
#         state = TensorInputType(type_domain="T"),
#     )

#     type_domains = {
#         "T": (types.fp16, types.fp32),
#     }

#     bindings = { 'class_name'  : 'CustomWKV7',
#                  'input_order' : ['r', 'w', 'k', 'v', 'a', 'b', 'state'],
#                  'parameters'  : [],
#                  'description' : "WKV7 Custom layer"
#                 }

#     def __init__(self, **kwargs):
#         super(custom_wkv7, self).__init__(**kwargs)

#     def type_inference(self):
#         state_shape = self.state.shape
#         r_shape = self.r.shape
#         r_shape_list = list(r_shape)
#         seq_length = r_shape_list[0]

#         ret_shape = list(state_shape)
#         if ret_shape[0] == 1:
#             ret_shape = ret_shape[1:]
#         num_heads, head_size, _ = ret_shape
#         return types.tensor(self.state.dtype, (seq_length, num_heads, 1, head_size)), types.tensor(self.state.dtype, (1, num_heads, head_size, head_size))

# @register_torch_op(torch_alias=["rwkv::wkv7"])
# def wkv7(context, node):
#     r, w, k, v, a, b, state = _get_inputs(context, node, expected=7)
#     x_output_name = node.outputs[0]
#     state_output_name = node.outputs[1]
#     x, state_out = mb.custom_wkv7(r=r, w=w, k=k, v=v, a=a, b=b, state=state, name=node.name)
#     x = mb.identity(x=x, name=x_output_name)
#     state_out = mb.identity(x=state_out, name=state_output_name)
#     context.add(x, x_output_name)
#     context.add(state_out, state_output_name)

def _add_wkv7_layer(r, w, k, v, a, b, state, seq_output_name, state_output_name):
    """
    Add a single GRU layer.
    Please note that the Core ML GRU has different definition from Torch,
    so we cannot use mb.gru, and need to implement it with while loop.
    To be more specific, in Core ML:

    o_t = activation(W_{io} x_t + r_t * W_{ho} h_(t−1) + b_{o})

    while torch has
    o_t = activation(W_{io} x_t + b_{io} + r_t * (W_{ho} h_(t−1) + b_{ho}))

    Inputs:
        _input : (seq_len, batch_size, input_dim)
        h0 : (1, batch_size, hidden_dim)
        wi : (3*hidden_dim, input_dim) for the first layer, else (3*hidden_dim, hidden_dim)
        wh : (3*hidden_dim, hidden_dim)
        bi : (3*hidden_dim)
        bh : (3*hidden_dim)

    Return:
        h_list : the list contains all hidden states for each time step
                 with shape (seq_len, batch_size, hidden_dim)
        h : the last hidden state, with shape (1, batch_size, hidden_dim
    """

    r_shape = mb.shape(x=r)
    state_shape = mb.shape(x=state)
    seq_len = mb.slice_by_index(x=r_shape, begin=[0], end=[1])
    num_heads = mb.slice_by_index(x=r_shape, begin=[1], end=[2])
    head_size = mb.slice_by_index(x=r_shape, begin=[2], end=[3])
    w_shape = mb.shape(x=w)

    # (seq_len, num_heads, 1, head_size)
    x_list = mb.fill(shape=w_shape)
    state_out = state

    def cond(i, x_list, state_out):
        return mb.less(x=i, y=seq_len)

    def body(i, x_list, state_out):
        rt = mb.gather(x=r, indices=i, axis=0)
        wt = mb.gather(x=w, indices=i, axis=0)
        kt = mb.gather(x=k, indices=i, axis=0)
        vt = mb.gather(x=v, indices=i, axis=0)
        at = mb.gather(x=a, indices=i, axis=0)
        bt = mb.gather(x=b, indices=i, axis=0)

        wt_shape = mb.shape(x=wt)
        state_shape = mb.shape(x=state_out)

        sa = mb.matmul(x=state_out, y=at)
        sab = mb.matmul(x=sa, y=bt)
        kv = mb.matmul(x=vt, y=kt)
        state_next = mb.mul(x=state_out, y=wt)
        state_next = mb.add(x=state_next, y=kv)
        state_out = mb.add(x=state_next, y=sab)
        xt = mb.matmul(x=state_out, y=rt)
        xt = mb.reshape(x=xt, shape=wt_shape)

        # update counter
        counter = mb.add(x=i, y=1)

        state_out = mb.reshape(x=state_out, shape=state_shape)
        x_list = mb.scatter(data=x_list, indices=counter, updates=xt)

        return (
            counter,
            x_list,
            state_out,
        )

    _, x_list, state_out = mb.while_loop(
        _cond=cond, _body=body, loop_vars=([0], x_list, state_out),
    )

    return x_list, state_out


@register_torch_op(torch_alias=["rwkv::wkv7"])
def wkv7(context, node):
    r, w, k, v, a, b, state = _get_inputs(context, node, expected=7)

    seq_output_name = node.outputs[0]  # output sequence name
    state_output_name = node.outputs[1]  # output state name

    x, state_out = _add_wkv7_layer(r, w, k, v, a, b, state, seq_output_name, state_output_name)

    # rnn output
    context.add(x, seq_output_name)

    # state output
    context.add(state_out, state_output_name)

if parser_args.stateful:
    inputs_decode = [torch.tensor([[0]*1 for _ in range(1)], dtype=torch.int32).to(model.device)]
    inputs_prefill = [torch.tensor([[0]*PREFILL_SEQ_LENGTH for _ in range(1)], dtype=torch.int32).to(model.device)]
else:
    inputs_decode = get_dummy_input_for_rwkv_causal_llm(1, 1, model.device, model.args, merged_states=merge_states)
    inputs_prefill = get_dummy_input_for_rwkv_causal_llm(1, PREFILL_SEQ_LENGTH, model.device, model.args, merged_states=merge_states)

tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-5-world-1b5", trust_remote_code=True)
prompt = "The Eiffel Tower is in the city of"

if parser_args.int4:
    config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "weight_dtype": "int4",
                "granularity": "per_block",
                "block_size": 128,
            },
            "module_type_configs": {
            }
        }
    )
    quantizer = PostTrainingQuantizer(model, config)
    model = quantizer.compress()
elif parser_args.int8:
    config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "weight_dtype": "int8",
                "granularity": "per_channel",
            },
            "module_type_configs": {
            }
        }
    )
    quantizer = PostTrainingQuantizer(model, config)
    model = quantizer.compress()
elif parser_args.lut8:
    palettization_config_dict = {
        "global_config": {"n_bits": 8, "granularity": "per_grouped_channel", "group_size": 128},
    }
    palettization_config = PostTrainingPalettizerConfig.from_dict(palettization_config_dict)
    palettizer = PostTrainingPalettizer(model, palettization_config)
    model = palettizer.compress()
elif parser_args.lut4:
    palettization_config_dict = {
        "global_config": {"n_bits": 4, "granularity": "per_grouped_channel", "group_size": 32},
    }
    palettization_config = PostTrainingPalettizerConfig.from_dict(palettization_config_dict)
    palettizer = PostTrainingPalettizer(model, palettization_config)
    model = palettizer.compress()

def _build_output_name(mode_tag: str) -> str:
    output_name = str(os.path.basename(parser_args.model)).replace('.pth', '')
    output_name += f'_{mode_tag}'
    if parser_args.stateful:
        output_name += '_stateful'
    if merge_states:
        output_name += '_mergestates'
    if parser_args.int4:
        output_name += '_int4'
    elif parser_args.int8:
        output_name += '_int8'
    elif parser_args.lut8:
        output_name += '_lut8'
    elif parser_args.lut4:
        output_name += '_lut4'
    if model_args.USE_CUSTOM_WKV:
        output_name += '_customop'
    return output_name

def _build_coreml_io(inputs):
    # Token ids input is always int32.
    ct_inputs = [ct.TensorType('in0', inputs[0].shape, dtype=np.int32)]
    dtype = np.float16
    if not parser_args.stateful:
        if not merge_states:
            ct_inputs += [
                ct.TensorType(f'state_{i}_in', inputs[i + 1].shape, dtype=dtype)
                for i in range(len(inputs) - 1)
            ]
        else:
            ct_inputs += [ct.TensorType('state_tokenshift_in', inputs[1].shape, dtype=dtype)]
            ct_inputs += [ct.TensorType('state_wkv_in', inputs[2].shape, dtype=dtype)]
    ct_outputs = [ct.TensorType(name='logits', dtype=dtype)]
    if not parser_args.stateful:
        if not merge_states:
            ct_outputs += [ct.TensorType(f'state_{i}_out', dtype=dtype) for i in range(len(inputs) - 1)]
        else:
            ct_outputs += [ct.TensorType('state_tokenshift_out', dtype=dtype)]
            ct_outputs += [ct.TensorType('state_wkv_out', dtype=dtype)]


    return ct_inputs, ct_outputs


def convert_and_save_coreml(jit_model, inputs, mode_tag: str):
    ct_inputs, ct_outputs = _build_coreml_io(inputs)
    output_name = _build_output_name(mode_tag)

    if parser_args.stateful:
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(2, args.n_layer, args.n_embd),
                ),
                name="state_tokenshift",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(args.n_layer, args.n_head, args.head_size, args.head_size),
                ),
                name="state_wkv",
            ),
        ]
        mlmodel = ct.convert(
            jit_model,
            inputs=ct_inputs,
            outputs=ct_outputs,
            states=states,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
    else:
        mlmodel = ct.convert(
            jit_model,
            inputs=ct_inputs,
            outputs=ct_outputs,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

    mlmodel.save(f'{output_name}.mlpackage')
    return output_name


# Export decode & prefill models separately (different sequence length traces).
jit_decode = torch.jit.trace(model, example_inputs=inputs_decode)
convert_and_save_coreml(jit_decode, inputs_decode, mode_tag='decode')
del jit_decode

jit_prefill = torch.jit.trace(model, example_inputs=inputs_prefill)
convert_and_save_coreml(jit_prefill, inputs_prefill, mode_tag='prefill')
del jit_prefill

desc = ct.utils.MultiFunctionDescriptor()

desc.add_function(
    _build_output_name('decode') + '.mlpackage',
    src_function_name="main",
    target_function_name="decode"
)
desc.add_function(
    _build_output_name('prefill') + '.mlpackage',
    src_function_name="main",
    target_function_name="prefill"
)

desc.default_function_name = "decode"
ct.utils.save_multifunction(desc, _build_output_name('combined') + '.mlpackage')
