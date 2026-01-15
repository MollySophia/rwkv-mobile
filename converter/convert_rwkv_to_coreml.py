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
parser.add_argument('--lut6', action='store_true', help='Use lut6 palettization')
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

PREFILL_SEQ_LENGTH = 32

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
elif parser_args.lut6:
    palettization_config_dict = {
        "global_config": {"n_bits": 6, "granularity": "per_grouped_channel", "group_size": 16},
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
    elif parser_args.lut6:
        output_name += '_lut6'
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

# os.remove(_build_output_name('decode') + '.mlpackage')
# os.remove(_build_output_name('prefill') + '.mlpackage')
