from rwkv_src.rwkv_modeling import RWKV_RNN, make_chunks, make_chunks_stateful
import coremltools as ct
from coremltools.optimize.torch.quantization import PostTrainingQuantizer, PostTrainingQuantizerConfig
from coremltools.optimize.torch.palettization import PostTrainingPalettizer, PostTrainingPalettizerConfig
from pathlib import Path
import argparse, types, os, shutil
import torch
import numpy as np

parser = argparse.ArgumentParser(description='Export coreml model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('--chunks', type=int, default=1, help='Number of chunks')
parser.add_argument('--int8', action='store_true', help='Use int8 quantization')
parser.add_argument('--int4', action='store_true', help='Use int4 quantization')
parser.add_argument('--lut8', action='store_true', help='Use lut8 palettization')
parser.add_argument('--lut6', action='store_true', help='Use lut6 palettization')
parser.add_argument('--lut4', action='store_true', help='Use lut4 palettization')
parser.add_argument(
    '--omniquant-parameters',
    type=Path,
    help='Import OmniQuant per-channel int4 metadata for supported Linear layers and fall back to int8 elsewhere',
)
parser.add_argument(
    '--att-output-int8',
    action='store_true',
    help='When importing OmniQuant int4 metadata, keep att.output weights in the int8 fallback path',
)
parser.add_argument(
    '--state-mode',
    choices=['coreml', 'tensor', 'wkv-coreml'],
    default='wkv-coreml',
    help='Use Core ML StateType, expose all RWKV state as tensors, or keep only WKV as Core ML state',
)
parser_args = parser.parse_args()

OMNIQUANT_BITWIDTH = 4
OMNIQUANT_QMIN = -(1 << (OMNIQUANT_BITWIDTH - 1))
OMNIQUANT_QMAX = (1 << (OMNIQUANT_BITWIDTH - 1)) - 1
OMNIQUANT_MODULE_MAP = {
    'attn.r_proj.weight_quantizer': 'att.receptance',
    'attn.k_proj.weight_quantizer': 'att.key',
    'attn.v_proj.weight_quantizer': 'att.value',
    'attn.o_proj.weight_quantizer': 'att.output',
    'ffn.key.weight_quantizer': 'ffn.key',
    'ffn.value.weight_quantizer': 'ffn.value',
}
active_omniquant_module_map = dict(OMNIQUANT_MODULE_MAP)
if parser_args.att_output_int8:
    active_omniquant_module_map.pop('attn.o_proj.weight_quantizer')

compression_modes = [
    parser_args.int8,
    parser_args.int4,
    parser_args.lut8,
    parser_args.lut6,
    parser_args.lut4,
]
if sum(bool(mode) for mode in compression_modes) > 1:
    raise ValueError('Choose only one of --int8/--int4/--lut8/--lut6/--lut4.')
if parser_args.omniquant_parameters is not None and any(compression_modes):
    raise ValueError('--omniquant-parameters already implies int8 fallback; do not combine it with other compression flags.')

omniquant_parameters = None
omniquant_target_modules = []
if parser_args.omniquant_parameters is not None:
    omniquant_parameters = torch.load(parser_args.omniquant_parameters, map_location='cpu')
    if not isinstance(omniquant_parameters, dict):
        raise TypeError(f'Expected OmniQuant parameters to be a dict, got {type(omniquant_parameters)}')

    for layer_idx, layer_params in omniquant_parameters.items():
        layer_idx = int(layer_idx)
        for omni_prefix, module_suffix in active_omniquant_module_map.items():
            up_key = f'{omni_prefix}.upbound_factor'
            low_key = f'{omni_prefix}.lowbound_factor'
            if up_key not in layer_params or low_key not in layer_params:
                continue
            omniquant_target_modules.append(f'blocks.{layer_idx}.{module_suffix}')

    omniquant_target_modules = sorted(set(omniquant_target_modules))


def _set_or_replace_buffer(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    value = value.detach().clone()
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value)


def _set_quantization_metadata(
    module: torch.nn.Module,
    param_name: str,
    n_bits: int,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
) -> None:
    _set_or_replace_buffer(module, f'_COREML_/{param_name}/compression_type', torch.tensor([3], dtype=torch.int32))
    _set_or_replace_buffer(module, f'_COREML_/{param_name}/quantization_n_bits', torch.tensor(n_bits, dtype=torch.int32))
    _set_or_replace_buffer(module, f'_COREML_/{param_name}/quantization_scale', scale.float())
    zero_point_name = f'_COREML_/{param_name}/zero_point'
    if zero_point is None:
        if zero_point_name in module._buffers:
            del module._buffers[zero_point_name]
    else:
        _set_or_replace_buffer(module, zero_point_name, zero_point)


def _compute_omniquant_int4_per_channel(
    weight: torch.Tensor,
    upbound_factor: torch.Tensor,
    lowbound_factor: torch.Tensor,
    module_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f'Expected a 2D Linear weight for {module_name}, got shape {tuple(weight.shape)}')

    weight_fp32 = weight.detach().float()
    out_features = weight_fp32.shape[0]
    if upbound_factor.numel() != out_features or lowbound_factor.numel() != out_features:
        raise ValueError(
            f'OmniQuant shape mismatch for {module_name}: '
            f'weight out_features={out_features}, '
            f'upbound_factor={tuple(upbound_factor.shape)}, '
            f'lowbound_factor={tuple(lowbound_factor.shape)}'
        )
    upbound_factor = upbound_factor.float().reshape(out_features, 1)
    lowbound_factor = lowbound_factor.float().reshape(out_features, 1)

    xmax = torch.amax(weight_fp32, dim=1, keepdim=True)
    xmin = torch.amin(weight_fp32, dim=1, keepdim=True)
    xmax = torch.sigmoid(upbound_factor) * xmax
    xmin = torch.sigmoid(lowbound_factor) * xmin

    abs_bound = torch.maximum(xmax.abs(), xmin.abs())
    scale = (2.0 * abs_bound) / float(OMNIQUANT_QMAX - OMNIQUANT_QMIN)
    scale = torch.clamp(scale, min=torch.finfo(weight_fp32.dtype).eps)

    quantized_weight = torch.round(weight_fp32 / scale).clamp(OMNIQUANT_QMIN, OMNIQUANT_QMAX)
    dequantized_weight = quantized_weight * scale
    return dequantized_weight.to(weight.dtype), scale


def _apply_omniquant_import(
    model: torch.nn.Module,
    omni_params: dict,
    module_map: dict[str, str],
) -> dict[str, list[str]]:
    _set_or_replace_buffer(model, '_COREML_/metadata_version', torch.tensor(1, dtype=torch.int32))

    stats = {
        'applied': [],
        'missing_module': [],
        'invalid_module': [],
        'missing_params': [],
    }

    for layer_idx, layer_params in omni_params.items():
        layer_idx = int(layer_idx)
        for omni_prefix, module_suffix in module_map.items():
            up_key = f'{omni_prefix}.upbound_factor'
            low_key = f'{omni_prefix}.lowbound_factor'
            module_name = f'blocks.{layer_idx}.{module_suffix}'

            if up_key not in layer_params or low_key not in layer_params:
                stats['missing_params'].append(module_name)
                continue

            try:
                submodule = model.get_submodule(module_name)
            except AttributeError:
                stats['missing_module'].append(module_name)
                continue

            if not isinstance(submodule, torch.nn.Linear):
                stats['invalid_module'].append(module_name)
                continue

            dequantized_weight, scale = _compute_omniquant_int4_per_channel(
                submodule.weight.data,
                layer_params[up_key],
                layer_params[low_key],
                module_name,
            )
            submodule.weight.data.copy_(dequantized_weight)
            _set_quantization_metadata(submodule, 'weight', OMNIQUANT_BITWIDTH, scale)
            stats['applied'].append(module_name)

    return stats

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = False
model_args.USE_EMBEDDING = True
model_args.SKIP_LMHEAD = False
# model_args.SKIP_LMHEAD = True

model_args.MODEL_NAME = str(parser_args.model).replace('.pth', '')
full_model = RWKV_RNN(model_args)
MODEL_DEVICE = full_model.device
args = full_model.args

layers_for_chunk = []
assert parser_args.chunks > 0, "chunks must be >= 1"
base_layers = args.n_layer // parser_args.chunks
extra_layers = args.n_layer % parser_args.chunks
for i in range(parser_args.chunks):
    if i < extra_layers:
        layers_in_chunk = base_layers + 1
        layer_start = i * layers_in_chunk
    else:
        layers_in_chunk = base_layers
        layer_start = extra_layers * (base_layers + 1) + (i - extra_layers) * base_layers
    layer_end = min(args.n_layer, layer_start + layers_in_chunk)
    layers_for_chunk.append(layer_end - layer_start)

PREFILL_SEQ_LENGTH = 16

def build_state_inputs(chunk_idx: int = 0):
    return [
        torch.zeros(1, 2 * layers_for_chunk[chunk_idx], args.n_embd).to(MODEL_DEVICE),
        torch.zeros(layers_for_chunk[chunk_idx], args.n_head, args.head_size, args.head_size).to(MODEL_DEVICE),
    ]

def build_tokenshift_input(chunk_idx: int = 0):
    return torch.zeros(1, 2 * layers_for_chunk[chunk_idx], args.n_embd).to(MODEL_DEVICE)

def build_inputs_decode(chunk_idx: int = 0):
    if chunk_idx == 0:
        inputs = [torch.tensor([[0]*1 for _ in range(1)], dtype=torch.int32).to(MODEL_DEVICE)]
    else:
        inputs = [torch.zeros(1, 1, args.n_embd).to(MODEL_DEVICE)]
        if parser_args.chunks > 1 and parser_args.state_mode == 'coreml':
            inputs.append(torch.zeros(1, 1, args.n_embd).to(MODEL_DEVICE))
    if parser_args.state_mode == 'tensor':
        inputs.extend(build_state_inputs(chunk_idx))
        if chunk_idx > 0 and parser_args.chunks > 1:
            inputs.append(torch.zeros(1, 1, args.n_embd).to(MODEL_DEVICE))
    elif parser_args.state_mode == 'wkv-coreml':
        inputs.append(build_tokenshift_input(chunk_idx))
        if chunk_idx > 0 and parser_args.chunks > 1:
            inputs.append(torch.zeros(1, 1, args.n_embd).to(MODEL_DEVICE))
    return inputs

def build_inputs_prefill(chunk_idx: int = 0):
    if chunk_idx == 0:
        inputs = [torch.tensor([[0]*PREFILL_SEQ_LENGTH for _ in range(1)], dtype=torch.int32).to(MODEL_DEVICE)]
    else:
        inputs = [torch.zeros(1, PREFILL_SEQ_LENGTH, args.n_embd).to(MODEL_DEVICE)]
        if parser_args.chunks > 1 and parser_args.state_mode == 'coreml':
            inputs.append(torch.zeros(1, PREFILL_SEQ_LENGTH, args.n_embd).to(MODEL_DEVICE))
    if parser_args.state_mode == 'tensor':
        inputs.extend(build_state_inputs(chunk_idx))
        if chunk_idx > 0 and parser_args.chunks > 1:
            inputs.append(torch.zeros(1, PREFILL_SEQ_LENGTH, args.n_embd).to(MODEL_DEVICE))
    elif parser_args.state_mode == 'wkv-coreml':
        inputs.append(build_tokenshift_input(chunk_idx))
        if chunk_idx > 0 and parser_args.chunks > 1:
            inputs.append(torch.zeros(1, PREFILL_SEQ_LENGTH, args.n_embd).to(MODEL_DEVICE))
    return inputs

use_int = False
use_lut = False
if parser_args.omniquant_parameters is not None:
    config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "weight_dtype": "int8",
                "granularity": "per_channel",
            },
            "module_name_configs": {
                module_name: None for module_name in omniquant_target_modules
            },
            "module_type_configs": {
            }
        }
    )
    use_int = True
elif parser_args.int4:
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
    use_int = True
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
    use_int = True
elif parser_args.lut8:
    palettization_config_dict = {
        "global_config": {"n_bits": 8, "granularity": "per_grouped_channel", "group_size": 128},
    }
    palettization_config = PostTrainingPalettizerConfig.from_dict(palettization_config_dict)
    use_lut = True
elif parser_args.lut6:
    palettization_config_dict = {
        "global_config": {"n_bits": 6, "granularity": "per_grouped_channel", "group_size": 16},
    }
    palettization_config = PostTrainingPalettizerConfig.from_dict(palettization_config_dict)
    use_lut = True
elif parser_args.lut4:
    palettization_config_dict = {
        "global_config": {"n_bits": 4, "granularity": "per_grouped_channel", "group_size": 32},
    }
    palettization_config = PostTrainingPalettizerConfig.from_dict(palettization_config_dict)
    use_lut = True

if use_lut:
    palettizer = PostTrainingPalettizer(full_model, palettization_config)
    full_model = palettizer.compress()
elif use_int:
    quantizer = PostTrainingQuantizer(full_model, config)
    full_model = quantizer.compress()

if omniquant_parameters is not None:
    if parser_args.att_output_int8:
        print('Keeping blocks.*.att.output weights in int8 fallback.')
    omniquant_stats = _apply_omniquant_import(full_model, omniquant_parameters, active_omniquant_module_map)
    print(
        'Applied OmniQuant int4 metadata to',
        len(omniquant_stats['applied']),
        'Linear layers;',
        len(omniquant_stats['missing_params']),
        'layers fell back to int8 because OmniQuant params were missing.'
    )
    if omniquant_stats['missing_module']:
        print('Warning: missing modules for OmniQuant import:', len(omniquant_stats['missing_module']))
    if omniquant_stats['invalid_module']:
        print('Warning: non-Linear OmniQuant targets skipped:', len(omniquant_stats['invalid_module']))

class PackedStateMixin:
    def _unpack_state(self, state_tokenshift, state_wkv):
        states = []
        for i in range(self.layers_this_chunk):
            att_idx = 2 * i
            ffn_idx = att_idx + 1
            states.append(state_tokenshift[:, att_idx:att_idx+1, :])
            states.append(state_wkv[i:i+1, :, :, :])
            states.append(state_tokenshift[:, ffn_idx:ffn_idx+1, :])
        return states

    def _pack_tokenshift(self, states):
        updates = []
        for i in range(self.layers_this_chunk):
            updates.append(states[3*i].reshape(1, 1, self.model.args.n_embd))
            updates.append(states[3*i+2].reshape(1, 1, self.model.args.n_embd))
        return torch.cat(updates, dim=1)

    def _pack_wkv(self, states):
        updates = []
        for i in range(self.layers_this_chunk):
            updates.append(states[3*i+1])
        return torch.cat(updates, dim=0)


class TensorStateRWKV(PackedStateMixin, torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers_this_chunk = model.layer_end - model.layer_begin

    def forward(self, in0, state_tokenshift, state_wkv, v_first=None):
        state = self._unpack_state(state_tokenshift, state_wkv)
        outputs = self.model(in0, state, v_first)
        if len(outputs) == 3:
            x, state, v_first = outputs
            return x, self._pack_tokenshift(state), self._pack_wkv(state), v_first
        x, state = outputs
        return x, self._pack_tokenshift(state), self._pack_wkv(state)


class WKVStateRWKV(PackedStateMixin, torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.args = model.args
        self.layer_begin = model.layer_begin
        self.layer_end = model.layer_end
        self.layers_this_chunk = self.layer_end - self.layer_begin
        self.chunk_idx = model.chunk_idx
        self.chunks = model.chunks
        self.register_buffer(
            'state_wkv',
            torch.zeros(
                self.layers_this_chunk,
                self.args.n_head,
                self.args.head_size,
                self.args.head_size,
            ),
        )

    def forward(self, in0, state_tokenshift, v_first=None):
        state = self._unpack_state(state_tokenshift, self.state_wkv)
        outputs = self.model(in0, state, v_first)
        if len(outputs) == 3:
            x, state, v_first = outputs
            state_tokenshift_out = self._pack_tokenshift(state)
            state_wkv_out = self._pack_wkv(state)
            self.state_wkv[:] = state_wkv_out + torch.finfo(torch.float32).smallest_normal
            return x, state_tokenshift_out, v_first
        x, state = outputs
        state_tokenshift_out = self._pack_tokenshift(state)
        state_wkv_out = self._pack_wkv(state)
        self.state_wkv[:] = state_wkv_out + torch.finfo(torch.float32).smallest_normal
        return x, state_tokenshift_out


if parser_args.state_mode == 'coreml':
    models = make_chunks_stateful(parser_args.chunks, model_args, full_model=full_model)
else:
    if parser_args.chunks == 1:
        models = [full_model]
    else:
        models = make_chunks(parser_args.chunks, model_args, full_model=full_model)
    if parser_args.state_mode == 'tensor':
        models = [TensorStateRWKV(model) for model in models]
    else:
        models = [WKVStateRWKV(model) for model in models]

def _build_output_name(mode_tag: str, chunk_idx: int = 0) -> str:
    output_name = str(os.path.basename(parser_args.model)).replace('.pth', '')
    output_name += f'-{mode_tag}'
    if parser_args.omniquant_parameters is not None:
        output_name += '-omni-int4int8mix'
        if parser_args.att_output_int8:
            output_name += '-attout-int8'
    elif parser_args.int4:
        output_name += '-int4'
    elif parser_args.int8:
        output_name += '-int8'
    elif parser_args.lut8:
        output_name += '-lut8'
    elif parser_args.lut6:
        output_name += '-lut6'
    elif parser_args.lut4:
        output_name += '-lut4'
    if parser_args.state_mode == 'tensor':
        output_name += '-tensorstate'
    elif parser_args.state_mode == 'wkv-coreml':
        output_name += '-wkvstate'
    # Add chunk suffix
    chunk_suffix = f'_chunk{chunk_idx + 1}of{parser_args.chunks}'
    output_name += chunk_suffix
    return output_name

def _build_combined_base_name() -> str:
    output_name = str(os.path.basename(parser_args.model)).replace('.pth', '')
    output_name += '-coreml'
    if parser_args.omniquant_parameters is not None:
        output_name += '-omni-int4int8mix'
        if parser_args.att_output_int8:
            output_name += '-attout-int8'
    elif parser_args.int4:
        output_name += '-int4'
    elif parser_args.int8:
        output_name += '-int8'
    elif parser_args.lut8:
        output_name += '-lut8'
    elif parser_args.lut6:
        output_name += '-lut6'
    elif parser_args.lut4:
        output_name += '-lut4'
    if parser_args.state_mode == 'tensor':
        output_name += '-tensorstate'
    elif parser_args.state_mode == 'wkv-coreml':
        output_name += '-wkvstate'
    return output_name

def _build_coreml_io(inputs, chunk_idx: int = 0, num_chunks: int = 1):
    dtype = np.float16
    if parser_args.state_mode == 'tensor':
        ct_inputs = [ct.TensorType('in0', inputs[0].shape, dtype=np.int32 if chunk_idx == 0 else dtype)]
        ct_inputs.append(ct.TensorType('state_tokenshift_in', inputs[1].shape, dtype=dtype))
        ct_inputs.append(ct.TensorType('state_wkv_in', inputs[2].shape, dtype=dtype))
        if chunk_idx > 0 and num_chunks > 1:
            ct_inputs.append(ct.TensorType('v_first_in', inputs[3].shape, dtype=dtype))

        ct_outputs = [
            ct.TensorType(name='out0', dtype=dtype),
            ct.TensorType(name='state_tokenshift_out', dtype=dtype),
            ct.TensorType(name='state_wkv_out', dtype=dtype),
        ]
        if chunk_idx == 0 and num_chunks > 1:
            ct_outputs.append(ct.TensorType(name='v_first_out', dtype=dtype))
        return ct_inputs, ct_outputs

    if parser_args.state_mode == 'wkv-coreml':
        ct_inputs = [ct.TensorType('in0', inputs[0].shape, dtype=np.int32 if chunk_idx == 0 else dtype)]
        ct_inputs.append(ct.TensorType('state_tokenshift_in', inputs[1].shape, dtype=dtype))
        if chunk_idx > 0 and num_chunks > 1:
            ct_inputs.append(ct.TensorType('v_first_in', inputs[2].shape, dtype=dtype))

        ct_outputs = [
            ct.TensorType(name='out0', dtype=dtype),
            ct.TensorType(name='state_tokenshift_out', dtype=dtype),
        ]
        if chunk_idx == 0 and num_chunks > 1:
            ct_outputs.append(ct.TensorType(name='v_first_out', dtype=dtype))
        return ct_inputs, ct_outputs

    # chunk0 uses token ids (int32), others use hidden state (float16)
    if chunk_idx == 0:
        ct_inputs = [ct.TensorType('in0', inputs[0].shape, dtype=np.int32)]
        ct_outputs = [ct.TensorType(name='out0', dtype=dtype)]
        if num_chunks > 1:
            ct_outputs.append(ct.TensorType(name='v_first_out', dtype=dtype))
    else:
        ct_inputs = [ct.TensorType('in0', inputs[0].shape, dtype=dtype), ct.TensorType('v_first_in', inputs[1].shape, dtype=dtype)]
        ct_outputs = [ct.TensorType(name='out0', dtype=dtype)]

    return ct_inputs, ct_outputs


def convert_and_save_coreml(jit_model, inputs, mode_tag: str, chunk_idx: int = 0, output_dir: Path | None = None):
    ct_inputs, ct_outputs = _build_coreml_io(inputs, chunk_idx, parser_args.chunks)
    output_name = _build_output_name(mode_tag, chunk_idx)

    states = []
    if parser_args.state_mode == 'coreml':
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, 2 * layers_for_chunk[chunk_idx], args.n_embd),
                    dtype=np.float16,
                ),
                name="state_tokenshift",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(layers_for_chunk[chunk_idx], args.n_head, args.head_size, args.head_size),
                    dtype=np.float16,
                ),
                name="state_wkv",
            ),
        ]
    elif parser_args.state_mode == 'wkv-coreml':
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(layers_for_chunk[chunk_idx], args.n_head, args.head_size, args.head_size),
                    dtype=np.float16,
                ),
                name="state_wkv",
            )
        ]

    mlmodel = ct.convert(
        jit_model,
        inputs=ct_inputs,
        outputs=ct_outputs,
        states=states,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    output_path = f'{output_name}.mlpackage'
    if output_dir is not None:
        output_path = str(output_dir / output_path)
    mlmodel.save(output_path)
    return output_name

combined_base_name = _build_combined_base_name()
output_dir = Path(combined_base_name)
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
    f.write(f'basename: {combined_base_name}\n')
    f.write(f'num_chunks: {parser_args.chunks}\n')
    f.write(f'state_mode: {parser_args.state_mode}\n')

def reset_state_buffers(model):
    for name, buffer in model.named_buffers():
        if name.startswith('state_'):
            buffer.zero_()

# Export combined models for each chunk (each containing decode and prefill functions).
for chunk_idx, model in enumerate(models):
    print(f"Converting chunk {chunk_idx + 1} of {parser_args.chunks}")
    model.eval()

    inputs_decode = build_inputs_decode(chunk_idx)
    inputs_prefill = build_inputs_prefill(chunk_idx)

    desc = ct.utils.MultiFunctionDescriptor()

    # Trace and convert decode model
    if parser_args.state_mode in ('coreml', 'wkv-coreml'):
        reset_state_buffers(model)
    jit_decode = torch.jit.trace(model, example_inputs=inputs_decode, check_trace=False)
    decode_output_name = convert_and_save_coreml(
        jit_decode,
        inputs_decode,
        mode_tag='decode',
        chunk_idx=chunk_idx,
        output_dir=output_dir,
    )
    del jit_decode

    # Trace and convert prefill model
    if parser_args.state_mode in ('coreml', 'wkv-coreml'):
        reset_state_buffers(model)
    jit_prefill = torch.jit.trace(model, example_inputs=inputs_prefill, check_trace=False)
    prefill_output_name = convert_and_save_coreml(
        jit_prefill,
        inputs_prefill,
        mode_tag='prefill',
        chunk_idx=chunk_idx,
        output_dir=output_dir,
    )
    del jit_prefill

    # Add functions to multi-function descriptor
    desc.add_function(
        str(output_dir / (decode_output_name + '.mlpackage')),
        src_function_name="main",
        target_function_name="decode"
    )
    desc.add_function(
        str(output_dir / (prefill_output_name + '.mlpackage')),
        src_function_name="main",
        target_function_name="prefill"
    )

    desc.default_function_name = "decode"

    # Save combined model for this chunk
    combined_output_name = combined_base_name + f'_chunk{chunk_idx + 1}of{parser_args.chunks}'
    ct.utils.save_multifunction(desc, str(output_dir / (combined_output_name + '.mlpackage')))

    # Clean up individual files
    shutil.rmtree(output_dir / (decode_output_name + '.mlpackage'))
    shutil.rmtree(output_dir / (prefill_output_name + '.mlpackage'))
