from rwkv_src.rwkv_modeling import RWKV_RNN, RWKV_RNN_Stateful, RWKV_LMHead, make_chunks_stateful
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
parser_args = parser.parse_args()

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

def build_inputs_decode(chunk_idx: int = 0):
    if chunk_idx == 0:
        return [torch.tensor([[0]*1 for _ in range(1)], dtype=torch.int32).to(MODEL_DEVICE)]
    else:
        inputs = [torch.zeros(1, 1, args.n_embd).to(MODEL_DEVICE)]
        if parser_args.chunks > 1:
            inputs.append(torch.zeros(1, 1, args.n_embd).to(MODEL_DEVICE))
        return inputs

def build_inputs_prefill(chunk_idx: int = 0):
    if chunk_idx == 0:
        return [torch.tensor([[0]*PREFILL_SEQ_LENGTH for _ in range(1)], dtype=torch.int32).to(MODEL_DEVICE)]
    else:
        inputs = [torch.zeros(1, PREFILL_SEQ_LENGTH, args.n_embd).to(MODEL_DEVICE)]
        if parser_args.chunks > 1:
            inputs.append(torch.zeros(1, PREFILL_SEQ_LENGTH, args.n_embd).to(MODEL_DEVICE))
        return inputs

palettization_config_dict = {
    "global_config": {"n_bits": 6, "granularity": "per_grouped_channel", "group_size": 32},
    "module_name_configs": {}
}
lut4_group8_config = {"n_bits": 4, "granularity": "per_grouped_channel", "group_size": 8}
lut4_group4_config = {"n_bits": 4, "granularity": "per_grouped_channel", "group_size": 4}
palettization_config_dict["module_name_configs"]["blocks.*.att.key"] = lut4_group8_config
palettization_config_dict["module_name_configs"]["blocks.*.att.value"] = lut4_group8_config
# palettization_config_dict["module_name_configs"]["blocks.*.att.receptance"] = lut4_group8_config
# palettization_config_dict["module_name_configs"]["blocks.*.att.gate"] = lut4_group8_config
# palettization_config_dict["module_name_configs"]["blocks.*.att.output"] = lut4_group8_config
palettization_config_dict["module_name_configs"]["blocks.*.ffn.key"] = lut4_group8_config
# palettization_config_dict["module_name_configs"]["blocks.*.ffn.value"] = lut4_group4_config

palettization_config = PostTrainingPalettizerConfig.from_dict(palettization_config_dict)

palettizer = PostTrainingPalettizer(full_model, palettization_config)
full_model = palettizer.compress()

models = make_chunks_stateful(parser_args.chunks, model_args, full_model=full_model)

def _build_output_name(mode_tag: str, chunk_idx: int = 0) -> str:
    output_name = str(os.path.basename(parser_args.model)).replace('.pth', '')
    output_name += f'-{mode_tag}'
    output_name += '-lut4'
    # Add chunk suffix
    chunk_suffix = f'_chunk{chunk_idx + 1}of{parser_args.chunks}'
    output_name += chunk_suffix
    return output_name

def _build_combined_base_name() -> str:
    output_name = str(os.path.basename(parser_args.model)).replace('.pth', '')
    output_name += '-coreml'
    output_name += '-lut4'
    return output_name

def _build_coreml_io(inputs, chunk_idx: int = 0, num_chunks: int = 1):
    dtype = np.float16
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

    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2, layers_for_chunk[chunk_idx], args.n_embd),
            ),
            name="state_tokenshift",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(layers_for_chunk[chunk_idx], args.n_head, args.head_size, args.head_size),
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

# Export combined models for each chunk (each containing decode and prefill functions).
for chunk_idx, model in enumerate(models):
    print(f"Converting chunk {chunk_idx + 1} of {parser_args.chunks}")

    inputs_decode = build_inputs_decode(chunk_idx)
    inputs_prefill = build_inputs_prefill(chunk_idx)

    desc = ct.utils.MultiFunctionDescriptor()

    # Trace and convert decode model
    jit_decode = torch.jit.trace(model, example_inputs=inputs_decode)
    decode_output_name = convert_and_save_coreml(
        jit_decode,
        inputs_decode,
        mode_tag='decode',
        chunk_idx=chunk_idx,
        output_dir=output_dir,
    )
    del jit_decode

    # Trace and convert prefill model
    jit_prefill = torch.jit.trace(model, example_inputs=inputs_prefill)
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
