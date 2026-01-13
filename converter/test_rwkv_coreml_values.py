import coremltools as ct
from pathlib import Path
import argparse, types, os
import numpy as np
from transformers import AutoTokenizer
import time

parser = argparse.ArgumentParser(description='Test coreml model')
parser.add_argument('model', type=Path, help='Path to RWKV mlpackage file')
parser_args = parser.parse_args()

model = ct.models.MLModel(str(parser_args.model), compute_units=ct.ComputeUnit.CPU_AND_NE)
model_prefill = ct.models.MLModel(str(parser_args.model), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name="prefill")

tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-5-world-1b5", trust_remote_code=True)

spec = model.get_spec()

# inputs = {'in0': np.array([[0.0]])}

state0 = model.make_state()
state1 = model.make_state()

prompt = "User: Where is the Eiffel Tower?\n\nAssistant: The Eiffel Tower is in the city of"

prompt_chunk_length = 16
prompt_tokens = tokenizer.encode(prompt)[:prompt_chunk_length]

outputs_prefill = model_prefill.predict({"in0": np.array([prompt_tokens]).astype(np.float32)}, state=state0)
outputs_decode = []
for token in prompt_tokens:
    outputs_decode.append(model.predict({"in0": np.array([[token]]).astype(np.float32)}, state=state1))

print(outputs_prefill['logits'][0][0])
print(outputs_decode[0]['logits'][0][0])

print(outputs_prefill['logits'][0][-1])
print(outputs_decode[-1]['logits'][0][0])