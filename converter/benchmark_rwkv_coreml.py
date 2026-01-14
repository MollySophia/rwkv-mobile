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

prefill_seq_length = 32
n_prompt_tokens = 512
n_decode_tokens = 128
trials = 3

durations_prefill = []
durations_decode = []

for _ in range(trials):
    state = model.make_state()
    for i in range(0, n_prompt_tokens, prefill_seq_length):
        tokens = np.random.randint(0, 65535, size=(1, prefill_seq_length)).astype(np.float32)
        start_time = time.time()
        outputs_prefill = model_prefill.predict({"in0": tokens}, state=state)
        durations_prefill.append((time.time() - start_time) * 1000)

for _ in range(trials):
    state = model.make_state()
    for i in range(0, n_decode_tokens):
        token = np.random.randint(0, 65535, size=(1, 1)).astype(np.float32)
        start_time = time.time()
        outputs_decode = model.predict({"in0": token}, state=state)
        durations_decode.append((time.time() - start_time) * 1000)

print(f"Prefill speed: {1000 * prefill_seq_length / np.mean(durations_prefill):.2f} tokens/s")
print(f"Decode speed: {1000 / np.mean(durations_decode):.2f} tokens/s")