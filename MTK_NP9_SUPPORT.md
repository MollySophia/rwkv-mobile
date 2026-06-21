# MediaTek NP9 Backend Support

This note summarizes the current `rwkv-mobile` support code for Dimensity 9500 / NeuroPilot 9.

## Current Readiness

- The `mtk_np9` backend is ready for app-side integration testing with existing <=2.9B packs.
- The best tested 2.9B pack is W4 VSQ32 scale8, shared weights, AR16 prefill, AR1 decode, and `n_chunks=1`.
- Smaller models should use W8 packs. `0.1B` has been validated; `0.4B` and `1.5B` should be re-run through LAMBADA/benchmark before pinning final app assets.
- Multi-chunk pack loading is implemented, including RWKV7 `v_first` handling in the underlying NP9 runtime, but the current production conversion recipe for <=2.9B is still single chunk.
- 7.2B is not ready for app deployment.
- The checked-in NP9 mobile prebuilt archive currently uses the single-chunk ABI and does not expose the newer `RWKVRuntimeOptions::vFirstOutputFirstChunkOnly` field. Do not add that field to `src/backends/mtk_np9/include/rwkv_mtk.h` until `src/backends/mtk_np9/prebuilt/arm64-v8a/librwkv_mtk.a` is rebuilt from the matching runtime.

## Backend Name

Use backend name:

```text
mtk_np9
```

The C API does not need a separate NP9-specific entry point. Flutter/native callers should pass `mtk_np9` to the normal model loading API:

```c
rwkvmobile_runtime_load_model(runtime, model_path, "mtk_np9", tokenizer_path);
```

If the app wants to provide the NP9 wrapper library path manually, pass it through the existing `extra` argument:

```c
rwkvmobile_runtime_load_model_with_extra(runtime, model_path, "mtk_np9", tokenizer_path, np9_library_path);
```

## Source Layout

```text
src/backends/mtk_np9/
  mtk_np9_backend.h
  mtk_np9_backend.cpp
  include/rwkv_mtk.h
  include/llm_types.h
  prebuilt/arm64-v8a/librwkv_mtk.a

src/backends/mtk_common/
  mtk_rwkv_api.h
  mtk_rwkv_dlopen.h
  mtk_rwkv_dlopen.cpp
  mtk_shared_anchor.cpp
```

`mtk_np9_backend.cpp` is the `rwkv-mobile` execution provider. It:

- loads `.rmpack` models;
- maps `embedding`, `decode_chunk*`, optional `prefill_chunk*`, optional `shared_weights`, and optional `lmhead`;
- initializes the NP9 `librwkv_mtk` runtime from in-memory buffers;
- exposes AR1 decode through `eval(int id, ...)`;
- exposes AR16/packed prompt prefill through `eval(std::vector<int> ids, ...)`;
- exposes embedding-input inference through `eval_with_embeddings`;
- implements state get/set/reset using the NP9 runtime state APIs.

`mtk_common/mtk_rwkv_dlopen.*` is shared by NP7 and NP9. It is intentionally separate because NP7 and NP9 both export the same C++ symbol names from different `librwkv_mtk.a` archives.
The common `dlopen` layer stores raw symbol addresses only. Each backend source file includes its own backend-local `include/rwkv_mtk.h` and casts the symbols locally, because NP7 and NP9 have different `RWKVRuntimeOptions` layouts.

## Why NP7 And NP9 Are Separate Backends

NP7 and NP9 must be buildable in the same `rwkv-mobile` binary. The two MediaTek runtime archives have overlapping symbol names, so directly linking both archives into `librwkv_mobile.so` is unsafe.

The current solution is:

- link NP7 archive into `librwkv_mtk_np7.so`;
- link NP9 archive into `librwkv_mtk_np9.so`;
- keep `librwkv_mobile.so` free of those duplicate runtime symbols;
- load the selected runtime wrapper with `dlopen` from the backend.

This lets one app ship both `mtk_np7` and `mtk_np9` backends.

## Build Flags

NP9 is Android-only:

```bash
cmake .. \
  -DENABLE_MTK_NP9_BACKEND=ON \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DANDROID_NDK=$ANDROID_NDK \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja
ninja
```

NP7 and NP9 can be enabled together:

```bash
cmake .. \
  -DENABLE_MTK_NP7_BACKEND=ON \
  -DENABLE_MTK_NP9_BACKEND=ON \
  ...
```

Expected Android artifacts:

```text
librwkv_mobile.so
librwkv_mtk_np9.so
librwkv_mtk_np7.so   # only when NP7 is enabled
```

The NP9 prebuilt archive must exist at:

```text
src/backends/mtk_np9/prebuilt/arm64-v8a/librwkv_mtk.a
```

## Runtime Library Resolution

`mtk_np9_backend::init()` calls:

```text
MtkRwkvDlopen::open("mtk_np9", "RWKV_MTK_NP9_LIB", "librwkv_mtk_np9.so", extra)
```

Library lookup order:

1. `extra` argument from `rwkvmobile_runtime_load_model_with_extra`
2. `RWKV_MTK_NP9_LIB`
3. generic `RWKV_MTK_LIB`
4. `librwkv_mtk_np9.so`
5. `$PWD/librwkv_mtk_np9.so`

The NP9 runtime still needs MediaTek adapter/runtime libraries on the dynamic loader path. For adb testing:

```bash
LD_LIBRARY_PATH=. \
RWKV_MTK_NP9_LIB=./librwkv_mtk_np9.so \
RWKV_MTK_NEURON_ADAPTER_LIB=./libneuronusdk_adapter.9.mtk.so \
./lambada b_rwkv_vocab_v20230424.txt model.rmpack mtk_np9 lambada_test_50.txt
```

For app integration, either put the required `.so` files in a directory visible to the loader or pass the full `librwkv_mtk_np9.so` path through `extra`.

## RMPack Requirements

The NP9 backend expects a `.rmpack` with:

```text
config.json
embedding
decode_chunk0
```

Optional files:

```text
shared_weights
prefill_chunk0
lmhead
```

Multiple chunks are supported by name (`decode_chunkN`, `prefill_chunkN`) but the current static NP9 conversion recipe uses `n_chunks=1`.

Important config fields consumed by the backend:

```text
hidden_size
vocab_size
n_layer
head_size
n_chunks
use_shared_weights
```

If every `prefill_chunkN` is present, `runtime.eval(vector<int>)` uses the NP9 prefill graph. If prefill chunks are absent, higher-level runtime code can still evaluate prompts through repeated AR1 decode.

## State Handling

NP9 state is stored as byte buffers, three entries per layer:

```text
[att, wkv, ffn] per layer
```

`get_state()` returns:

```cpp
std::shared_ptr<std::vector<std::vector<uint8_t>>>
```

`set_state()` accepts that shared pointer form and also accepts the vector by value.

`load_raw_states()` is compatible with the existing `rwkv-mobile` rmpack state loader for WKV state files. It loads one FP16 WKV state per layer and zeros the corresponding ATT/FFN states.

## Logits Type

The NP9 runtime returns FP16 logits. The backend returns a `Tensor1D` FP16 view directly to avoid converting the full vocabulary to FP32 on every token. Callers that need FP32 should use `tensor1d_get_f32` or convert only where required.

## Validation Commands

Use `rwkv-mobile` examples, not `rwkv_mtk_inference/eval`.

```bash
adb -s A87V026102003508 shell 'cd /data/local/tmp/<dir> && \
  LD_LIBRARY_PATH=. \
  RWKV_MTK_NP9_LIB=./librwkv_mtk_np9.so \
  RWKV_MTK_NEURON_ADAPTER_LIB=./libneuronusdk_adapter.9.mtk.so \
  RWKV_MTK_BOOST_HINT=100 \
  RWKV_MTK_QOS_CPU_BOOST=100 \
  RWKV_MTK_QOS_DDR_BOOST=100 \
  ./simple_benchmark model.rmpack mtk_np9'
```

```bash
adb -s A87V026102003508 shell 'cd /data/local/tmp/<dir> && \
  LD_LIBRARY_PATH=. \
  RWKV_MTK_NP9_LIB=./librwkv_mtk_np9.so \
  RWKV_MTK_NEURON_ADAPTER_LIB=./libneuronusdk_adapter.9.mtk.so \
  RWKV_MTK_BOOST_HINT=100 \
  RWKV_MTK_QOS_CPU_BOOST=100 \
  RWKV_MTK_QOS_DDR_BOOST=100 \
  ./lambada b_rwkv_vocab_v20230424.txt model.rmpack mtk_np9 lambada_test_50.txt'
```

For prefill/decode parity testing:

```bash
./lambada b_rwkv_vocab_v20230424.txt model.rmpack mtk_np9 lambada_test_50.txt --decode-only
```

## Current Known Good 2.9B Smoke

Pack family:

```text
rwkv7-g1g-2.9b-20260526-ctx8192-MT6993-static-a16w4vec32scale8-...-cal4096binidx-...-prefill16-lastlogits-...-shared-dla-...
```

On A87V026102003508 with `rwkv-mobile` NP9 tools:

```text
simple_benchmark steady state: about 267-273 tok/s prefill, 28.2-28.6 tok/s decode
LAMBADA50 prefill+decode: 34/50, PPL 3.90786
LAMBADA50 decode-only:    35/50, PPL 3.91359
Full LAMBADA:             3619/5153, PPL 3.96716
```

## Integration Checklist

- Build with `-DENABLE_MTK_NP9_BACKEND=ON`.
- Package `librwkv_mobile.so`.
- Package `librwkv_mtk_np9.so`.
- Package MediaTek NP9 runtime/adapter `.so` files needed by `librwkv_mtk_np9.so`.
- Load the model with backend name `mtk_np9`.
- If the NP9 wrapper `.so` is not discoverable by normal loader paths, pass its full path through `rwkvmobile_runtime_load_model_with_extra`.
- Keep NP7 and NP9 wrapper `.so` files separate when shipping both backends.
