# rwkv-mobile

An inference runtime with multiple backends supported.

## Goal:

- Easy integration on different platforms using flutter or native cpp, including mobile devices.
- Support inference using different hardware like Qualcomm Hexagon NPU, or general CPU/GPU.
- Provide easy-to-use C apis
- Provide an api server compatible with AI00_server(openai api)

## Supported or planned backends:

- [x] WebRWKV (WebGPU): Compatible with most PC graphics cards, as well as macOS Metal. Doesn't work on Qualcomm's proprietary Adreno GPU driver though.
- [x] llama.cpp: Run on Android devices with CPU inference.
- [x] ncnn: Initial support for rwkv v6/v7 unquantized models (suitable for running tiny models everywhere).
- [x] Qualcomm Hexagon NPU: Based on Qualcomm's QNN SDK 2.42.0.
- [x] MLX: Running RWKV on Apple Silicon devices using Apple's MLX framework.
- [x] MediaTek Neuropilot7: Running RWKV on MediaTek NPU. Currently only supports Dimensity 9300 devices.
- [x] CoreML: Running RWKV with Apple Neural Engine. Based on Apple's CoreML framework.
- [ ] To be continued...

## How to build:

- Install [rust](https://www.rust-lang.org/tools/install) and [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) (for building the web-rwkv backend)
- Check https://github.com/MollySophia/rwkv-mobile/blob/master/.github/workflows/build.yml

### Build for Android:
- Install Android NDK r25c (recommended version) (Download from https://dl.google.com/android/repository/android-ndk-r25c-linux.zip)
- Install Ninja
```
git clone https://github.com/MollySophia/rwkv-mobile.git
cd rwkv-mobile
mkdir build && cd build
cmake .. -DENABLE_NCNN_BACKEND=ON -DENABLE_WEBRWKV_BACKEND=ON -DENABLE_QNN_BACKEND=ON -DENABLE_MTK_NP7_BACKEND=ON \
    -DENABLE_TTS=ON -DENABLE_VISION=ON -DENABLE_WHISPER=ON -DBUILD_EXAMPLES=ON -DENABLE_SERVER=ON \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_NDK=$HOME/android-ndk-r25c \
    -DCMAKE_TOOLCHAIN_FILE=$HOME/android-ndk-r25c/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -G Ninja
ninja
```

## TODO:
- [ ] Better tensor abstraction for different backends
- [ ] Batch inference for all backends
