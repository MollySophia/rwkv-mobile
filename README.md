# rwkv-mobile

An inference runtime with multiple backends supported.

## Goal:
- Easy integration on different platforms using flutter or native cpp, including mobile devices.
- Support inference using different hardware like Qualcomm Hexagon NPU, or general CPU/GPU.
- Provide easy-to-use C apis
- Provide an api server compatible with AI00_server(openai api)

## Supported or planned backends:
- [x] WebRWKV (WebGPU): Compatible with most PC graphics cards, as well as macOS Metal. Doesn't work on Qualcomm's proprietary Adreno GPU driver though.
- [ ] RWKV.cpp (TODO): Based on ggml, focusing on CPU inference.
- [ ] Qualcomm Hexagon NPU (TODO: move code from the experimental repo [rwkv-qualcomm](github.com/MollySophia/rwkv-qualcomm)): Based on Qualcomm's QNN SDK.
- [ ] To be continued...
  
## How to build:
- Install rust and cargo (for building the web-rwkv backend)
- ``git clone https://github.com/MollySophia/rwkv-mobile``
- ``cd rwkv-mobile && mkdir build && cd build``
- ``cmake ..``
- ``cmake --build . -j $(nproc)``