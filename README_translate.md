### Build for Android:
- Install Android NDK r25c (recommended version) (Download from https://dl.google.com/android/repository/android-ndk-r25c-linux.zip)
- Install Ninja
```bash
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

### Download model file
- 0.4B translate model for 8Elite：https://huggingface.co/mollysama/rwkv-mobile-models/blob/main/qnn/2.42-260118/RWKV_v7_G1a_0.4B_Translate_ctx4096_20250915_latest-a16w8-8elite-batch.rmpack
- 1.5B translate model for 8Elite：https://huggingface.co/mollysama/rwkv-mobile-models/blob/main/qnn/2.42-260118/RWKV_v7_G1c_1.5B_Translate_ctx4096_20260118-a16w8-8elite-batch.rmpack


### Push executables/libraries to Android device:
- unzip `qnn-libs-2.42.0.zip` and run: `adb push qnn-libs-2.42.0/*.so /data/local/tmp/`
- run: `adb push build/examples/rwkv_server /data/local/tmp/`
- run: `adb push <model_file> /data/local/tmp/`
- run: `adb shell`
- under adb shell:
```bash
cd /data/local/tmp
export LD_LIBRARY_PATH=$PWD
export ADSP_LIBRARY_PATH=$PWD
./rwkv_server --model <model_file> --tokenizer b_rwkv_vocab_v20230424.txt --backend qnn
```
This will start a basic OpenAI API server listening on port 8000.

### Get supported batch sizes
```bash
adb shell -c "curl -s http://127.0.0.1:8000/v1/batch/supported_batch_sizes"
```
Output:
```json
{"model":"rwkv","supported_batch_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}
```

### Translate text
1. Set user_role and assistant_role according to the language of the text to be translated:
- Chinese to English: user_role = "Chinese", assistant_role = "English"
```bash
adb shell 'curl -X POST -s http://127.0.0.1:8000/v1/chat/roles -H "Content-Type: application/json" -d '\''{"user_role": "Chinese", "assistant_role": "English"}'\'''
```
Output:
```json
{"assistant_role":"English","model":"rwkv","user_role":"Chinese"}
```
- English to Chinese: user_role = "English", assistant_role = "Chinese"
```bash
adb shell 'curl -X POST -s http://127.0.0.1:8000/v1/chat/roles -H "Content-Type: application/json" -d '\''{"user_role": "English", "assistant_role": "Chinese"}'\'''
```
Output:
```json
{"assistant_role":"Chinese","model":"rwkv","user_role":"English"}
```

2. Translate text (single text)：
- English to Chinese:
```bash
adb shell 'curl -X POST -s http://127.0.0.1:8000/v1/chat/roles -H "Content-Type: application/json" -d '\''{"user_role": "English", "assistant_role": "Chinese"}'\'''
adb shell 'curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '\''{"messages":[{"role":"user","content":"Please help me write a regular expression to match email addresses, including username, @ symbol, domain, and TLD, supporting common formats."}],"max_tokens":1024}'\'''
```
Output:
```json
{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"请帮助我编写一个正则表达式来匹配电子邮件地址，包括用户名、@符号、域名和顶级域名，支持常见格式。","role":"assistant"}}],"created":1770622698,"id":"chatcmpl-1770622698-2","model":"rwkv","object":"chat.completion","timings":{"predicted_n":49,"predicted_per_second":109.67847800457491,"predicted_per_token_ms":9.117559052545277,"prompt_per_second":817.7868642984922,"prompt_per_token_ms":1.2228125}}
```

- Chinese to English:
```bash
adb shell 'curl -X POST -s http://127.0.0.1:8000/v1/chat/roles -H "Content-Type: application/json" -d '\''{"user_role": "Chinese", "assistant_role": "English"}'\'''
adb shell 'curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '\''{"messages":[{"role":"user","content":"你好，世界！"}],"max_tokens":1024}'\'''
```
Output:
```json
{"assistant_role":"English","model":"rwkv","user_role":"Chinese"}{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"Hello, world!","role":"assistant"}}],"created":1770622813,"id":"chatcmpl-1770622813-3","model":"rwkv","object":"chat.completion","timings":{"predicted_n":4,"predicted_per_second":100.52143492705501,"predicted_per_token_ms":9.94812699127968,"prompt_per_second":103.71292263015972,"prompt_per_token_ms":9.642}}
```

3. Translate text (batch)：
*Note: the whole batch should have the same translation direction.*
*Note: the supported batch sizes can be checked by the command above.*
- English to Chinese:
```bash
adb shell 'curl -X POST -s http://127.0.0.1:8000/v1/chat/roles -H "Content-Type: application/json" -d '\''{"user_role": "English", "assistant_role": "Chinese"}'\'''
adb shell 'curl -s http://127.0.0.1:8000/v1/batch/chat -H "Content-Type: application/json" -d '\''{"conversations":[{"messages":[{"role":"user","content":"Please help me write a regular expression to match email addresses, including username, @ symbol, domain, and TLD, supporting common formats."}]},{"messages":[{"role":"user","content":"I want to write an urban fantasy novel set in modern Shanghai, where ordinary people can trade something for special powers. Help me design a main plot, three characters, and the initial conflict."}]}],"max_tokens":1024}'\'''
```

Output:
```json
{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"请帮我编写一个正则表达式来匹配电子邮件地址，包括用户名、@符号、域名和顶级域名，支持常见格式。","role":"assistant"}},{"finish_reason":"stop","index":1,"message":{"content":"我想写一部以现代上海为背景的都市奇幻小说，其中普通人可以用某种东西换取特殊能力。请帮助我设计主要情节、三个角色和初始冲突。","role":"assistant"}}],"created":1770623090,"id":"chat-batch-1770623090-6","model":"rwkv","object":"batch.chat.completion","timings":[{"predicted_n":48,"predicted_per_second":167.5156524823367,"predicted_per_token_ms":5.9695914094084,"prompt_per_second":770.341839191141,"prompt_per_token_ms":1.298125},{"predicted_n":62,"predicted_per_second":167.5156524823367,"predicted_per_token_ms":5.9695914094084,"prompt_per_second":770.341839191141,"prompt_per_token_ms":1.298125}]}
```

- Chinese to English:
```bash
adb shell 'curl -X POST -s http://127.0.0.1:8000/v1/chat/roles -H "Content-Type: application/json" -d '\''{"user_role": "Chinese", "assistant_role": "English"}'\'''
adb shell 'curl -s http://127.0.0.1:8000/v1/batch/chat -H "Content-Type: application/json" -d '\''{"conversations":[{"messages":[{"role":"user","content":"请帮我编写一个正则表达式来匹配电子邮件地址，包括用户名、@符号、域名和顶级域名，支持常见格式。"}]},{"messages":[{"role":"user","content":"我想写一部以现代上海为背景的都市奇幻小说，其中普通人可以用某种东西换取特殊能力。请帮助我设计主要情节、三个角色和初始冲突。"}]}],"max_tokens":1024}'\'''
```

Output:
```json
{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"Please help me write a regular expression to match email addresses, including usernames, @ symbols, domain names, and top-level domains, supporting common formats.","role":"assistant"}},{"finish_reason":"stop","index":1,"message":{"content":"I want to write a city fantasy novel set in modern Shanghai, where ordinary people can exchange something for special abilities. Please help me design the main plot, three characters, and initial conflict.","role":"assistant"}}],"created":1770623163,"id":"chat-batch-1770623163-7","model":"rwkv","object":"batch.chat.completion","timings":[{"predicted_n":32,"predicted_per_second":172.31790109204397,"predicted_per_token_ms":5.803227602371084,"prompt_per_second":551.9570962206893,"prompt_per_token_ms":1.8117350186221166},{"predicted_n":38,"predicted_per_second":172.31790109204397,"predicted_per_token_ms":5.803227602371084,"prompt_per_second":551.9570962206893,"prompt_per_token_ms":1.8117350186221166}]}
```