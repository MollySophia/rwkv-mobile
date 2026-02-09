# RWKV HTTP API

Simple HTTP API for RWKV inference. Base URL: `http://localhost:8000` (default).

## Endpoints

### GET /health

Health check.

**Response:**
```json
{"status":"ok"}
```

---

### GET /v1/batch/supported_batch_sizes

Returns batch sizes supported by the model for batch completion/chat.

**Response:**
```json
{"supported_batch_sizes":[1,2,4,8],"model":"rwkv"}
```

---

### POST /v1/completions

Text completion (OpenAI-compatible).

**Request:**
```json
{
  "prompt": "The Eiffel Tower is in the city of",
  "max_tokens": 64,
  "stream": false,
  "stop_code": 0,
  "temperature": 1.0,
  "top_k": 1,
  "top_p": 1.0,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "penalty_decay": 0.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| prompt | string | yes | Input prompt |
| max_tokens | int | no | Max tokens to generate (default: 256) |
| stream | bool | no | SSE streaming (default: false) |
| stop_code | int | no | Token id to stop at (default: 0) |
| temperature | float | no | Sampling temperature |
| top_k | int | no | Top-k sampling |
| top_p | float | no | Top-p (nucleus) sampling |
| presence_penalty | float | no | Presence penalty |
| frequency_penalty | float | no | Frequency penalty |
| penalty_decay | float | no | Penalty decay |

**Response (non-stream):**
```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1234567890,
  "model": "rwkv",
  "choices": [{"index": 0, "text": "...", "finish_reason": "stop"}],
  "timings": {
    "prompt_per_token_ms": 0.49,
    "prompt_per_second": 2057.3,
    "predicted_n": 64,
    "predicted_per_token_ms": 6.2,
    "predicted_per_second": 161.2
  }
}
```

---

### POST /v1/chat/completions

Chat completion (OpenAI-compatible).

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "max_tokens": 64,
  "stream": false,
  "enable_reasoning": false,
  "force_reasoning": false,
  "force_lang": 0,
  "force_language": ""
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| messages | array | yes | Chat messages (`role`: user/assistant/system, `content`: string) |
| max_tokens | int | no | Max tokens (default: 256) |
| stream | bool | no | SSE streaming (default: false) |
| enable_reasoning | bool | no | Enable thinking/reasoning |
| force_reasoning | bool | no | Force reasoning mode |
| force_lang | int | no | Force language (1 = Chinese) |
| force_language | string | no | "zh" or "zh-CN" for Chinese |
| temperature, top_k, top_p, ... | - | no | Same as completions |

**Response (non-stream):**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "rwkv",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "timings": {...}
}
```

---

### POST /v1/batch/completions

Batch completion. Batch size must be in `supported_batch_sizes`.

**Request:**
```json
{
  "prompts": ["1+1=", "2+2=", "3+3=", "4+4="],
  "max_tokens": 32,
  "stop_code": 0,
  "temperature": 1.0,
  "top_k": 1,
  "top_p": 1.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| prompts | string[] | yes | Array of prompts |
| max_tokens | int | no | Max tokens per item |
| stop_code | int | no | Token id to stop at |
| temperature, top_k, top_p, ... | - | no | Sampling params |

**Response:**
```json
{
  "id": "cmpl-batch-...",
  "object": "batch.completion",
  "created": 1234567890,
  "model": "rwkv",
  "choices": [
    {"index": 0, "text": "2", "finish_reason": "stop"},
    {"index": 1, "text": "4", "finish_reason": "stop"},
    ...
  ],
  "timings": [{...}, {...}, ...]
}
```

---

### POST /v1/batch/chat

Batch chat. Batch size must be in `supported_batch_sizes`.

**Request:**
```json
{
  "conversations": [
    {"messages": [{"role": "user", "content": "Hello"}]},
    {"messages": [{"role": "user", "content": "你好"}]}
  ],
  "max_tokens": 64,
  "enable_reasoning": false,
  "force_reasoning": false,
  "force_lang": 0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| conversations | array | yes | Each item has `messages` (same format as chat) |
| max_tokens | int | no | Max tokens per conversation |
| enable_reasoning, force_reasoning, force_lang | - | no | Same as chat |

**Response:**
```json
{
  "id": "chat-batch-...",
  "object": "batch.chat.completion",
  "created": 1234567890,
  "model": "rwkv",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"},
    ...
  ],
  "timings": [{...}, {...}, ...]
}
```

---

## Error Response

All errors return JSON:
```json
{
  "error": {
    "message": "error description",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

---

## cURL Examples

```bash
# Health
curl http://127.0.0.1:8000/health

# Supported batch sizes
curl http://127.0.0.1:8000/v1/batch/supported_batch_sizes

# Completion
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":32}'

# Completion (streaming)
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":32,"stream":true}'

# Chat
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'

# Batch completion
curl -s http://127.0.0.1:8000/v1/batch/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompts":["1+1=","2+2="],"max_tokens":16}'

# Batch chat
curl -s http://127.0.0.1:8000/v1/batch/chat \
  -H 'Content-Type: application/json' \
  -d '{"conversations":[{"messages":[{"role":"user","content":"Hi"}]},{"messages":[{"role":"user","content":"Hello"}]}],"max_tokens":32}'
```
