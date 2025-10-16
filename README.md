# MLX OpenAI Server

OpenAI-compatible API server using Apple's MLX framework for fast local inference on Apple Silicon.

## Features

- 🚀 OpenAI-compatible `/v1/chat/completions` endpoint
- 📡 Streaming and non-streaming responses
- 🧠 Uses Qwen2.5-Coder-7B-Instruct-4bit model
- ⚡ Optimized for Apple Silicon with MLX
- 🔄 Lazy model loading

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ralfbecher/mlx-openai-server.git
cd mlx-openai-server
```

2. Create a virtual environment:

```bash
python -m venv .
source bin/activate  # On macOS/Linux
```

3. Install dependencies:

```bash
pip install -r requirements-mlx.txt
```

## Usage

Start the server:

```bash
python -m uvicorn mlx_server:app --host 0.0.0.0 --port 2244
```

The server will be available at `http://localhost:2244`

### API Endpoints

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion (streaming and non-streaming)

### Example Request

**Non-streaming:**

```bash
curl -X POST http://localhost:2244/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen30b-mlx",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

**Streaming:**

```bash
curl -X POST http://localhost:2244/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen30b-mlx",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## VS Code Continue Plugin Configuration

To use this server with the [Continue](https://continue.dev) VS Code extension:

1. Install the Continue extension in VS Code
2. Open Continue settings (Command Palette: "Continue: Open config.yaml")
3. Add the following configuration to your `config.yaml`:

```yaml
models:
  - title: MLX Qwen 7B
    provider: openai
    model: qwen30b-mlx
    apiBase: http://localhost:2244/v1
    apiKey: not-needed
```

4. Save the configuration and restart VS Code
5. Make sure the MLX server is running on port 2244
6. Select "MLX Qwen 7B" from the Continue model dropdown

### Configuration Options

You can customize the model behavior in Continue by adding these parameters:

```yaml
models:
  - title: MLX Qwen 7B
    provider: openai
    model: qwen30b-mlx
    apiBase: http://localhost:2244/v1
    apiKey: not-needed
    completionOptions:
      temperature: 0.2
      maxTokens: 4096
    useStreaming: true
```

## Model Configuration

The server uses `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` by default. To change the model, edit the `get_model()` function in `mlx_server.py`:

```python
def get_model():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load("mlx-community/YOUR-MODEL-HERE")
    return _model, _tokenizer
```

Browse available MLX models at: https://huggingface.co/mlx-community

## Requirements

- Apple Silicon Mac (M1/M2/M3)
- Python 3.8+
- MLX framework

## Troubleshooting

### HF_HUB_ENABLE_HF_TRANSFER Error

If you see an error like `ValueError: Fast download using 'hf_transfer' is enabled but 'hf_transfer' package is not available`, you have two options:

1. **Disable fast transfer** (recommended):

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 python -m uvicorn mlx_server:app --host 0.0.0.0 --port 2244
```

2. **Install hf_transfer**:

```bash
pip install hf-transfer
```

To permanently disable this, unset the environment variable:

```bash
unset HF_HUB_ENABLE_HF_TRANSFER
```

## License

MIT
