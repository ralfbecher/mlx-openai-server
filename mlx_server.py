from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
from mlx_lm import load, generate, stream_generate



app = FastAPI()

# Lazy load model and tokenizer
_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    return _model, _tokenizer

class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: Optional[str] = None
    messages: List[Msg]
    max_tokens: int = 512
    temperature: float = 0.2
    stream: bool = False

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "qwen30b-mlx", "object": "model"}]}

@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    model, tok = get_model()
    prompt = tok.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in req.messages],
        add_generation_prompt=True
    )

    if req.stream:
        def generate_chunks():
            for response in stream_generate(
                model, tok,
                prompt=prompt,
                max_tokens=req.max_tokens,
                temp=req.temperature,
                verbose=False
            ):
                chunk = {
                    "id": "mlx-chat-stream",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": response.text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Send final chunk with finish_reason
            final_chunk = {
                "id": "mlx-chat-stream",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_chunks(), media_type="text/event-stream")
    else:
        out = generate(model, tok, prompt=prompt,
                       max_tokens=req.max_tokens, temp=req.temperature, verbose=False)
        return {
            "id": "mlx-chat-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out},
                "finish_reason": "stop"
            }]
        }