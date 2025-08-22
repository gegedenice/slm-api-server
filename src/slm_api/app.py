from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Generator, List, Optional

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()  # charge .env si présent à la racine du projet

try:
    from llama_cpp import Llama
except Exception as e:
    raise RuntimeError(
        "llama-cpp-python is required. Install with `pip install --upgrade llama-cpp-python`."
    ) from e

try:
    from huggingface_hub import hf_hub_download
    HAS_HF = True
except Exception:
    HAS_HF = False


def getenv_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    return v if (v is None or v.strip() != "") else default


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


@dataclass
class ModelConfig:
    model_path: Optional[str] = getenv_str("MODEL_PATH")
    n_ctx: int = getenv_int("N_CTX", 4096)
    n_threads: int = getenv_int("N_THREADS", os.cpu_count() or 4)
    n_batch: int = getenv_int("N_BATCH", 256)
    n_gpu_layers: int = getenv_int("N_GPU_LAYERS", 0)
    verbose: bool = os.getenv("LLM_VERBOSE", "0") == "1"

    def to_display(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("model_path"):
            d["model_path"] = os.path.basename(str(d["model_path"]))
        return d


class ModelManager:
    def __init__(self, config: Optional[ModelConfig] = None):
        self._lock = threading.RLock()
        self._llm: Optional[Llama] = None
        self._config = config or ModelConfig()

    @property
    def config(self) -> ModelConfig:
        return self._config

    def get_llm(self) -> Llama:
        with self._lock:
            if self._llm is not None:
                return self._llm
            self._llm = self._load_model(self._config)
            return self._llm

    def _load_model(self, cfg: ModelConfig) -> Llama:
        if not cfg.model_path or not os.path.exists(cfg.model_path):
            raise RuntimeError(
                f"MODEL_PATH does not exist: {cfg.model_path!r}. Use /load or set MODEL_PATH."
            )
        t0 = time.time()
        llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_batch=cfg.n_batch,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=cfg.verbose,
        )
        print(f"[LLM] Loaded in {time.time()-t0:.2f}s | {cfg.to_display()}")
        return llm

    def reload(self, new_cfg: Optional[ModelConfig] = None) -> Dict[str, Any]:
        with self._lock:
            self._llm = None
            if new_cfg is not None:
                self._config = new_cfg
            _ = self.get_llm()
            return {"status": "reloaded", "config": self._config.to_display()}


app = Flask(__name__)
CORS(app)
manager = ModelManager()
MODELS_DIR = getenv_str("MODELS_DIR", "/models") or "/models"
os.makedirs(MODELS_DIR, exist_ok=True)

@app.get("/health")
def health() -> Any:
    try:
        _ = manager.get_llm()
        return jsonify({"status": "ok", "model": manager.config.to_display()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.get("/models")
def models() -> Any:
    cfg = manager.config
    files = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".gguf")]
    return jsonify({
        "object": "list",
        "data": [{
            "id": os.path.basename(cfg.model_path or "unknown.gguf"),
            "object": "model",
            "owned_by": "local",
            "config": cfg.to_display(),
            "available": files,
        }]
    })

@app.post("/reload")
def reload_model() -> Any:
    payload = request.get_json(silent=True) or {}
    cfg = ModelConfig(
        model_path=payload.get("model_path", manager.config.model_path),
        n_ctx=int(payload.get("n_ctx", manager.config.n_ctx)),
        n_threads=int(payload.get("n_threads", manager.config.n_threads)),
        n_batch=int(payload.get("n_batch", manager.config.n_batch)),
        n_gpu_layers=int(payload.get("n_gpu_layers", manager.config.n_gpu_layers)),
        verbose=bool(payload.get("verbose", manager.config.verbose)),
    )
    try:
        info = manager.reload(cfg)
        return jsonify(info)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

@app.post("/load")
def load_model_file() -> Any:
    if not HAS_HF:
        return jsonify({"error": {"message": "huggingface_hub is not installed on server."}}), 500
    body = request.get_json(force=True)
    repo_id = (body.get("repo_id") or "").strip()
    filename = (body.get("filename") or "").strip()
    if not repo_id or not filename:
        return jsonify({"error": {"message": "repo_id and filename are required"}}), 400

    try:
        local_cached = hf_hub_download(repo_id=repo_id, filename=filename)
        target_path = os.path.join(MODELS_DIR, os.path.basename(filename))
        if os.path.abspath(local_cached) != os.path.abspath(target_path):
            with open(local_cached, "rb") as src, open(target_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
        return jsonify({
            "status": "downloaded",
            "repo_id": repo_id,
            "filename": filename,
            "path": target_path,
            "hint": f'POST /reload {{"model_path":"{target_path}"}}'
        })
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500


def sse(data: Dict[str, Any]) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

@app.post("/v1/chat/completions")
def chat_completions() -> Any:
    llm = manager.get_llm()
    body = request.get_json(force=True)

    messages: List[Dict[str, str]] = body.get("messages", [])
    if not messages:
        return jsonify({"error": {"message": "messages[] is required"}}), 400

    stream = bool(body.get("stream", False))
    params: Dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_k", "min_p", "max_tokens", "stop"):
        if key in body:
            params[key] = body[key]
    if "response_format" in body:
        params["response_format"] = body["response_format"]

    if not stream:
        t0 = time.time()
        resp = llm.create_chat_completion(messages=messages, **params)
        return jsonify({
            "id": resp.get("id", f"chatcmpl-{int(time.time()*1000)}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": os.path.basename(manager.config.model_path or "local.gguf"),
            "choices": resp.get("choices", []),
            "usage": resp.get("usage", {}),
            "latency_ms": int((time.time()-t0)*1000),
        })

    def generate() -> Generator[str, None, None]:
        t0 = time.time()
        try:
            for chunk in llm.create_chat_completion(messages=messages, stream=True, **params):
                out = {
                    "id": chunk.get("id", f"chatcmpl-{int(time.time()*1000)}"),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": os.path.basename(manager.config.model_path or "local.gguf"),
                    "choices": chunk.get("choices", []),
                }
                yield sse(out)
        except Exception as e:
            yield sse({"error": {"message": str(e)}})
        finally:
            yield sse({"done": True})
            yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.post("/v1/extract")
def extract() -> Any:
    llm = manager.get_llm()
    body = request.get_json(force=True)
    text: str = body.get("text", "").strip()
    if not text:
        return jsonify({"error": {"message": "'text' is required"}}), 400

    schema = body.get("schema")
    system_prompt = body.get(
        "system_prompt",
        "You are a helpful assistant that outputs ONLY valid minified JSON. Do not include prose.",
    )
    params: Dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_k", "min_p", "max_tokens", "stop"):
        if key in body:
            params[key] = body[key]
    if isinstance(schema, dict):
        params["response_format"] = {"type": "json_object", "schema": schema}

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    t0 = time.time()
    resp = llm.create_chat_completion(messages=messages, **params)
    content = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    try:
        parsed = json.loads(content)
    except Exception as pe:
        return jsonify({"error": {"message": "Invalid JSON", "model_output": content, "parse_error": str(pe)}}), 422

    if isinstance(schema, dict) and "properties" in schema:
        missing = [k for k in schema["properties"].keys() if not (isinstance(parsed, dict) and k in parsed)]
        if missing:
            return jsonify({"error": {"message": "JSON missing required keys", "missing": missing, "model_output": parsed}}), 206

    return jsonify({
        "object": "extract.result",
        "created": int(time.time()),
        "model": os.path.basename(manager.config.model_path or "local.gguf"),
        "latency_ms": int((time.time()-t0)*1000),
        "data": parsed,
    })


if __name__ == "__main__":
    host = getenv_str("HOST", "0.0.0.0") or "0.0.0.0"
    port = getenv_int("PORT", 8000)
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    print("Starting Flask on", host, port, "| MODELS_DIR=", MODELS_DIR)
    app.run(host=host, port=port, debug=debug)