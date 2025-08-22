# app.py
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

load_dotenv()  # charge .env si présent


# -------------------------------
# Dépendances externes
# -------------------------------
try:
    from llama_cpp import Llama
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "llama-cpp-python is required. Install with `pip install --upgrade llama-cpp-python`."
    ) from e

try:
    from huggingface_hub import hf_hub_download
    HAS_HF = True
except Exception:
    HAS_HF = False


# -------------------------------
# Helpers ENV
# -------------------------------
def getenv_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    return v if (v is None or v.strip() != "") else default


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def getenv_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# -------------------------------
# Modèle & Configuration
# -------------------------------
@dataclass
class ModelConfig:
    # Chemin local vers un GGUF (sous MODELS_DIR)
    model_path: Optional[str] = getenv_str("MODEL_PATH")

    # Paramètres d'inférence
    n_ctx: int = getenv_int("N_CTX", 4096)
    n_threads: int = getenv_int("N_THREADS", os.cpu_count() or 4)
    n_batch: int = getenv_int("N_BATCH", 256)
    n_gpu_layers: int = getenv_int("N_GPU_LAYERS", 0)  # CPU par défaut
    verbose: bool = os.getenv("LLM_VERBOSE", "0") == "1"

    def to_display(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("model_path"):
            d["model_path"] = os.path.basename(str(d["model_path"]))
        return d


class ModelManager:
    """Gestion thread-safe du modèle (lazy load + reload)."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self._lock = threading.RLock()
        self._llm: Optional[Llama] = None
        self._config = config or ModelConfig()

    @property
    def config(self) -> ModelConfig:
        return self._config

    def is_loaded(self) -> bool:
        with self._lock:
            return self._llm is not None

    def set_config(self, new_cfg: ModelConfig) -> None:
        """Met à jour la config sans charger en mémoire."""
        with self._lock:
            self._config = new_cfg
            # ne touche pas à _llm

    def get_llm(self) -> Llama:
        with self._lock:
            if self._llm is not None:
                return self._llm
            self._llm = self._load_model(self._config)
            return self._llm

    def _load_model(self, cfg: ModelConfig) -> Llama:
        if not cfg.model_path or not os.path.exists(cfg.model_path):
            raise RuntimeError(
                f"MODEL_PATH does not exist: {cfg.model_path!r}. Use /load to fetch a model or set MODEL_PATH."
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
        dt = time.time() - t0
        print(f"[LLM] Loaded model in {dt:.2f}s | cfg={cfg.to_display()}")
        return llm

    def reload(self, new_cfg: Optional[ModelConfig] = None) -> Dict[str, Any]:
        with self._lock:
            self._llm = None
            if new_cfg is not None:
                self._config = new_cfg
            _ = self.get_llm()  # force le chargement pour valider
            return {"status": "reloaded", "config": self._config.to_display()}


# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__)
CORS(app)

MODELS_DIR = getenv_str("MODELS_DIR", "/models") or "/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Options d'auto-load au démarrage
AUTO_REPO_ID = getenv_str("AUTO_REPO_ID")
AUTO_FILENAME = getenv_str("AUTO_FILENAME")  # ex: "gemma-3-270m-it-Q4_K_M.gguf"
AUTO_IF_MISSING = getenv_bool("AUTO_IF_MISSING", True)  # ne télécharge que si le fichier manque
AUTO_PRELOAD = getenv_bool("AUTO_PRELOAD", False)  # charge en mémoire au boot

manager = ModelManager()


def _bootstrap_autoload() -> Dict[str, Any]:
    """
    Si MODEL_PATH n'existe pas et que AUTO_REPO_ID/AUTO_FILENAME sont fournis,
    tente un téléchargement dans MODELS_DIR, puis prépare MODEL_PATH.
    Si AUTO_PRELOAD=True, charge aussi le modèle en mémoire (via reload()).
    """
    info: Dict[str, Any] = {
        "models_dir": MODELS_DIR,
        "auto_repo_id": AUTO_REPO_ID,
        "auto_filename": AUTO_FILENAME,
        "auto_if_missing": AUTO_IF_MISSING,
        "auto_preload": AUTO_PRELOAD,
        "performed": False,
        "path": None,
        "error": None,
    }

    cfg = manager.config
    target_path = cfg.model_path

    # Si MODEL_PATH n'est pas défini mais qu'on a un AUTO_FILENAME, on propose une cible.
    if (not target_path) and AUTO_FILENAME:
        target_path = os.path.join(MODELS_DIR, os.path.basename(AUTO_FILENAME))

    # Si on a déjà un fichier présent, pas besoin de télécharger.
    if target_path and os.path.exists(target_path):
        # Met à jour la config (sans précharger)
        manager.set_config(
            ModelConfig(
                model_path=target_path,
                n_ctx=cfg.n_ctx,
                n_threads=cfg.n_threads,
                n_batch=cfg.n_batch,
                n_gpu_layers=cfg.n_gpu_layers,
                verbose=cfg.verbose,
            )
        )
        if AUTO_PRELOAD:
            try:
                manager.reload()  # charge en mémoire
            except Exception as e:
                info["error"] = f"preload_failed: {e}"
        info["performed"] = False
        info["path"] = target_path
        print(f"[BOOT] Model file already present at {target_path}")
        return info

    # Ici, le fichier n'existe pas : on tente un auto-load si possible.
    if AUTO_IF_MISSING and AUTO_REPO_ID and AUTO_FILENAME and HAS_HF:
        try:
            print(f"[BOOT] Auto-load from HF: {AUTO_REPO_ID}/{AUTO_FILENAME}")
            local_cached = hf_hub_download(repo_id=AUTO_REPO_ID, filename=AUTO_FILENAME)
            target_path = os.path.join(MODELS_DIR, os.path.basename(AUTO_FILENAME))
            if os.path.abspath(local_cached) != os.path.abspath(target_path):
                # Copie chunkée
                with open(local_cached, "rb") as src, open(target_path, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

            new_cfg = ModelConfig(
                model_path=target_path,
                n_ctx=cfg.n_ctx,
                n_threads=cfg.n_threads,
                n_batch=cfg.n_batch,
                n_gpu_layers=cfg.n_gpu_layers,
                verbose=cfg.verbose,
            )

            if AUTO_PRELOAD:
                manager.reload(new_cfg)  # charge en mémoire tout de suite
            else:
                manager.set_config(new_cfg)  # défini le chemin, chargera à la 1re requête

            info["performed"] = True
            info["path"] = target_path
            print(f"[BOOT] Auto-loaded model to {target_path} (preload={AUTO_PRELOAD})")
            return info
        except Exception as e:
            info["error"] = str(e)
            print(f"[BOOT][ERROR] Auto-load failed: {e}")
            return info

    # Pas d'auto-load possible / activé
    return info


# Effectue l'auto-load au démarrage (non bloquant au niveau de la disponibilité HTTP)
BOOT_INFO = _bootstrap_autoload()


# -------------------------------
# Endpoints d'info
# -------------------------------
@app.get("/health")
def health() -> Any:
    """
    Health check "soft":
    - Si MODEL_PATH absent ou fichier manquant -> status="unloaded" (HTTP 200), sans 500.
    - Si fichier présent mais modèle non encore chargé -> status="available".
    - Si modèle en mémoire -> status="ready".
    """
    try:
        cfg = manager.config
        path = cfg.model_path
        if not path or not os.path.exists(path):
            return jsonify(
                {
                    "status": "unloaded",
                    "reason": "MODEL_PATH is not set or file not found",
                    "model": cfg.to_display(),
                    "exists": bool(path and os.path.exists(path)),
                    "boot_info": BOOT_INFO,
                    "hint": 'Use /load then /reload with {"model_path": "/models/your.gguf"} '
                            'or set AUTO_REPO_ID/AUTO_FILENAME in .env for auto-load.',
                }
            )
        if not manager.is_loaded():
            # Fichier OK, modèle pas encore chargé
            return jsonify(
                {
                    "status": "available",
                    "model": cfg.to_display(),
                    "path": path,
                    "boot_info": BOOT_INFO,
                    "hint": 'Call /reload with {"model_path": "%s"}' % path,
                }
            )
        # Modèle déjà en mémoire
        return jsonify({"status": "ready", "model": cfg.to_display(), "path": path})
    except Exception as e:
        # Cas vraiment inattendu
        return jsonify({"status": "error", "error": str(e)}), 500


@app.get("/models")
def models() -> Any:
    """
    Liste les GGUF disponibles et l'état du modèle actif.
    """
    cfg = manager.config

    available = []
    for f in os.listdir(MODELS_DIR):
        if f.lower().endswith(".gguf"):
            full = os.path.join(MODELS_DIR, f)
            try:
                size = os.path.getsize(full)
            except Exception:
                size = None
            available.append({"name": f, "path": full, "size_bytes": size})

    active_path = cfg.model_path
    active_exists = bool(active_path and os.path.exists(active_path))
    state = (
        "ready"
        if manager.is_loaded()
        else ("available" if active_exists else "unloaded")
    )

    return jsonify(
        {
            "object": "list",
            "active": {
                "path": active_path,
                "basename": os.path.basename(active_path) if active_path else None,
                "exists": active_exists,
                "loaded": manager.is_loaded(),
                "state": state,
                "config": cfg.to_display(),
            },
            "available": available,
            "hint": 'Switch with POST /reload {"model_path": "/models/<file>.gguf"}',
            "boot_info": BOOT_INFO,
        }
    )


# -------------------------------
# Gestion des modèles (load/reload)
# -------------------------------
@app.post("/load")
def load_model_file() -> Any:
    """
    Télécharge un GGUF depuis Hugging Face Hub vers MODELS_DIR.
    Body: {"repo_id": str, "filename": str}
    Ne recharge pas automatiquement le modèle — appeler /reload ensuite.
    """
    if not HAS_HF:
        return (
            jsonify(
                {"error": {"message": "huggingface_hub is not installed on server."}}
            ),
            500,
        )

    body = request.get_json(force=True)
    repo_id = (body.get("repo_id") or "").strip()
    filename = (body.get("filename") or "").strip()
    if not repo_id or not filename:
        return jsonify({"error": {"message": "repo_id and filename are required"}}), 400

    try:
        # Respecte HF_TOKEN éventuellement défini dans l'env.
        local_cached = hf_hub_download(repo_id=repo_id, filename=filename)
        target_path = os.path.join(MODELS_DIR, os.path.basename(filename))
        if os.path.abspath(local_cached) != os.path.abspath(target_path):
            # Copie (chunkée) depuis le cache HF vers MODELS_DIR
            with open(local_cached, "rb") as src, open(target_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
        return jsonify(
            {
                "status": "downloaded",
                "repo_id": repo_id,
                "filename": filename,
                "path": target_path,
                "hint": 'POST /reload {"model_path": "%s"}' % target_path,
            }
        )
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500


@app.post("/reload")
def reload_model() -> Any:
    """
    Recharge le modèle avec un nouveau MODEL_PATH et/ou nouveaux hyperparamètres.
    Body: {"model_path": str, "n_ctx": int, "n_threads": int, "n_batch": int, "n_gpu_layers": int, "verbose": bool}
    """
    payload = request.get_json(silent=True) or {}
    model_path = payload.get("model_path", manager.config.model_path)

    cfg = ModelConfig(
        model_path=model_path,
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


# -------------------------------
# Chat Completions (OpenAI-like) + SSE
# -------------------------------
def sse_format(data: Dict[str, Any]) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


@app.post("/v1/chat/completions")
def chat_completions() -> Any:
    """
    Endpoint OpenAI-like (non-stream ou SSE si {"stream": true}).
    Body minimal:
      {
        "messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}],
        "stream": true|false,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "min_p": 0.001,
        "max_tokens": 256,
        "stop": ["..."],
        "response_format": {...}
      }
    """
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
        dt = time.time() - t0
        out = {
            "id": resp.get("id", f"chatcmpl-{int(time.time()*1000)}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": os.path.basename(manager.config.model_path or "local.gguf"),
            "choices": resp.get("choices", []),
            "usage": resp.get("usage", {}),
            "latency_ms": int(dt * 1000),
        }
        return jsonify(out)

    # Streaming SSE
    def generate() -> Generator[str, None, None]:
        t0 = time.time()
        try:
            for chunk in llm.create_chat_completion(
                messages=messages, stream=True, **params
            ):
                # Chunks similaires au format OpenAI
                out = {
                    "id": chunk.get("id", f"chatcmpl-{int(time.time()*1000)}"),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": os.path.basename(manager.config.model_path or "local.gguf"),
                    "choices": chunk.get("choices", []),
                }
                yield sse_format(out)
        except Exception as e:
            yield sse_format({"error": {"message": str(e)}})
        finally:
            yield sse_format({"done": True})
            yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# -------------------------------
# Extraction JSON (non-stream)
# -------------------------------
@app.post("/v1/extract")
def extract() -> Any:
    """
    Extraction JSON avec schéma optionnel.
    Body:
      - text: str (requis)
      - schema: dict (optionnel) -> transmis via response_format si supporté
      - system_prompt: str (optionnel)
      - temperature/top_p/top_k/min_p/max_tokens/stop: optionnels
    """
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    try:
        t0 = time.time()
        resp = llm.create_chat_completion(messages=messages, **params)
        dt = time.time() - t0
        content = ((resp.get("choices") or [{}])[0].get("message") or {}).get(
            "content", ""
        )
        try:
            parsed = json.loads(content)
        except Exception as pe:
            return (
                jsonify(
                    {
                        "error": {
                            "message": "Model did not produce valid JSON.",
                            "model_output": content,
                            "parse_error": str(pe),
                        }
                    }
                ),
                422,
            )

        # Contrôle léger des clés attendues
        if isinstance(schema, dict) and "properties" in schema:
            missing = [
                k
                for k in schema["properties"].keys()
                if not (isinstance(parsed, dict) and k in parsed)
            ]
            if missing:
                return (
                    jsonify(
                        {
                            "error": {
                                "message": "JSON is valid but missing required keys (best-effort check).",
                                "missing": missing,
                                "model_output": parsed,
                            }
                        }
                    ),
                    206,
                )

        return jsonify(
            {
                "object": "extract.result",
                "created": int(time.time()),
                "model": os.path.basename(manager.config.model_path or "local.gguf"),
                "latency_ms": int(dt * 1000),
                "data": parsed,
            }
        )
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500


# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    host = getenv_str("HOST", "0.0.0.0") or "0.0.0.0"
    port = getenv_int("PORT", 8000)
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    print("Starting Flask on", host, port, "| MODELS_DIR=", MODELS_DIR)
    app.run(host=host, port=port, debug=debug)