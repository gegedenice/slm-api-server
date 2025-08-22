# slm-llamacpp-api-server

Flask API autour de [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) avec :

* ✅ Chargement **local** de modèles GGUF (`/models`)
* ✅ Téléchargement Hugging Face Hub via `/load`
* ✅ Recharge dynamique du modèle avec `/reload`
* ✅ API **OpenAI-like** (`/v1/chat/completions`, `/v1/extract`)
* ✅ **Streaming SSE** (`stream: true`) compatible OpenAI
* ✅ Support `.env` pour la configuration
* ✅ Déploiement via Docker ou en local

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/you/slm-llamacpp-api-server.git
cd slm-api-server
```

### 2. Préparer l’environnement

```bash
cp .env.example .env
# Édite .env pour définir le modèle par défaut (MODEL_PATH)
```

### 3. Installation locale (venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

---

## ⚙️ Configuration `.env`

Exemple minimal :

```dotenv
# Dossier où stocker les modèles
MODELS_DIR=./models

# Modèle actif (chemin local vers un GGUF)
MODEL_PATH=./models/gemma-3-270m-it-Q4_K_M.gguf

# Paramètres d’inférence
N_THREADS=8
N_CTX=4096
N_BATCH=256
N_GPU_LAYERS=0

# Réseau
HOST=0.0.0.0
PORT=8000
```

> 🔹 **Note HF cache** :
>
> * Par défaut, Hugging Face utilise `~/.cache/huggingface` → rien à faire.
> * Si tu définis `HF_HOME=./hf_cache`, crée manuellement le dossier en local :
>
>   ```bash
>   mkdir hf_cache
>   ```
> * En Docker, le `Dockerfile` crée déjà `/hf_cache` automatiquement ✅

---

## 🖥️ Démarrage

### Local (Flask dev server)

```bash
make dev
```

### Local (Gunicorn, prod-like)

```bash
gunicorn -w 1 -b 0.0.0.0:8000 slm_api.app:app
```

### Docker

```bash
docker compose up --build -d
```

---

## 🔌 Endpoints API

### `GET /health`

Ping + infos modèle.

### `GET /models`

Retourne le modèle actif + la liste des fichiers GGUF présents dans `MODELS_DIR`.

### `POST /load`

Télécharge un modèle Hugging Face Hub dans `MODELS_DIR`.

```json
{
  "repo_id": "unsloth/gemma-3-270m-it-GGUF",
  "filename": "gemma-3-270m-it-Q4_K_M.gguf"
}
```

Réponse :

```json
{
  "status": "downloaded",
  "repo_id": "unsloth/gemma-3-270m-it-GGUF",
  "filename": "gemma-3-270m-it-Q4_K_M.gguf",
  "path": "/models/gemma-3-270m-it-Q4_K_M.gguf",
  "hint": "POST /reload {\"model_path\":\"/models/gemma-3-270m-it-Q4_K_M.gguf\"}"
}
```

### `POST /reload`

Recharge le modèle depuis un chemin local.

```json
{
  "model_path": "./models/gemma-3-270m-it-Q4_K_M.gguf",
  "n_threads": 8,
  "n_ctx": 4096
}
```

### `POST /v1/chat/completions`

Compatible OpenAI.

* `stream: false` → réponse directe
* `stream: true` → **SSE** (Server-Sent Events)

Exemple minimal :

```json
{
  "messages": [
    {"role": "system", "content": "Tu es un assistant concis."},
    {"role": "user", "content": "Explique Gemma 3 270M en 2 phrases."}
  ],
  "stream": true,
  "temperature": 0.2
}
```

### `POST /v1/extract`

Extraction JSON avec schéma optionnel.

```json
{
  "text": "We introduce NuExtract 2.0 ...",
  "schema": {
    "type": "object",
    "properties": {
      "model_name": {"type": "string"},
      "model_version": {"type": "string"},
      "model_description": {"type": "string"}
    }
  },
  "temperature": 0.0
}
```

Réponse :

```json
{
  "object": "extract.result",
  "created": 1699999999,
  "model": "gemma-3-270m-it-Q4_K_M.gguf",
  "latency_ms": 1234,
  "data": {
    "model_name": "NuExtract",
    "model_version": "2.0",
    "model_description": "Specialized in extracting structured info"
  }
}
```

---

## 🧪 Tests

```bash
pytest -q
```

---

## 📦 Déploiement Docker

Le `Dockerfile` contient déjà :

```dockerfile
RUN mkdir -p /models /hf_cache
```

Donc dans le conteneur :

* `/models` est toujours présent
* `/hf_cache` est prêt si `HF_HOME=/hf_cache`

### Exemple Compose

```bash
docker compose up --build -d
```

Cela lance le serveur sur `http://localhost:8000`.

---

## 🔐 Sécurité

* Restreins l’accès à `/load` et `/reload` en production (authentification, firewall).
* Surveille l’espace disque (les GGUF peuvent être volumineux).
* Monte un volume externe pour `/models` et `/hf_cache` pour la persistance.

---

## 📜 Licence

MIT
