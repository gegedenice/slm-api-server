# slm-api-server

---
**Vibe-codé avec GPT-5**

---

Flask API autour de [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) avec :

* ✅ Chargement **local** de modèles GGUF
* ✅ Téléchargement Hugging Face Hub via `/load`
* ✅ **Auto-load au démarrage** (si le fichier manque)
* ✅ Recharge dynamique du modèle via `/reload`
* ✅ API **OpenAI-like** (`/v1/chat/completions`, `/v1/extract`)
* ✅ **Streaming SSE** (`stream: true`) compatible OpenAI
* ✅ Support `.env` (via `python-dotenv`)
* ✅ Déploiement **Docker** ou local

---

## 🚀 Installation

### Cloner & préparer

```bash
git clone https://github.com/you/slm-api-server.git
cd slm-api-server
cp .env.example .env
# édite .env selon le contexte (voir section Configuration)
```

### Installation locale (venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

---

## ⚙️ Configuration (.env)

### Variables principales

| Variable                                        | Rôle                                                                   | Exemple / Valeur recommandée                                  |
| ----------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------- |
| `MODELS_DIR`                                    | Dossier **dans l’environnement où tourne l’app** pour stocker les GGUF | `/models` (en Docker) ou `./models` (local)                   |
| `MODEL_PATH`                                    | **Chemin du fichier GGUF actif** (le modèle que l’API utilisera)       | `/models/gemma-3-270m-it-Q4_K_M.gguf` (peut être laissé vide) |
| `AUTO_REPO_ID`                                  | Repo HF pour auto-load au boot si le fichier manque                    | `unsloth/gemma-3-270m-it-GGUF`                                |
| `AUTO_FILENAME`                                 | Nom du fichier `.gguf` à récupérer sur HF                              | `gemma-3-270m-it-Q4_K_M.gguf`                                 |
| `AUTO_IF_MISSING`                               | `true/false` — ne télécharge **que si** le fichier local est absent    | `true`                                                        |
| `AUTO_PRELOAD`                                  | `true/false` — charge le modèle **en mémoire au boot**                 | `false` (lazy)                                                |
| `N_THREADS`, `N_CTX`, `N_BATCH`, `N_GPU_LAYERS` | Paramètres d’inférence CPU (llama.cpp)                                 | `8`, `4096`, `256`, `0`                                       |
| `HOST`, `PORT`                                  | Réseau API                                                             | `0.0.0.0`, `8000`                                             |
| `HF_HOME` (optionnel)                           | Dossier cache HF                                                       | `/hf_cache` (Docker)                                          |

> ℹ️ **`MODEL_PATH` = “pointeur” vers le fichier GGUF actif.**
>
> * Si le fichier existe : l’API peut l’utiliser immédiatement (lazy ou pré-chargé selon `AUTO_PRELOAD`).
> * S’il n’existe pas et que `AUTO_REPO_ID`/`AUTO_FILENAME` sont fournis, l’app **télécharge** ce modèle dans `MODELS_DIR` au démarrage (si `AUTO_IF_MISSING=true`) et ajuste `MODEL_PATH`.
> * Sinon, laisser `MODEL_PATH` vide et piloter tout via `/load` puis `/reload`.

### Exemple de `.env` (Docker recommandé)

```dotenv
# Stockage des modèles dans le conteneur
MODELS_DIR=/models
MODEL_PATH=/models/gemma-3-270m-it-Q4_K_M.gguf

# Auto-load au démarrage si le fichier manque
AUTO_REPO_ID=unsloth/gemma-3-270m-it-GGUF
AUTO_FILENAME=gemma-3-270m-it-Q4_K_M.gguf
AUTO_IF_MISSING=true
AUTO_PRELOAD=false

# Inférence CPU
N_THREADS=8
N_CTX=4096
N_BATCH=256
N_GPU_LAYERS=0

# Réseau
HOST=0.0.0.0
PORT=8000

# (Optionnel) Cache HF
# HF_HOME=/hf_cache
```

---

## 🖥️ Démarrage

### Local (Flask dev server)

```bash
make dev
```

### Local (gunicorn, prod-like)

```bash
gunicorn -w 1 -b 0.0.0.0:8000 slm_api.app:app
```

### Docker

* **Sans persistance host** (simple) — *les modèles sont stockés dans le conteneur* :

```yaml
# docker-compose.yml (extrait)
services:
  api:
    build: .
    env_file: .env
    ports:
      - "${PORT:-8000}:8000"
    restart: unless-stopped
```

* **Avec persistance** (recommandé en prod) :

```yaml
services:
  api:
    build: .
    env_file: .env
    ports:
      - "${PORT:-8000}:8000"
    volumes:
      - ./models:/models
      - ./hf_cache:/hf_cache
    restart: unless-stopped
```

> 💡 Si les volumes sont commentés, les `.gguf` seront **dans le conteneur** (perdus à la suppression).
> Avec volumes, ils sont persistants côté host.

---

## 🩺 Santé & cycle de vie du modèle

### `/health` — statuts

* `unloaded` : `MODEL_PATH` non défini ou fichier absent.
  → utilise `/load` puis `/reload`, ou configure `AUTO_REPO_ID`/`AUTO_FILENAME`.
* `available` : le fichier `MODEL_PATH` **existe**, mais le modèle n’a pas encore été chargé en mémoire.
  → un appel `/reload` ou une première requête de complétion le chargera.
* `ready` : le modèle est **chargé en mémoire**, l’API est fully ready.

Exemple :

```bash
curl -s http://localhost:8000/health | jq .
```

### `/models` — vue d’ensemble

Renvoie l’état actif + la liste des `.gguf` trouvés :

```json
{
  "object": "list",
  "active": {
    "path": "/models/gemma-3-270m-it-Q4_K_M.gguf",
    "basename": "gemma-3-270m-it-Q4_K_M.gguf",
    "exists": true,
    "loaded": false,
    "state": "available",
    "config": { "model_path": "gemma-3-270m-it-Q4_K_M.gguf", "n_ctx": 4096, ... }
  },
  "available": [
    { "name": "gemma-3-270m-it-Q4_K_M.gguf", "path": "/models/gemma-3-270m-it-Q4_K_M.gguf", "size_bytes": 1234 }
  ],
  "hint": "Switch with POST /reload {\"model_path\": \"/models/<file>.gguf\"}",
  "boot_info": { "...": "..." }
}
```

---

## 🔌 Endpoints API

### `POST /load`

Télécharge un modèle depuis Hugging Face Hub dans `MODELS_DIR` (ne fait **pas** de reload auto).

```bash
curl -sX POST http://localhost:8000/load \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"unsloth/gemma-3-270m-it-GGUF","filename":"gemma-3-270m-it-Q4_K_M.gguf"}' | jq .
```

Réponse (ex.) :

```json
{
  "status": "downloaded",
  "repo_id": "unsloth/gemma-3-270m-it-GGUF",
  "filename": "gemma-3-270m-it-Q4_K_M.gguf",
  "path": "/models/gemma-3-270m-it-Q4_K_M.gguf",
  "hint": "POST /reload {\"model_path\": \"/models/gemma-3-270m-it-Q4_K_M.gguf\"}"
}
```

### `POST /reload`

Active un modèle local et (re)charge en mémoire.
Params utiles : `model_path`, `n_ctx`, `n_threads`, `n_batch`, `n_gpu_layers`, `verbose`.

```bash
curl -sX POST http://localhost:8000/reload \
  -H "Content-Type: application/json" \
  -d '{"model_path":"/models/gemma-3-270m-it-Q4_K_M.gguf","n_threads":8,"n_ctx":4096}' | jq .
```

### `POST /v1/chat/completions`

OpenAI-like.

* Réponse directe si `stream: false` (par défaut)
* **SSE** si `stream: true` (chunking type OpenAI + terminaison `data: [DONE]`)

```bash
curl -N -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "Tu es un assistant concis."},
      {"role": "user", "content": "Explique Gemma 3 270M en 2 phrases."}
    ],
    "stream": true,
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 64
  }'
```

### `POST /v1/extract`

Extraction JSON avec schéma optionnel (`response_format` transmis si supporté).

```bash
curl -s http://localhost:8000/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
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
  }' | jq .
```

---

## 🧠 Auto-load : ce qui se passe au démarrage

1. L’app lit `.env`.
2. Si `MODEL_PATH` **pointe vers un fichier existant**, elle l’utilisera (lazy ou `AUTO_PRELOAD`).
3. Sinon, si **`AUTO_REPO_ID` + `AUTO_FILENAME`** sont fournis et `AUTO_IF_MISSING=true` :

   * l’app **télécharge** le `.gguf` dans `MODELS_DIR`,
   * met à jour `MODEL_PATH` vers ce fichier,
   * si `AUTO_PRELOAD=true`, **charge** le modèle en mémoire immédiatement.
4. À défaut, `/health` reste en `unloaded` avec des **hints** d’action.

---

## 🧪 Tests

```bash
pytest -q
```

---

## 🔐 Sécurité

* Protégez `/load` et `/reload` en prod (authN, firewall, réseau privé).
* Surveillez l’espace disque (les GGUF sont volumineux).
* En Docker, préférer des **volumes** pour persister `/models` et `/hf_cache`.

---

## ❓Dépannage rapide

* `/health` → `unloaded` : le fichier `MODEL_PATH` est manquant.
  → utilisez `/load` puis `/reload`, ou `AUTO_REPO_ID`/`AUTO_FILENAME`.
* Vous ne voyez pas `/models` en Docker :
  → **dans le conteneur**, le chemin doit être absolu (`/models/...`).
  → si vous avez commenté les volumes, tout reste **dans le conteneur**.
* Valider l’état dans le conteneur :

  ```bash
  docker compose exec api sh -lc 'echo MODELS_DIR=$MODELS_DIR; echo MODEL_PATH=$MODEL_PATH; ls -la /models'
  ```

---

## 📜 Licence

MIT