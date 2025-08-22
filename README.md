# slm-api-server

Flask API autour de `llama-cpp-python` avec **SSE streaming**, **chargement local de GGUF** et **/load** (Hugging Face).

## Démarrage local

```bash
cp .env.example .env
# édite .env, fixe MODEL_PATH ou utilise /load + /reload
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
make dev  # lance Flask en debug
```

## Docker

```
cp .env.example .env
docker compose up --build -d
```

## API

- GET /health – ping + modèle
- GET /models – modèle actif + liste de GGUF dans $MODELS_DIR
- POST /load – { repo_id, filename } → télécharge vers $MODELS_DIR
- POST /reload – { model_path } → active le modèle
- POST /v1/chat/completions – stream SSE si stream: true
- POST /v1/extract – extraction JSON avec schéma (non-stream)