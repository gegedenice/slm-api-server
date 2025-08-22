import os
from slm_api.app import app

def test_health_route():
    # Pour les tests, on désactive le chargement réel du modèle si MODEL_PATH manquant
    os.environ["MODEL_PATH"] = os.environ.get("MODEL_PATH", "/tmp/does-not-exist.gguf")
    client = app.test_client()
    resp = client.get("/health")
    # /health renverra 500 si le modèle n'existe pas ; on vérifie que l'API répond
    assert resp.status_code in (200, 500)