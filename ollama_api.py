import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


class OllamaError(RuntimeError):
    pass


_HEALTH_CACHE: Dict[str, Any] = {"ts": 0.0, "value": None}


def _base_url() -> str:
    raw = os.getenv("OLLAMA_BASE_URL")
    if not raw or not raw.strip():
        raw = "http://localhost:11434"
    return raw.rstrip("/")


def _list_models() -> List[str]:
    url = f"{_base_url()}/api/tags"
    try:
        r = requests.get(url, timeout=20)
    except requests.RequestException:
        return []
    if r.status_code >= 400:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    models = data.get("models")
    if not isinstance(models, list):
        return []
    out: List[str] = []
    for m in models:
        if isinstance(m, dict) and isinstance(m.get("name"), str):
            out.append(m["name"])
    return out


def _pick_model(requested: str, available: List[str]) -> Optional[str]:
    if not available:
        return None
    if requested in available:
        return requested

    family = (requested or "").split(":", 1)[0].strip()
    family_candidates = [m for m in available if m == family or m.startswith(f"{family}:")]

    def score(name: str) -> Tuple[int, int, int, str]:
        s = name.lower()
        return (
            1 if ("instruct" in s or "chat" in s) else 0,
            1 if (family and (s == family.lower() or s.startswith(family.lower() + ":"))) else 0,
            -len(name),
            name,
        )

    candidates = family_candidates or available
    return sorted(candidates, key=score, reverse=True)[0]


def _looks_like_missing_model(status_code: int, body_text: str) -> bool:
    if status_code != 404:
        return False
    t = (body_text or "").lower()
    return "model" in t and "not found" in t


def _suggest_models_text() -> str:
    models = _list_models()
    if not models:
        return "No pude listar modelos en /api/tags. Revisa que Ollama esté arrancado y accesible."
    shown = ", ".join(models[:12])
    more = "" if len(models) <= 12 else f" (+{len(models) - 12} más)"
    return f"Modelos instalados: {shown}{more}"


def health(*, max_age_s: int = 5) -> Dict[str, Any]:
    now = time.monotonic()
    cached_ts = float(_HEALTH_CACHE.get("ts") or 0.0)
    cached_val = _HEALTH_CACHE.get("value")
    if isinstance(cached_val, dict) and (now - cached_ts) <= float(max_age_s):
        return cached_val

    url = f"{_base_url()}/api/tags"
    try:
        r = requests.get(url, timeout=2)
    except requests.RequestException as e:
        val = {"ok": False, "base_url": _base_url(), "error": str(e), "models": []}
        _HEALTH_CACHE["ts"] = now
        _HEALTH_CACHE["value"] = val
        return val

    if r.status_code >= 400:
        val = {"ok": False, "base_url": _base_url(), "error": f"HTTP {r.status_code}: {r.text}", "models": []}
        _HEALTH_CACHE["ts"] = now
        _HEALTH_CACHE["value"] = val
        return val

    try:
        data = r.json()
    except Exception as e:
        val = {"ok": False, "base_url": _base_url(), "error": f"JSON inválido: {e}", "models": []}
        _HEALTH_CACHE["ts"] = now
        _HEALTH_CACHE["value"] = val
        return val

    models_raw = data.get("models")
    models: List[str] = []
    if isinstance(models_raw, list):
        for m in models_raw:
            if isinstance(m, dict) and isinstance(m.get("name"), str):
                models.append(m["name"])

    val = {"ok": True, "base_url": _base_url(), "error": None, "models": models}
    _HEALTH_CACHE["ts"] = now
    _HEALTH_CACHE["value"] = val
    return val



def chat(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
) -> str:
    model_name = model or os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b-instruct")
    num_ctx = int(num_ctx if num_ctx is not None else os.getenv("OLLAMA_NUM_CTX", "2048"))
    num_predict = int(num_predict if num_predict is not None else os.getenv("OLLAMA_NUM_PREDICT", "256"))
    url = f"{_base_url()}/api/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    try:
        r = requests.post(url, json=payload, timeout=int(os.getenv("OLLAMA_TIMEOUT", "90")))
    except requests.RequestException as e:
        raise OllamaError(f"No se puede conectar a Ollama en {url}: {e}") from e
    if r.status_code >= 400:
        if _looks_like_missing_model(r.status_code, r.text):
            available = _list_models()
            fallback = _pick_model(model_name, available)
            if fallback and fallback != model_name:
                payload2 = dict(payload)
                payload2["model"] = fallback
                try:
                    r2 = requests.post(url, json=payload2, timeout=int(os.getenv("OLLAMA_TIMEOUT", "90")))
                except requests.RequestException as e:
                    raise OllamaError(f"No se puede conectar a Ollama en {url}: {e}") from e
                if r2.status_code < 400:
                    data = r2.json()
                    return (data.get("message") or {}).get("content") or ""
            raise OllamaError(
                f"Error Ollama {r.status_code}: {r.text}\n"
                f"El modelo configurado ({model_name}) no está instalado. "
                f"Configura OLLAMA_CHAT_MODEL con uno existente o instala el modelo con 'ollama pull {model_name}'.\n"
                f"{_suggest_models_text()}"
            )
        raise OllamaError(f"Error Ollama {r.status_code}: {r.text}")
    data = r.json()
    return (data.get("message") or {}).get("content") or ""


def _try_embed_batch(texts: List[str], model_name: str) -> Optional[List[List[float]]]:
    url = f"{_base_url()}/api/embed"
    payload = {"model": model_name, "input": texts}
    try:
        r = requests.post(url, json=payload, timeout=int(os.getenv("OLLAMA_EMBED_TIMEOUT", "90")))
    except requests.RequestException:
        return None
    if r.status_code >= 400:
        if _looks_like_missing_model(r.status_code, r.text):
            available = _list_models()
            fallback = _pick_model(model_name, available)
            if fallback and fallback != model_name:
                try:
                    r2 = requests.post(
                        url, json={"model": fallback, "input": texts}, timeout=int(os.getenv("OLLAMA_EMBED_TIMEOUT", "90")))
                except requests.RequestException:
                    return None
                if r2.status_code < 400:
                    data = r2.json()
                    embeddings = data.get("embeddings")
                    if isinstance(embeddings, list):
                        return embeddings
        return None
    data = r.json()
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list):
        return None
    return embeddings


def embed(texts: List[str], *, model: Optional[str] = None) -> List[List[float]]:
    model_name = model or os.getenv("OLLAMA_EMBED_MODEL", "phi3:mini")
    if not texts:
        return []
    batch = _try_embed_batch(texts, model_name)
    if batch is not None and len(batch) == len(texts):
        return batch

    url = f"{_base_url()}/api/embeddings"
    out: List[List[float]] = []
    for t in texts:
        payload = {"model": model_name, "prompt": t}
        try:
            r = requests.post(url, json=payload, timeout=int(os.getenv("OLLAMA_EMBED_TIMEOUT", "90")))
        except requests.RequestException as e:
            raise OllamaError(f"Fallo embeddings en {url}: {e}") from e
        if r.status_code >= 400:
            if _looks_like_missing_model(r.status_code, r.text):
                available = _list_models()
                fallback = _pick_model(model_name, available)
                if fallback and fallback != model_name:
                    try:
                        r2 = requests.post(
                            url,
                            json={"model": fallback, "prompt": t},
                            timeout=int(os.getenv("OLLAMA_EMBED_TIMEOUT", "90")),
                        )
                    except requests.RequestException as e:
                        raise OllamaError(f"Fallo embeddings en {url}: {e}") from e
                    if r2.status_code < 400:
                        data = r2.json()
                        vec = data.get("embedding")
                        if isinstance(vec, list):
                            out.append(vec)
                            continue
                raise OllamaError(
                    f"Error embeddings {r.status_code}: {r.text}\n"
                    f"El modelo configurado ({model_name}) no está instalado. "
                    f"Configura OLLAMA_EMBED_MODEL con uno existente o instala el modelo con 'ollama pull {model_name}'.\n"
                    f"{_suggest_models_text()}"
                )
            raise OllamaError(f"Error embeddings {r.status_code}: {r.text}")
        data = r.json()
        vec = data.get("embedding")
        if not isinstance(vec, list):
            raise OllamaError(f"Respuesta embeddings inválida: {data}")
        out.append(vec)
    return out


def force_json(system: str, user: str) -> Dict[str, Any]:
    content = chat(
        [
            {
                "role": "system",
                "content": system
                + "\nDevuelve únicamente JSON válido. Sin texto adicional. Sin markdown.",
            },
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise
