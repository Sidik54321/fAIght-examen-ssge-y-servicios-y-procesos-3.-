import json
import os
import random
import socket
import subprocess
from typing import Any, Dict, Optional, Tuple

import requests
from flask import Flask, has_request_context, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

import db
import rag_engine
import saas_backend
from agentic_mission import run_agent
from ollama_api import OllamaError, chat, force_json, health


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-me")

BOXING_KNOWLEDGE_SEED = """
Boxeo: conceptos básicos

1) Guardia y postura
- Guardia diestra (orthodox): pie izquierdo delante, jab con la izquierda, cross con la derecha.
- Guardia zurda (southpaw): pie derecho delante, jab con la derecha, cross con la izquierda.
- Puntos clave: barbilla abajo, hombros relajados, manos arriba, codos cerca del cuerpo, peso equilibrado, talones ligeros.

2) Golpes principales
- Jab: recto con la mano adelantada para medir distancia, puntuar y preparar combinaciones.
- Cross: recto con la mano atrasada, más potencia; requiere giro de cadera y pivot del pie trasero.
- Gancho: golpe curvo lateral, útil a cabeza y cuerpo; potencia desde cadera y apoyo.
- Uppercut: golpe ascendente desde corta distancia; mantener la muñeca firme y no “telegrafiar”.

3) Distancia y pies
- Larga: jab y control del ritmo.
- Media: combinaciones (1-2, 1-2-3) y entradas/salidas.
- Corta: ganchos, uppercuts y trabajo de clinch según reglamento.
- Principio: entra con intención y sal con seguridad; no te quedes parado tras golpear.

4) Defensa básica
- High guard (bloqueo alto): guantes arriba, codos protegiendo costillas.
- Slip: mover la cabeza fuera de la línea del golpe con un desplazamiento mínimo.
- Parry: desviar el golpe con un toque corto y volver a la guardia.
- Paso atrás / pivot: usar pies para salir del ángulo, no solo “echar el cuerpo atrás”.

5) Combinaciones típicas
- 1-2 (jab-cross): básica para puntuar y marcar potencia.
- 1-2-3 (jab-cross-gancho): cambia plano y rompe la defensa.
- Jab al cuerpo + cross a la cabeza: alternar niveles.

6) Errores comunes (principiante)
- Bajar la mano al lanzar el cross.
- No girar cadera/pies y perder potencia.
- Mirar al suelo o cerrar los ojos.
- Quedarse en la línea tras golpear (sin salida lateral).

7) Seguridad y entrenamiento
- Calentamiento y movilidad antes de pegar fuerte.
- Vendas y guantes adecuados; en sparring, control y protección.
- Técnica antes que potencia; progresión gradual.
""".strip()


def _json_body() -> Dict[str, Any]:
    if request.is_json:
        return request.get_json(silent=True) or {}
    return {}


def _default_collection() -> str:
    if has_request_context():
        c = session.get("rag_collection")
        if isinstance(c, str) and c.strip():
            return c.strip()
    return os.getenv("RAG_COLLECTION", "boxeo_apuntes")


def _current_user() -> Optional[Dict[str, Any]]:
    user_id = session.get("user_id")
    if not user_id:
        return None
    row = db.fetch_one("SELECT id, username FROM users WHERE id = ?", (int(user_id),))
    if not row:
        return None
    return {"id": int(row["id"]), "username": str(row["username"])}


def _require_login() -> Optional[Any]:
    if _current_user():
        return None
    return redirect(url_for("login", next=request.path))


@app.before_request
def guard():
    path = request.path or "/"
    if path.startswith("/static/"):
        return None
    if path == "/ollama" or path.startswith("/ollama/"):
        return None
    if path.startswith("/api/ollama/"):
        return None
    if path in {"/login", "/register"}:
        return None
    if path.startswith("/api/"):
        if _current_user():
            return None
        return jsonify({"error": "No autenticado"}), 401
    if _current_user():
        return None
    return redirect(url_for("login", next=path))


def _ollama_proxy_target() -> str:
    raw = os.getenv("OLLAMA_PROXY_TARGET") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    target = (raw or "").strip() or "http://localhost:11434"
    target = target.rstrip("/")
    if target.endswith("/ollama"):
        target = target[: -len("/ollama")].rstrip("/")
    if target.startswith("http://127.0.0.1:5001") or target.startswith("http://localhost:5001"):
        target = "http://localhost:11434"
    return target


@app.route("/ollama/<path:subpath>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def ollama_proxy(subpath: str):
    target = _ollama_proxy_target()
    url = f"{target}/{subpath.lstrip('/')}"
    headers = {}
    for k, v in request.headers.items():
        lk = k.lower()
        if lk in {"host", "content-length", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}:
            continue
        headers[k] = v
    try:
        r = requests.request(
            method=request.method,
            url=url,
            params=request.args,
            data=request.get_data(),
            headers=headers,
            timeout=int(os.getenv("OLLAMA_PROXY_TIMEOUT", "120")),
        )
    except requests.RequestException as e:
        return jsonify({"error": f"Proxy Ollama falló hacia {target}: {e}"}), 502
    resp_headers = {}
    for k, v in r.headers.items():
        lk = k.lower()
        if lk in {"content-length", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}:
            continue
        resp_headers[k] = v
    return (r.content, int(r.status_code), resp_headers)


@app.route("/api/ollama/<path:subpath>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def api_ollama_proxy(subpath: str):
    return ollama_proxy(subpath)


def _seed_admin_if_empty() -> None:
    row = db.fetch_one("SELECT COUNT(*) AS c FROM users")
    if row and int(row["c"]) > 0:
        return
    username = os.getenv("DEMO_ADMIN_USER", "admin")
    password = os.getenv("DEMO_ADMIN_PASS", "admin123")
    db.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, generate_password_hash(password), db.now_iso()),
    )


def run_mcp(question: str) -> Tuple[int, Dict[str, Any]]:
    question = (question or "").strip()
    if not question:
        return 400, {"error": "Falta question"}

    system = (
        "Eres un traductor de lenguaje natural a peticiones JSON para un MiniSaaS de un club/gimnasio de boxeo.\n"
        "El usuario hace una pregunta humana. Tú devuelves un JSON con:\n"
        '- "action": string (una de: list_customers, search_invoices, kpi_sales, top_products)\n'
        '- "params": objeto (puede estar vacío)\n'
        "Reglas:\n"
        "- Si el usuario pide cuotas/facturas por estado, usa search_invoices con status pagada o pendiente.\n"
        "- Si pide KPIs o resumen de ingresos, usa kpi_sales.\n"
        "- Si pide ranking de servicios (clases, planes), usa top_products.\n"
        "- Si pide socios/boxeadores, usa list_customers.\n"
    )
    try:
        tool_call = force_json(system, question)
    except Exception as e:
        return 502, {"error": f"No pude generar JSON: {e}"}

    action = tool_call.get("action")
    params = tool_call.get("params") if isinstance(tool_call.get("params"), dict) else {}
    if action not in {"list_customers", "search_invoices", "kpi_sales", "top_products"}:
        return 400, {"error": f"Acción inválida: {action}", "tool_call": tool_call}

    try:
        service_data = saas_backend.execute(action, params)
    except Exception as e:
        return 500, {"error": f"Error ejecutando SaaS: {e}", "tool_call": tool_call}

    interpret_system = (
        "Eres un analista de sistemas de gestión empresarial aplicado a un club/gimnasio de boxeo. Responde en español claro y directo.\n"
        "Usa únicamente los datos del SaaS. Si faltan datos, dilo."
    )
    interpret_user = json.dumps(
        {"pregunta": question, "peticion_json": tool_call, "respuesta_saas": service_data},
        ensure_ascii=False,
        indent=2,
    )
    try:
        answer = chat(
            [{"role": "system", "content": interpret_system}, {"role": "user", "content": interpret_user}],
            temperature=0.2,
        ).strip()
    except OllamaError as e:
        return 502, {"error": str(e), "tool_call": tool_call, "saas": service_data}

    return 200, {"tool_call": tool_call, "saas": service_data, "answer": answer}


def rag_ingest(collection: str, text: str, source: str) -> Tuple[int, Dict[str, Any]]:
    collection = (collection or _default_collection()).strip()
    text = (text or "").strip()
    source = (source or "manual").strip()
    if not text:
        return 400, {"error": "Falta text"}
    try:
        result = rag_engine.ingest_text(collection, text, source=source)
    except Exception as e:
        msg = str(e)
        if "dimension" in msg.lower() or "embedding" in msg.lower():
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", "phi3:mini")
            return 400, {"error": f"Incompatibilidad de embeddings en la colección. Usa siempre el mismo modelo de embeddings (OLLAMA_EMBED_MODEL={embed_model}) o reinicia la base de Chroma (carpeta chroma_db). Detalle: {msg}"}
        return 500, {"error": f"Error ingesta: {msg}"}
    except OllamaError as e:
        return 502, {"error": str(e)}
    return 200, {"collection": collection, **result}


def rag_ask(collection: str, question: str, k: int) -> Tuple[int, Dict[str, Any]]:
    collection = (collection or _default_collection()).strip()
    question = (question or "").strip()
    k = int(k or 5)
    if not question:
        return 400, {"error": "Falta question"}
    try:
        chunks = rag_engine.query(collection, question, k=k)
    except Exception as e:
        msg = str(e)
        if "dimension" in msg.lower() or "embedding" in msg.lower():
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", "phi3:mini")
            return 400, {"error": f"Incompatibilidad de embeddings en la colección. Usa siempre el mismo modelo de embeddings (OLLAMA_EMBED_MODEL={embed_model}) o reinicia la base de Chroma (carpeta chroma_db). Detalle: {msg}"}
        return 500, {"error": f"Error RAG: {msg}"}
    except OllamaError as e:
        return 502, {"error": str(e)}

    context = "\n\n".join([f"[Chunk {i+1}]\n{(c['chunk'] or '')[:280]}" for i, c in enumerate(chunks)])
    if not chunks:
        return 200, {"collection": collection, "question": question, "chunks": [], "answer": "No hay datos en esa colección. Primero ingesta/aprende apuntes."}

    system = (
        "Eres un asistente de boxeo que responde usando un RAG.\n"
        "Responde en español claro y breve (máximo 6 líneas).\n"
        "No inventes: si el contexto no contiene la respuesta, dilo.\n"
    )
    user = f"Pregunta: {question}\n\nContexto:\n{context}"
    try:
        rag_model = os.getenv("OLLAMA_RAG_MODEL", "qwen2.5:3b-instruct")
        answer = chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=rag_model,
            temperature=0.2,
            num_ctx=1024,
            num_predict=120,
        ).strip()
    except OllamaError as e:
        return 502, {"error": str(e), "chunks": chunks}
    return 200, {"collection": collection, "question": question, "chunks": chunks, "answer": answer}


def _template_ctx(active: str) -> Dict[str, Any]:
    collections = []
    try:
        collections = rag_engine.list_collections()
    except Exception:
        collections = []
    default = _default_collection()
    if default and default not in collections:
        collections = [default] + collections
    h = health()
    return {
        "active": active,
        "user": _current_user(),
        "default_collection": default,
        "collections": collections,
        "chat_model": os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b-instruct"),
        "embed_model": os.getenv("OLLAMA_EMBED_MODEL", "phi3:mini"),
        "tuned_model": os.getenv("OLLAMA_TUNED_MODEL", "ceacfp-tuned"),
        "ollama_ok": bool(h.get("ok")),
        "ollama_base_url": str(h.get("base_url") or ""),
        "ollama_error": h.get("error"),
        "ollama_models": list(h.get("models") or [])[:12],
    }


@app.get("/login")
def login():
    return render_template("login.html", next=request.args.get("next") or "/")


@app.post("/login")
def login_post():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    next_url = request.form.get("next") or "/"
    row = db.fetch_one("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
    if not row or not check_password_hash(row["password_hash"], password):
        return render_template("login.html", next=next_url, error="Usuario o contraseña incorrectos")
    session["user_id"] = int(row["id"])
    return redirect(next_url)


@app.get("/register")
def register():
    return render_template("register.html", next=request.args.get("next") or "/")


@app.post("/register")
def register_post():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    next_url = request.form.get("next") or "/"
    if len(username) < 3 or len(password) < 6:
        return render_template("register.html", next=next_url, error="Usuario mínimo 3 caracteres; contraseña mínimo 6")
    exists = db.fetch_one("SELECT id FROM users WHERE username = ?", (username,))
    if exists:
        return render_template("register.html", next=next_url, error="Ese usuario ya existe")
    user_id = db.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, generate_password_hash(password), db.now_iso()),
    )
    session["user_id"] = int(user_id)
    return redirect(next_url)


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
def dashboard():
    boxer_count = db.fetch_one("SELECT COUNT(*) AS c FROM boxers")
    pending = saas_backend.execute("search_invoices", {"status": "pendiente"}).get("invoices", [])
    kpi = saas_backend.execute("kpi_sales", {}).get("kpi", {})
    boxer_total = int(boxer_count["c"]) if boxer_count else 0
    return render_template(
        "dashboard.html",
        **_template_ctx("dashboard"),
        boxer_count=boxer_total,
        pending_invoices=pending,
        kpi=kpi,
    )


@app.get("/boxers")
def boxers():
    rows = db.fetch_all("SELECT * FROM boxers ORDER BY created_at DESC, id DESC")
    return render_template("boxers.html", **_template_ctx("boxers"), boxers=[dict(r) for r in rows])


@app.post("/boxers")
def boxers_add():
    name = (request.form.get("name") or "").strip()
    if not name:
        return redirect(url_for("boxers"))
    nickname = (request.form.get("nickname") or "").strip() or None
    weight_class = (request.form.get("weight_class") or "").strip() or None
    stance = (request.form.get("stance") or "").strip() or None
    wins = int(request.form.get("wins") or 0)
    losses = int(request.form.get("losses") or 0)
    draws = int(request.form.get("draws") or 0)
    city = (request.form.get("city") or "").strip() or None
    notes = (request.form.get("notes") or "").strip() or None
    db.execute(
        """
        INSERT INTO boxers (name, nickname, weight_class, stance, wins, losses, draws, city, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (name, nickname, weight_class, stance, wins, losses, draws, city, notes, db.now_iso()),
    )
    return redirect(url_for("boxers"))


@app.get("/boxers/<int:boxer_id>")
def boxer_edit(boxer_id: int):
    row = db.fetch_one("SELECT * FROM boxers WHERE id = ?", (boxer_id,))
    if not row:
        return redirect(url_for("boxers"))
    return render_template("boxer_edit.html", **_template_ctx("boxers"), boxer=dict(row))


@app.post("/boxers/<int:boxer_id>/save")
def boxer_save(boxer_id: int):
    row = db.fetch_one("SELECT id FROM boxers WHERE id = ?", (boxer_id,))
    if not row:
        return redirect(url_for("boxers"))
    name = (request.form.get("name") or "").strip()
    if not name:
        return redirect(url_for("boxer_edit", boxer_id=boxer_id))
    nickname = (request.form.get("nickname") or "").strip() or None
    weight_class = (request.form.get("weight_class") or "").strip() or None
    stance = (request.form.get("stance") or "").strip() or None
    wins = int(request.form.get("wins") or 0)
    losses = int(request.form.get("losses") or 0)
    draws = int(request.form.get("draws") or 0)
    city = (request.form.get("city") or "").strip() or None
    notes = (request.form.get("notes") or "").strip() or None
    db.execute(
        """
        UPDATE boxers
        SET name = ?, nickname = ?, weight_class = ?, stance = ?, wins = ?, losses = ?, draws = ?, city = ?, notes = ?
        WHERE id = ?
        """,
        (name, nickname, weight_class, stance, wins, losses, draws, city, notes, boxer_id),
    )
    return redirect(url_for("boxers"))


@app.post("/boxers/<int:boxer_id>/delete")
def boxer_delete(boxer_id: int):
    db.execute("DELETE FROM boxers WHERE id = ?", (boxer_id,))
    return redirect(url_for("boxers"))


@app.get("/mcp")
def mcp_page():
    return render_template("mcp.html", **_template_ctx("mcp"))


@app.post("/mcp")
def mcp_page_post():
    question = (request.form.get("question") or "").strip()
    status, data = run_mcp(question)
    return render_template("mcp.html", **_template_ctx("mcp"), question=question, result=data, status=status)


@app.get("/rag")
def rag_page():
    selected = _default_collection()
    preview = rag_engine.collection_preview(selected, limit=20, offset=0) if selected else []
    return render_template(
        "rag.html",
        **_template_ctx("rag"),
        selected_collection=selected,
        selected_count=rag_engine.collection_count(selected) if selected else 0,
        selected_preview=preview,
    )


@app.post("/rag/ingest")
def rag_ingest_form():
    collection = (request.form.get("collection") or _default_collection()).strip()
    text = (request.form.get("text") or "").strip()
    source = (request.form.get("source") or "manual").strip()
    status, data = rag_ingest(collection, text, source)
    session["rag_collection"] = collection
    return render_template(
        "rag.html",
        **_template_ctx("rag"),
        ingest=data,
        ingest_status=status,
        selected_collection=collection,
        selected_count=rag_engine.collection_count(collection) if collection else 0,
        selected_preview=rag_engine.collection_preview(collection, limit=20, offset=0) if collection else [],
    )


@app.post("/rag/ask")
def rag_ask_form():
    collection = (request.form.get("collection") or _default_collection()).strip()
    question = (request.form.get("question") or "").strip()
    status, data = rag_ask(collection, question, 2)
    session["rag_collection"] = collection
    return render_template(
        "rag.html",
        **_template_ctx("rag"),
        ask=data,
        ask_status=status,
        question=question,
        selected_collection=collection,
        selected_count=rag_engine.collection_count(collection) if collection else 0,
        selected_preview=rag_engine.collection_preview(collection, limit=20, offset=0) if collection else [],
    )


@app.post("/rag/select")
def rag_select_collection():
    collection = (request.form.get("collection") or _default_collection()).strip()
    if collection:
        session["rag_collection"] = collection
    return redirect(url_for("rag_page"))


@app.post("/rag/clear")
def rag_clear_collection():
    collection = (request.form.get("collection") or _default_collection()).strip()
    if collection:
        session["rag_collection"] = collection
        rag_engine.clear_collection(collection)
    return redirect(url_for("rag_page"))


@app.post("/rag/delete")
def rag_delete_chunk():
    collection = (request.form.get("collection") or _default_collection()).strip()
    chunk_id = (request.form.get("chunk_id") or "").strip()
    if collection and chunk_id:
        session["rag_collection"] = collection
        try:
            rag_engine.delete_ids(collection, [chunk_id])
        except Exception:
            pass
    return redirect(url_for("rag_page"))


@app.get("/training")
def training_page():
    pairs = db.fetch_all("SELECT * FROM training_pairs ORDER BY created_at DESC, id DESC")
    return render_template("training.html", **_template_ctx("training"), pairs=[dict(r) for r in pairs])


@app.post("/training/pairs")
def training_add_pair():
    q = (request.form.get("question") or "").strip()
    a = (request.form.get("answer") or "").strip()
    if q and a:
        db.execute("INSERT INTO training_pairs (question, answer, created_at) VALUES (?, ?, ?)", (q, a, db.now_iso()))
    return redirect(url_for("training_page"))


@app.post("/training/pairs/<int:pair_id>/delete")
def training_delete_pair(pair_id: int):
    db.execute("DELETE FROM training_pairs WHERE id = ?", (pair_id,))
    return redirect(url_for("training_page"))


@app.post("/training/export")
def training_export():
    pairs = db.fetch_all("SELECT question, answer FROM training_pairs ORDER BY id ASC")
    data = [{"question": str(r["question"]), "answer": str(r["answer"])} for r in pairs]
    path = os.path.join(os.getcwd(), "training_data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return redirect(url_for("training_page"))


@app.post("/training/run")
def training_run():
    try:
        p = subprocess.run(
            ["python", "train_ollama_model.py"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        status = int(p.returncode or 0)
    except Exception as e:
        output = str(e)
        status = 1
    pairs = db.fetch_all("SELECT * FROM training_pairs ORDER BY created_at DESC, id DESC")
    return render_template(
        "training.html",
        **_template_ctx("training"),
        pairs=[dict(r) for r in pairs],
        train_output=output,
        train_status=status,
    )


@app.post("/training/compare")
def training_compare():
    prompt = (request.form.get("prompt") or "").strip()
    base_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b-instruct")
    tuned_model = os.getenv("OLLAMA_TUNED_MODEL", "ceacfp-tuned")
    before = ""
    after = ""
    error = None
    if prompt:
        try:
            before = chat(
                [{"role": "system", "content": "Responde en español."}, {"role": "user", "content": prompt}],
                model=base_model,
                temperature=0.2,
            ).strip()
            after = chat(
                [{"role": "system", "content": "Responde en español."}, {"role": "user", "content": prompt}],
                model=tuned_model,
                temperature=0.2,
            ).strip()
        except Exception as e:
            error = str(e)
    pairs = db.fetch_all("SELECT * FROM training_pairs ORDER BY created_at DESC, id DESC")
    return render_template(
        "training.html",
        **_template_ctx("training"),
        pairs=[dict(r) for r in pairs],
        compare_prompt=prompt,
        compare_before=before,
        compare_after=after,
        compare_error=error,
    )


@app.get("/agent")
def agent_page():
    return render_template("agent.html", **_template_ctx("agent"))


@app.post("/agent/run")
def agent_run():
    mission = (request.form.get("mission") or "").strip()
    if not mission:
        mission = "Genera un informe del club con KPIs, cuotas pendientes y un consejo técnico sobre el jab."
    try:
        report = run_agent(mission)
        status = 0
        error = None
    except Exception as e:
        report = ""
        status = 1
        error = str(e)
    return render_template("agent.html", **_template_ctx("agent"), mission=mission, report=report, run_status=status, run_error=error)


@app.get("/coach")
def coach_page():
    return render_template("coach.html", **_template_ctx("coach"))


@app.post("/coach")
def coach_post():
    prompt = (request.form.get("prompt") or "").strip()
    answer = ""
    error = None
    model_used = None
    if prompt:
        system = (
            "Eres un entrenador de boxeo. Responde en español claro, directo y seguro.\n"
            "Da pasos accionables (técnica, guardia, pies, defensa, combinaciones) y errores comunes.\n"
            "Si la pregunta es de salud/lesiones, recomienda consultar a un profesional y prioriza la seguridad.\n"
        )
        tuned = os.getenv("OLLAMA_TUNED_MODEL", "ceacfp-tuned")
        base = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b-instruct")
        coach_primary = os.getenv("OLLAMA_COACH_MODEL") or tuned
        coach_num_ctx = int(os.getenv("OLLAMA_COACH_NUM_CTX", "1024"))
        coach_num_predict = int(os.getenv("OLLAMA_COACH_NUM_PREDICT", "160"))
        try:
            model_used = coach_primary
            answer = chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                model=coach_primary,
                temperature=0.3,
                num_ctx=coach_num_ctx,
                num_predict=coach_num_predict,
            ).strip()
        except Exception:
            try:
                model_used = base
                answer = chat(
                    [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    model=base,
                    temperature=0.3,
                    num_ctx=coach_num_ctx,
                    num_predict=coach_num_predict,
                ).strip()
            except Exception as e:
                error = str(e)
                model_used = None
                answer = ""
        if answer:
            try:
                exists = db.fetch_one("SELECT 1 FROM training_pairs WHERE question = ? LIMIT 1", (prompt,))
                if not exists:
                    db.execute(
                        "INSERT INTO training_pairs (question, answer, created_at) VALUES (?, ?, ?)",
                        (prompt, answer, db.now_iso()),
                    )
            except Exception:
                pass

    return render_template(
        "coach.html",
        **_template_ctx("coach"),
        prompt=prompt,
        answer=answer,
        coach_error=error,
        model_used=model_used,
    )


@app.get("/css3d")
def css3d():
    return render_template("css3d.html", **_template_ctx("css3d"))


def _boxer_public(row: Any) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "name": str(row["name"]),
        "nickname": str(row["nickname"]) if row["nickname"] is not None else None,
        "weight_class": str(row["weight_class"]) if row["weight_class"] is not None else None,
        "stance": str(row["stance"]) if row["stance"] is not None else None,
        "wins": int(row["wins"] or 0),
        "losses": int(row["losses"] or 0),
        "draws": int(row["draws"] or 0),
        "city": str(row["city"]) if row["city"] is not None else None,
    }


def _strength(b: Dict[str, Any]) -> float:
    wins = float(b["wins"])
    losses = float(b["losses"])
    draws = float(b["draws"])
    experience = wins + losses + draws
    efficiency = (wins + 1.0) / (losses + 1.0)
    return 8.0 + efficiency * 6.0 + experience * 0.35 + draws * 0.15


def _round_line(r: int, a: Dict[str, Any], b: Dict[str, Any], winner: str) -> str:
    a_name = a["name"]
    b_name = b["name"]
    scripts = [
        f"R{r}: {a_name} marca el jab y corta la distancia.",
        f"R{r}: {b_name} responde con contra y defensa alta.",
        f"R{r}: Intercambio en corto, suben los ganchos al cuerpo.",
        f"R{r}: Ritmo alto, trabajo de pies y salidas laterales.",
    ]
    base = scripts[(r - 1) % len(scripts)]
    return f"{base} 10-9 {winner}"


@app.get("/api/boxers")
def api_boxers():
    rows = db.fetch_all(
        "SELECT id, name, nickname, weight_class, stance, wins, losses, draws, city FROM boxers ORDER BY created_at DESC, id DESC"
    )
    return jsonify({"boxers": [_boxer_public(r) for r in rows]})


@app.post("/api/boxers")
def api_boxers_create():
    body = _json_body()
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Falta name"}), 400
    nickname = (body.get("nickname") or "").strip() or None
    weight_class = (body.get("weight_class") or "").strip() or None
    stance = (body.get("stance") or "").strip() or None
    city = (body.get("city") or "").strip() or None
    notes = (body.get("notes") or "").strip() or None
    try:
        wins = int(body.get("wins") or 0)
        losses = int(body.get("losses") or 0)
        draws = int(body.get("draws") or 0)
    except Exception:
        return jsonify({"error": "Wins/losses/draws inválidos"}), 400
    if wins < 0 or losses < 0 or draws < 0:
        return jsonify({"error": "Wins/losses/draws no pueden ser negativos"}), 400

    boxer_id = db.execute(
        """
        INSERT INTO boxers (name, nickname, weight_class, stance, wins, losses, draws, city, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (name, nickname, weight_class, stance, wins, losses, draws, city, notes, db.now_iso()),
    )
    row = db.fetch_one("SELECT * FROM boxers WHERE id = ?", (boxer_id,))
    return jsonify({"boxer": _boxer_public(row)}), 201


@app.delete("/api/boxers/<int:boxer_id>")
def api_boxers_delete(boxer_id: int):
    row = db.fetch_one("SELECT id FROM boxers WHERE id = ?", (boxer_id,))
    if not row:
        return jsonify({"error": "No encontrado"}), 404
    db.execute("DELETE FROM boxers WHERE id = ?", (boxer_id,))
    return jsonify({"ok": True})


@app.post("/api/fight")
def api_fight():
    body = _json_body()
    try:
        a_id = int(body.get("boxer_a_id") or 0)
        b_id = int(body.get("boxer_b_id") or 0)
    except Exception:
        return jsonify({"error": "IDs inválidos"}), 400
    if not a_id or not b_id or a_id == b_id:
        return jsonify({"error": "Selecciona dos boxeadores distintos"}), 400

    a_row = db.fetch_one("SELECT * FROM boxers WHERE id = ?", (a_id,))
    b_row = db.fetch_one("SELECT * FROM boxers WHERE id = ?", (b_id,))
    if not a_row or not b_row:
        return jsonify({"error": "Boxeador no encontrado"}), 404

    a = _boxer_public(a_row)
    b = _boxer_public(b_row)

    sa = _strength(a)
    sb = _strength(b)

    points_a = 0
    points_b = 0
    log = []
    ko = None

    for r in range(1, 4):
        ra = sa + random.gauss(0.0, 1.4)
        rb = sb + random.gauss(0.0, 1.4)
        diff = ra - rb
        ko_prob = max(0.0, abs(diff) - 5.0) / 40.0
        if random.random() < ko_prob:
            winner = "A" if diff > 0 else "B"
            ko = {"round": r, "winner": winner}
            break
        if diff >= 0:
            points_a += 10
            points_b += 9
            log.append(_round_line(r, a, b, "A"))
        else:
            points_a += 9
            points_b += 10
            log.append(_round_line(r, a, b, "B"))

    if ko:
        winner_id = a["id"] if ko["winner"] == "A" else b["id"]
        winner_name = a["name"] if ko["winner"] == "A" else b["name"]
        headline = f"KO en el round {ko['round']}: {winner_name}"
        meta = f"{a['name']} vs {b['name']}"
        log.append(headline)
        return jsonify({"winner_id": winner_id, "headline": headline, "meta": meta, "log": log})

    if points_a == points_b:
        winner_id = random.choice([a["id"], b["id"]])
        headline = "Empate técnico (ajustado)"
    elif points_a > points_b:
        winner_id = a["id"]
        headline = f"Decisión para {a['name']}"
    else:
        winner_id = b["id"]
        headline = f"Decisión para {b['name']}"

    meta = f"Puntuación: A {points_a} - {points_b} B"

    try:
        commentary = chat(
            [
                {
                    "role": "system",
                    "content": "Eres comentarista de boxeo. Redacta 4-6 líneas de resumen del combate en español, sin inventar datos fuera de lo dado.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"A": a, "B": b, "puntuacion": {"A": points_a, "B": points_b}, "log": log},
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.4,
        ).strip()
        if commentary:
            log.append("")
            log.append(commentary)
    except Exception:
        pass

    return jsonify({"winner_id": winner_id, "headline": headline, "meta": meta, "log": log})


@app.post("/api/mcp")
def api_mcp():
    body = _json_body()
    status, data = run_mcp(body.get("question"))
    return jsonify(data), status


@app.post("/api/rag/ingest")
def api_rag_ingest():
    body = _json_body()
    status, data = rag_ingest(body.get("collection"), body.get("text"), body.get("source"))
    return jsonify(data), status


@app.post("/api/rag/ask")
def api_rag_ask():
    body = _json_body()
    status, data = rag_ask(body.get("collection"), body.get("question"), int(body.get("k") or 5))
    return jsonify(data), status


def bootstrap() -> None:
    db.init_db()
    _seed_admin_if_empty()
    db.seed_boxers_if_empty()
    db.seed_training_pairs_minimum()
    try:
        rag_engine.ensure_seeded(_default_collection(), BOXING_KNOWLEDGE_SEED)
    except Exception:
        pass


bootstrap()


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5001"))
    app.run(host=host, port=port, debug=True, use_reloader=False)
