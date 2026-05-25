import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def db_path() -> str:
    return os.getenv("APP_DB_PATH", os.path.join(os.getcwd(), "app.db"))


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT NOT NULL UNIQUE,
              password_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS boxers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              nickname TEXT,
              weight_class TEXT,
              stance TEXT,
              wins INTEGER NOT NULL DEFAULT 0,
              losses INTEGER NOT NULL DEFAULT 0,
              draws INTEGER NOT NULL DEFAULT 0,
              city TEXT,
              notes TEXT,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_pairs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              question TEXT NOT NULL,
              answer TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def fetch_one(query: str, args: Tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
    with connect() as conn:
        cur = conn.execute(query, args)
        return cur.fetchone()


def fetch_all(query: str, args: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
    with connect() as conn:
        cur = conn.execute(query, args)
        return cur.fetchall()


def execute(query: str, args: Tuple[Any, ...] = ()) -> int:
    with connect() as conn:
        cur = conn.execute(query, args)
        conn.commit()
        return int(cur.lastrowid or 0)


def seed_boxers_if_empty() -> None:
    row = fetch_one("SELECT COUNT(*) AS c FROM boxers")
    if not row or int(row["c"]) > 0:
        return
    sample = [
        ("Leo Martín", "El Zurdo", "Ligero", "Zurdo", 5, 1, 0, "Barcelona", "Jab rápido; trabaja salida lateral."),
        ("Amina Benali", "La Rápida", "Pluma", "Diestra", 7, 0, 0, "Bilbao", "Buen 1-2; entra y sale."),
        ("Carlos Vega", "El Muro", "Welter", "Diestra", 10, 2, 1, "Madrid", "Defensa alta; contragolpe."),
        ("Sofía Ruiz", "La Técnica", "Mosca", "Diestra", 3, 1, 0, "Valencia", "Paso atrás y jab al pecho."),
        ("Hugo Santos", "El Pitbull", "Medio", "Zurdo", 4, 3, 0, "Málaga", "Presión constante; ganchos al cuerpo."),
    ]
    with connect() as conn:
        for b in sample:
            conn.execute(
                """
                INSERT INTO boxers (name, nickname, weight_class, stance, wins, losses, draws, city, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*b, now_iso()),
            )
        conn.commit()


def seed_training_pairs_minimum() -> None:
    row = fetch_one("SELECT COUNT(*) AS c FROM training_pairs")
    existing = int(row["c"]) if row else 0
    if existing >= 30:
        return
    pairs = [
        (
            "Convierte esta intención en un JSON de petición: \"Dame las cuotas pendientes\".",
            "{\"action\":\"search_invoices\",\"params\":{\"status\":\"pendiente\"}}",
        ),
        (
            "Explica RAG con un ejemplo de boxeo en 2 frases.",
            "RAG busca en una base vectorial los párrafos más relevantes (chunks) sobre tu pregunta y luego la IA redacta la respuesta usando esos chunks. Ejemplo: preguntas por el jab, el sistema recupera tus apuntes de técnica y la IA lo explica sin inventar.",
        ),
        (
            "Cuando el contexto no contiene la respuesta, ¿qué haces?",
            "Digo que el contexto no contiene la información necesaria y pido o sugiero qué dato falta, sin inventar.",
        ),
        (
            "¿Qué es la guardia diestra (orthodox) y qué es la guardia zurda (southpaw)?",
            "La guardia diestra (orthodox) lleva el pie izquierdo delante y la mano izquierda como jab; la guardia zurda (southpaw) invierte la posición (pie derecho delante) y suele usar el jab con la mano derecha.",
        ),
        (
            "Define jab, cross, gancho y uppercut en una frase cada uno.",
            "Jab: golpe recto con la mano adelantada para medir y puntuar. Cross: recto potente con la mano atrasada. Gancho: golpe curvo lateral (a cabeza o cuerpo) con giro de cadera. Uppercut: golpe ascendente desde abajo, útil en corta distancia.",
        ),
        (
            "¿Qué es el 1-2 en boxeo y cuál es un error típico al hacerlo?",
            "El 1-2 es la combinación jab-cross. Un error típico es quedarse plantado sin recuperar la guardia o lanzar el cross sin girar la cadera y el pie trasero, perdiendo potencia y equilibrio.",
        ),
        (
            "Explica slip, parry y bloqueo alto (high guard) de forma simple.",
            "Slip: desviar la cabeza ligeramente fuera de la línea del golpe. Parry: desviar el golpe con un toque de la mano/antebrazo. Bloqueo alto: cubrirse con guantes y codos para absorber golpes, manteniendo la barbilla protegida.",
        ),
        (
            "¿Cómo se puntúa un asalto en boxeo amateur de forma general?",
            "De forma general se valora el golpe limpio y efectivo, la iniciativa, el control del ring y la defensa; en profesional suele usarse 10-9 por asalto para el ganador, y en amateur se prioriza la calidad y cantidad de golpes claros según el reglamento local.",
        ),
        (
            "Dame 5 consejos de seguridad para entrenar boxeo.",
            "Calienta y moviliza articulaciones, usa vendas y guantes adecuados, controla la intensidad en sparring, protege cabeza y dentadura cuando toque, y detente si hay mareo/dolor anormal; la técnica y la progresión pesan más que la fuerza bruta.",
        ),
        (
            "En fAIght, ¿para qué sirve el RAG?",
            "Sirve para cargar apuntes o teoría de boxeo en una base vectorial y que la IA recupere los chunks relevantes antes de responder, mejorando precisión y evitando inventar.",
        ),
        (
            "Convierte esta intención en un JSON de petición: \"Dame las cuotas pagadas\".",
            "{\"action\":\"search_invoices\",\"params\":{\"status\":\"pagada\"}}",
        ),
        (
            "Convierte esta intención en un JSON de petición: \"Cuotas pendientes del socio C003\".",
            "{\"action\":\"search_invoices\",\"params\":{\"status\":\"pendiente\",\"customer_id\":\"C003\"}}",
        ),
        (
            "Convierte esta intención en un JSON de petición: \"Dame el KPI de ingresos y pendientes\".",
            "{\"action\":\"kpi_sales\",\"params\":{}}",
        ),
        (
            "Convierte esta intención en un JSON de petición: \"Ranking de servicios más vendidos\".",
            "{\"action\":\"top_products\",\"params\":{\"limit\":5}}",
        ),
        (
            "Convierte esta intención en un JSON de petición: \"Lista de socios/boxeadores\".",
            "{\"action\":\"list_customers\",\"params\":{\"limit\":50}}",
        ),
        (
            "Si el usuario pide “cuotas pendientes”, ¿qué acción y status usas?",
            "Uso action = search_invoices y params.status = \"pendiente\".",
        ),
        (
            "¿Qué valores válidos tiene el status de facturas/cuotas en el MiniSaaS?",
            "Los valores válidos son \"pendiente\" y \"pagada\".",
        ),
        (
            "Si la consulta search_invoices devuelve 0 resultados, ¿cómo respondes?",
            "Digo que no se encontraron cuotas/facturas con esos filtros y sugiero revisar status o customer_id, sin inventar datos.",
        ),
        (
            "Si el usuario pide “todas las cuotas” sin filtrar, ¿qué JSON generas?",
            "{\"action\":\"search_invoices\",\"params\":{}}",
        ),
        (
            "Si te piden KPIs, ¿devuelves JSON o texto?",
            "En el paso MCP devuelvo JSON válido para ejecutar kpi_sales; luego interpreto el resultado en español claro.",
        ),
        (
            "¿Qué haces si el modelo no está instalado en Ollama?",
            "Devuelvo un error claro indicando que el modelo no existe y sugiero elegir uno instalado o instalarlo con ollama pull.",
        ),
        (
            "En una respuesta RAG, ¿puedes usar información que no esté en los chunks?",
            "No. Solo uso los chunks recuperados; si no hay evidencia, lo digo y no invento.",
        ),
        (
            "¿Qué recomendación das si el RAG está vacío?",
            "Indico que la colección no tiene datos y pido que ingesten apuntes antes de preguntar.",
        ),
        (
            "Escribe una respuesta breve a: \"Me duele la muñeca al hacer saco\".",
            "Reduce intensidad, revisa técnica y alineación de muñeca, usa vendas/guantes adecuados y descansa; si el dolor persiste o hay inflamación, consulta a un profesional.",
        ),
        (
            "Dame un checklist de 4 puntos para un jab correcto.",
            "Barbilla abajo y mano contraria protegiendo, extensión recta sin abrir el codo, pequeño paso/transferencia de peso para medir distancia, y regreso rápido a guardia.",
        ),
        (
            "Dame un consejo para no quedarte en la línea tras el 1-2.",
            "Tras el cross, recupera guardia y sal con paso lateral o pivot, evitando quedarte estático delante del rival.",
        ),
        (
            "¿Qué significa “no telegrafiar” un golpe?",
            "Significa no avisar con gestos previos (cargar el hombro, bajar la mano o tensarte) para que el rival no lo vea venir.",
        ),
        (
            "Dame una respuesta de 3 líneas a: \"¿Cómo mejoro la defensa?\".",
            "Trabaja guardia alta, slips y parries en sombra y manoplas. Prioriza pies: entra y sal de ángulo, no solo mover la cabeza. Grábate y corrige errores básicos antes de subir intensidad.",
        ),
        (
            "Si el usuario pide “ranking de servicios”, ¿qué tool del SaaS usas y qué devuelve?",
            "Uso top_products y devuelve una lista de productos/servicios con sus ventas en euros.",
        ),
    ]
    with connect() as conn:
        for q, a in pairs:
            exists = conn.execute(
                "SELECT 1 FROM training_pairs WHERE question = ? LIMIT 1", (q,)
            ).fetchone()
            if exists:
                continue
            conn.execute(
                "INSERT INTO training_pairs (question, answer, created_at) VALUES (?, ?, ?)",
                (q, a, now_iso()),
            )
        conn.commit()
