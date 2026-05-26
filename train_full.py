"""
train_full.py — Entrena ceacfp-tuned con TODOS los datos locales del club.

Embebe en el modelo:
  - Todos los socios y boxeadores de la BD
  - Todas las facturas/cuotas y KPIs
  - Todos los servicios y ventas
  - Todo el conocimiento técnico de boxeo
  - Todos los pares Q&A de entrenamiento acumulados
  - Las reglas MCP (JSON) para consultas al SaaS

Al terminar, el modelo responde de MEMORIA sin llamadas externas ni RAG.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Asegura imports locales
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db
import saas_backend

BOXING_KNOWLEDGE = """
1) GUARDIA Y POSTURA
- Orthodox (diestra): pie izquierdo delante, jab izquierda, cross derecha.
- Southpaw (zurda): pie derecho delante, jab derecha, cross izquierda.
- Claves: barbilla abajo, manos arriba, codos al cuerpo, peso equilibrado, talones ligeros.

2) GOLPES PRINCIPALES
- Jab: recto mano adelantada; mide distancia y puntúa.
- Cross: recto mano atrasada; más potencia, requiere giro de cadera y pivot.
- Gancho: curvo lateral a cabeza o cuerpo; potencia desde cadera.
- Uppercut: ascendente desde corta distancia; muñeca firme, no telegrafiar.

3) DISTANCIAS
- Larga: jab, control de ritmo.
- Media: combinaciones 1-2, 1-2-3, entradas/salidas.
- Corta: ganchos, uppercuts, clinch.
- Regla: entra con intención, sal con seguridad, no te quedes estático.

4) DEFENSA
- High guard: guantes arriba, codos protegen costillas.
- Slip: mover cabeza fuera de la línea del golpe.
- Parry: desviar el golpe con toque corto, volver a guardia.
- Pivot / paso atrás: salir del ángulo con pies, no solo echando el cuerpo.

5) COMBINACIONES TÍPICAS
- 1-2 (jab-cross): básica, puntúa y marca potencia.
- 1-2-3 (jab-cross-gancho): cambia plano, rompe defensa.
- Jab cuerpo + cross cabeza: alterna niveles.

6) ERRORES COMUNES
- Bajar la mano de guardia al lanzar el cross.
- No girar cadera/pies → pierde potencia.
- Mirar al suelo o cerrar los ojos.
- Quedarse en la línea tras golpear sin salida lateral.

7) SEGURIDAD
- Calentar y movilizar articulaciones antes de golpear fuerte.
- Vendas y guantes adecuados siempre.
- En sparring: control y protección obligatorios.
- Técnica antes que potencia; progresión gradual.
- Ante mareo, dolor anormal o lesión: detener y consultar profesional.
""".strip()


def safe(text: str) -> str:
    """Escapa triple-comillas para no romper el Modelfile."""
    return text.replace('"""', "'''")


def main() -> int:
    base_model  = os.getenv("OLLAMA_CHAT_MODEL",  "qwen2.5:3b-instruct")
    new_model   = os.getenv("OLLAMA_TUNED_MODEL", "ceacfp-tuned")
    mf_path     = Path("Modelfile.ceacfp")

    # ── 1. Cargar datos de la BD ──────────────────────────────────────────────
    db.init_db()

    pairs_rows  = db.fetch_all("SELECT question, answer FROM training_pairs ORDER BY id ASC")
    pairs       = [(str(r["question"]).strip(), str(r["answer"]).strip()) for r in pairs_rows]

    boxers_rows = db.fetch_all(
        "SELECT name, nickname, weight_class, stance, wins, losses, draws, city, notes "
        "FROM boxers ORDER BY name"
    )
    boxers_list = [dict(r) for r in boxers_rows]

    # ── 2. Cargar datos del SaaS ──────────────────────────────────────────────
    customers = saas_backend.execute("list_customers",  {"limit": 100})["customers"]
    invoices  = saas_backend.execute("search_invoices", {})["invoices"]
    kpi       = saas_backend.execute("kpi_sales",       {})["kpi"]
    products  = saas_backend.execute("top_products",    {"limit": 20})["products"]

    # Generar líneas de resumen legibles para el modelo
    invoices_pendientes = [i for i in invoices if i["status"] == "pendiente"]
    invoices_pagadas    = [i for i in invoices if i["status"] == "pagada"]

    # ── 3. Construir system prompt con TODO embebido ──────────────────────────
    system = f"""Eres el asistente oficial del Club de Boxeo fAIght. Respondes SIEMPRE en español claro y directo.
NUNCA inventas datos. NUNCA usas internet. Toda la información que necesitas está aquí abajo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DATOS DEL CLUB (responde directamente con estos datos)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[SOCIOS / CLIENTES]
{json.dumps(customers, ensure_ascii=False)}

[BOXEADORES EN BASE DE DATOS]
{json.dumps(boxers_list, ensure_ascii=False)}

[TODAS LAS FACTURAS/CUOTAS]
Total: {len(invoices)} | Pendientes: {len(invoices_pendientes)} | Pagadas: {len(invoices_pagadas)}
{json.dumps(invoices, ensure_ascii=False)}

[CUOTAS PENDIENTES]
{json.dumps(invoices_pendientes, ensure_ascii=False)}

[CUOTAS PAGADAS]
{json.dumps(invoices_pagadas, ensure_ascii=False)}

[KPIs DEL CLUB]
Total facturado : {kpi['total_invoiced_eur']} EUR
Cobrado         : {kpi['paid_eur']} EUR
Pendiente       : {kpi['pending_eur']} EUR
Nº facturas     : {kpi['invoice_count']}

[SERVICIOS MAS VENDIDOS]
{json.dumps(products, ensure_ascii=False)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CONOCIMIENTO TECNICO DE BOXEO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{BOXING_KNOWLEDGE}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ACCIONES MCP (devuelve JSON valido exacto)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cuando el usuario pida datos del club, devuelves UNO de estos JSON sin texto extra:
  Cuotas pendientes  → {{"action":"search_invoices","params":{{"status":"pendiente"}}}}
  Cuotas pagadas     → {{"action":"search_invoices","params":{{"status":"pagada"}}}}
  Todas las cuotas   → {{"action":"search_invoices","params":{{}}}}
  KPIs ingresos      → {{"action":"kpi_sales","params":{{}}}}
  Ranking servicios  → {{"action":"top_products","params":{{}}}}
  Lista socios       → {{"action":"list_customers","params":{{}}}}
"""

    # ── 4. Escribir Modelfile ─────────────────────────────────────────────────
    lines = [
        f"FROM {base_model}",
        "PARAMETER num_ctx 2048",
        "PARAMETER num_predict 200",
        "PARAMETER temperature 0.1",
        "PARAMETER repeat_penalty 1.1",
        f'SYSTEM """{safe(system)}"""',
        "",
    ]

    added = 0
    for q, a in pairs:
        if not q or not a:
            continue
        lines.append(f'MESSAGE user """{safe(q)}"""')
        lines.append(f'MESSAGE assistant """{safe(a)}"""')
        added += 1

    content = "\n".join(lines) + "\n"
    mf_path.write_text(content, encoding="utf-8")

    size_kb = len(content.encode("utf-8")) // 1024
    print(f"\nOK Modelfile generado: {mf_path}  ({size_kb} KB)")
    print(f"  Datos embebidos:")
    print(f"  · {len(customers)} socios SaaS")
    print(f"  · {len(boxers_list)} boxeadores (BD)")
    print(f"  · {len(invoices)} facturas  ({len(invoices_pendientes)} pendientes / {len(invoices_pagadas)} pagadas)")
    print(f"  · {len(products)} servicios")
    print(f"  · {added} pares Q&A")
    print(f"  · Conocimiento completo de boxeo\n")

    # ── 5. Crear modelo en Ollama ─────────────────────────────────────────────
    print(f"→ Creando modelo '{new_model}' (base: {base_model})...")
    print("  Esto puede tardar 1-3 minutos la primera vez.\n")

    cmd = ["ollama", "create", new_model, "-f", str(mf_path)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("ERROR: 'ollama' no encontrado en PATH. ¿Está instalado?")
        return 2
    except subprocess.CalledProcessError as e:
        print(f"ERROR: 'ollama create' falló: {e}")
        return 3

    # ── 6. Verificar que el modelo existe ────────────────────────────────────
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if new_model.split(":")[0] in result.stdout:
        print(f"\nLISTO: Modelo '{new_model}' listo y verificado en Ollama.")
    else:
        print(f"\nAVISO: El modelo no aparece en 'ollama list', revisa manualmente.")

    print("\n🔁 Reinicia la app (python app.py) para que use el nuevo modelo.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
