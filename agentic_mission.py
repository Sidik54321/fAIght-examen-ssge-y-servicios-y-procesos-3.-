import json
import os
from typing import Any, Dict, List

import rag_engine
import saas_backend
from ollama_api import force_json


def _tool_spec() -> Dict[str, Any]:
    return {
        "tools": [
            {
                "name": "saas",
                "description": "Consulta el MiniSaaS del club de boxeo",
                "args": {
                    "action": "list_customers|search_invoices|kpi_sales|top_products",
                    "params": "objeto",
                },
            },
            {
                "name": "rag_query",
                "description": "Consulta el RAG (ChromaDB) y devuelve chunks",
                "args": {"collection": "str", "question": "str", "k": "int"},
            },
            {"name": "finish", "description": "Finaliza con el informe final", "args": {"report": "str"}},
        ]
    }


def run_agent(mission: str, *, max_steps: int = 8) -> str:
    state: Dict[str, Any] = {"mission": mission, "facts": [], "last_result": None}
    system = (
        "Eres una IA agentica y autónoma para un club de boxeo. Tu objetivo es completar la misión usando herramientas.\n"
        "Devuelve únicamente JSON válido con el siguiente esquema:\n"
        '{"tool":"saas|rag_query|finish","args":{...},"note":"muy corto"}\n'
        "Reglas:\n"
        "- Usa saas para datos numéricos/KPIs.\n"
        "- Usa rag_query para definiciones/teoría si hay colección disponible.\n"
        '- En saas, "action" debe ser una de: list_customers, search_invoices, kpi_sales, top_products.\n'
        '- En search_invoices, "status" debe ser "pendiente" o "pagada".\n'
        "- Cuando tengas suficiente, usa finish con un informe final."
    )
    tool_spec = _tool_spec()

    for _ in range(max_steps):
        user = json.dumps({"state": state, **tool_spec}, ensure_ascii=False)
        action = force_json(system, user)
        tool = action.get("tool")
        args = action.get("args") if isinstance(action.get("args"), dict) else {}

        if tool == "saas":
            allowed_actions = {"list_customers", "search_invoices", "kpi_sales", "top_products"}
            action_name = str(args.get("action") or "").strip()
            params_in = args.get("params")
            params: Dict[str, Any] = params_in if isinstance(params_in, dict) else {}

            mission_lc = (mission or "").lower()
            if action_name not in allowed_actions:
                if "cuota" in mission_lc or "factur" in mission_lc or "invoice" in mission_lc or "pendient" in mission_lc:
                    action_name = "search_invoices"
                elif "kpi" in mission_lc or "ingres" in mission_lc or "ventas" in mission_lc:
                    action_name = "kpi_sales"
                elif "ranking" in mission_lc or "top" in mission_lc or "producto" in mission_lc or "servicio" in mission_lc:
                    action_name = "top_products"
                else:
                    action_name = "list_customers"

            if action_name == "search_invoices":
                status = params.get("status")
                if isinstance(status, str) and status.strip():
                    s = status.strip().lower()
                    if s.startswith("pend") or s in {"unpaid", "pending"}:
                        params["status"] = "pendiente"
                    elif s.startswith("pag") or s in {"paid", "pay"}:
                        params["status"] = "pagada"
                customer_id = params.get("customer_id")
                if customer_id is not None and not isinstance(customer_id, str):
                    params.pop("customer_id", None)

            try:
                result = saas_backend.execute(action_name, params)
            except Exception as e:
                result = {"error": str(e)}
            state["last_result"] = {"tool": "saas", "args": {"action": action_name, "params": params}, "result": result}
            state["facts"].append(state["last_result"])
            continue

        if tool == "rag_query":
            collection = (args.get("collection") or os.getenv("RAG_COLLECTION", "boxeo_apuntes")).strip()
            question = (args.get("question") or "").strip() or mission
            k = int(args.get("k") or 4)
            try:
                chunks = rag_engine.query(collection, question, k=k)
                result = {"chunks": chunks}
            except Exception as e:
                result = {"error": str(e)}
            state["last_result"] = {"tool": "rag_query", "args": {"collection": collection, "question": question, "k": k}, "result": result}
            state["facts"].append(state["last_result"])
            continue

        if tool == "finish":
            report = (args.get("report") or "").strip()
            if report:
                return report
            return "Misión finalizada, pero el modelo no devolvió report."

        state["last_result"] = {"tool": "invalid", "result": action}

    return "No se pudo completar la misión dentro del máximo de pasos."


def main() -> int:
    mission = os.getenv(
        "MISSION",
        "Crea un informe corto del club de boxeo con: KPIs de ingresos, cuotas pendientes y una explicación breve de jab y guardia si existe contexto en el RAG.",
    )
    report = run_agent(mission)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
