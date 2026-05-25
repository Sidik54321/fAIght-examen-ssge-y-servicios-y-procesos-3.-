from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Customer:
    id: str
    name: str
    segment: str
    city: str


@dataclass(frozen=True)
class Invoice:
    id: str
    customer_id: str
    invoice_date: date
    total_eur: float
    status: str


CUSTOMERS: List[Customer] = [
    Customer("C001", "Leo \"El Zurdo\" Martín", "Amateur", "Barcelona"),
    Customer("C002", "Amina \"La Rápida\" Benali", "Semi-pro", "Bilbao"),
    Customer("C003", "Carlos \"El Muro\" Vega", "Pro", "Madrid"),
    Customer("C004", "Sofía \"La Técnica\" Ruiz", "Amateur", "Valencia"),
    Customer("C005", "Hugo \"El Pitbull\" Santos", "Amateur", "Málaga"),
]

INVOICES: List[Invoice] = [
    Invoice("F-2001", "C001", date(2026, 4, 2), 45.00, "pagada"),
    Invoice("F-2002", "C003", date(2026, 4, 10), 60.00, "pendiente"),
    Invoice("F-2003", "C002", date(2026, 5, 1), 55.00, "pagada"),
    Invoice("F-2004", "C004", date(2026, 5, 8), 45.00, "pendiente"),
    Invoice("F-2005", "C005", date(2026, 5, 15), 45.00, "pagada"),
]

PRODUCT_SALES_EUR: Dict[str, float] = {
    "Cuota mensual": 1260.0,
    "Clases 1:1": 980.0,
    "Sparring guiado": 620.0,
    "Plan de fuerza": 540.0,
}


def list_actions() -> List[Dict[str, Any]]:
    return [
        {
            "action": "list_customers",
            "description": "Lista boxeadores/socios del club",
            "params": {"limit": "int opcional"},
        },
        {
            "action": "search_invoices",
            "description": "Busca cuotas/facturas por estado y/o socio",
            "params": {"status": "pagada|pendiente opcional", "customer_id": "str opcional"},
        },
        {
            "action": "kpi_sales",
            "description": "KPIs simples del club (ingresos por cuotas)",
            "params": {},
        },
        {
            "action": "top_products",
            "description": "Ranking de servicios (clases, planes, etc.) por ventas",
            "params": {"limit": "int opcional"},
        },
    ]


def _customer_to_dict(c: Customer) -> Dict[str, Any]:
    return {"id": c.id, "name": c.name, "segment": c.segment, "city": c.city}


def _invoice_to_dict(i: Invoice) -> Dict[str, Any]:
    return {
        "id": i.id,
        "customer_id": i.customer_id,
        "date": i.invoice_date.isoformat(),
        "total_eur": i.total_eur,
        "status": i.status,
    }


def execute(action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = params or {}
    if action == "list_customers":
        limit = int(params.get("limit") or 50)
        return {"customers": [_customer_to_dict(c) for c in CUSTOMERS[:limit]]}

    if action == "search_invoices":
        status = params.get("status")
        customer_id = params.get("customer_id")
        out = INVOICES
        if status:
            out = [i for i in out if i.status == status]
        if customer_id:
            out = [i for i in out if i.customer_id == customer_id]
        return {"invoices": [_invoice_to_dict(i) for i in out]}

    if action == "kpi_sales":
        total = sum(i.total_eur for i in INVOICES)
        pending = sum(i.total_eur for i in INVOICES if i.status == "pendiente")
        paid = total - pending
        return {
            "kpi": {
                "total_invoiced_eur": round(total, 2),
                "paid_eur": round(paid, 2),
                "pending_eur": round(pending, 2),
                "invoice_count": len(INVOICES),
            }
        }

    if action == "top_products":
        limit = int(params.get("limit") or 10)
        items = sorted(PRODUCT_SALES_EUR.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "products": [
                {"product": name, "sales_eur": value} for name, value in items[:limit]
            ]
        }

    raise ValueError(f"Acción desconocida: {action}")
