import os
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import Any, Dict, List, Literal, Optional
from dotenv import load_dotenv
from google import genai  # pip install google-genai

load_dotenv()
# -----------------------------
# Types
# -----------------------------
@dataclass
class TransactionRecord:
    id: str
    created_at: str
    amount: float
    vendor: str
    category: str


# -----------------------------
# Helpers
# -----------------------------
def _parse_iso(dt: str) -> datetime:
    # Handles "...Z" UTC timestamps too
    if dt.endswith("Z"):
        dt = dt[:-1] + "+00:00"
    return datetime.fromisoformat(dt)


def run_eda(transactions: List[TransactionRecord], forecast: List[float]) -> Dict[str, Any]:
    tx_sorted = sorted(transactions, key=lambda t: _parse_iso(t.created_at).timestamp())

    # Income = negative values (reported positive)
    total_income = sum(-t.amount for t in tx_sorted if t.amount < 0)
    # Expense = positive values
    total_expense = sum(t.amount for t in tx_sorted if t.amount > 0)
    # Net = income - expense
    total_net = total_income - total_expense

    # ----- Daily net series (income positive, expense negative) -----
    date_map: Dict[str, float] = {}
    for t in tx_sorted:
        key = t.created_at.split("T")[0]
        delta = -t.amount  # income (+), expense (−)
        date_map[key] = date_map.get(key, 0.0) + delta

    daily = [{"date": d, "value": v} for d, v in sorted(date_map.items(), key=lambda kv: _parse_iso(kv[0]).timestamp())]
    daily_vals = [d["value"] for d in daily]

    # ----- Volatility + trend on net daily values -----
    volatility = 0.0
    slope = 0.0
    trend: Literal["increasing", "decreasing", "flat"] = "flat"

    if len(daily_vals) > 1:
        mean = sum(daily_vals) / len(daily_vals)
        variance = sum((v - mean) ** 2 for v in daily_vals) / len(daily_vals)
        volatility = sqrt(variance)

        xs = list(range(len(daily_vals)))
        ys = daily_vals
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)

        num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(len(xs)))
        den = sum((xs[i] - x_mean) ** 2 for i in range(len(xs)))
        slope = 0.0 if den == 0 else num / den

        if slope > 0:
            trend = "increasing"
        elif slope < 0:
            trend = "decreasing"

    # ----- Category stats (net, using same delta sign) -----
    cat_map: Dict[str, Dict[str, float]] = {}
    for t in tx_sorted:
        entry = cat_map.get(t.category, {"sum": 0.0, "count": 0, "max": float("-inf")})
        delta = -t.amount
        entry["sum"] += delta
        entry["count"] += 1
        entry["max"] = max(entry["max"], delta)
        cat_map[t.category] = entry

    category_stats = [
        {"category": cat, "sum": v["sum"], "count": v["count"], "max": v["max"]}
        for cat, v in cat_map.items()
    ]

    # ----- Forecast summary (unchanged) -----
    forecast_mean = 0.0
    forecast_trend: Literal["upward", "downward", "flat"] = "flat"
    if forecast:
        forecast_mean = sum(forecast) / len(forecast)
        if forecast[-1] > forecast[0]:
            forecast_trend = "upward"
        elif forecast[-1] < forecast[0]:
            forecast_trend = "downward"

    return {
        "totals": {
            "totalNet": total_net,
            "totalIncome": total_income,
            "totalExpense": total_expense,
            "avgDaily": (total_net / len(daily_vals)) if daily_vals else 0.0,
        },
        "volatility": volatility,
        "trend": {"slope": slope, "direction": trend},
        "daily": daily,
        "categoryStats": category_stats,
        "forecastSummary": {"mean": forecast_mean, "trend": forecast_trend},
    }


def build_prompt(eda: Dict[str, Any], question: str) -> str:
    lines: List[str] = []
    lines.append("You are a financial analysis assistant.")
    lines.append("Use the EDA below to answer the user question with clear pros and cons and actionable tips.")
    lines.append("")
    lines.append("=== TOTALS ===")
    lines.append(f"Total net: {eda['totals']['totalNet']}")
    lines.append(f"Income: {eda['totals']['totalIncome']}")
    lines.append(f"Expense: {eda['totals']['totalExpense']}")
    lines.append(f"Average daily net: {eda['totals']['avgDaily']}")
    lines.append("")
    lines.append("=== VOLATILITY & TREND ===")
    lines.append(f"Volatility (std dev): {eda['volatility']}")
    lines.append(f"Trend: {eda['trend']['direction']} (slope {eda['trend']['slope']})")
    lines.append("")
    lines.append("=== TOP CATEGORIES ===")
    sorted_cats = sorted(eda["categoryStats"], key=lambda c: abs(c["sum"]), reverse=True)
    for c in sorted_cats[:5]:
        lines.append(f"- {c['category']}: sum={c['sum']}, count={c['count']}, max={c['max']}")
    lines.append("")
    lines.append("=== FORECAST SUMMARY ===")
    lines.append(f"Forecast mean: {eda['forecastSummary']['mean']}")
    lines.append(f"Forecast trend: {eda['forecastSummary']['trend']}")
    lines.append("")
    lines.append("=== USER QUESTION ===")
    lines.append(question)
    lines.append("")
    lines.append(
        "Answer in 3–7 sentences. Explain good and bad patterns, highlight risks, and give 1–3 concrete actionable "
        "suggestions. Do not repeat raw numbers excessively."
    )
    return "\n".join(lines)


# -----------------------------
# One-shot runner
# -----------------------------
def analyze_once(
    transactions: List[TransactionRecord],
    forecast: Optional[List[float]],
    question: str,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) or pass api_key=...")

    if not transactions:
        raise ValueError("transactions must be non-empty")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is required")

    eda = run_eda(transactions, forecast or [])
    prompt = build_prompt(eda, question)

    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model=model, contents=prompt)
    answer = getattr(resp, "text", "") or ""

    return {"eda": eda, "answer": answer}


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    tx = transactions = [
        TransactionRecord(id="t1",  created_at="2025-12-11T00:00:00Z", amount=12.00,   vendor="McDonald's",               category="FOOD_AND_DRINK"),
        TransactionRecord(id="t2",  created_at="2025-12-11T00:00:00Z", amount=4.33,    vendor="Starbucks",                category="FOOD_AND_DRINK"),
        TransactionRecord(id="t3",  created_at="2025-12-10T00:00:00Z", amount=89.40,   vendor="FUN",                      category="ENTERTAINMENT"),
        TransactionRecord(id="t4",  created_at="2025-12-09T00:00:00Z", amount=-4.22,   vendor="INTRST PYMNT",             category="INCOME"),
        TransactionRecord(id="t5",  created_at="2025-11-29T00:00:00Z", amount=500.00,  vendor="United Airlines",          category="TRAVEL"),
        TransactionRecord(id="t6",  created_at="2025-11-27T00:00:00Z", amount=6.33,    vendor="Uber",                     category="TRANSPORTATION"),
        TransactionRecord(id="t7",  created_at="2025-11-24T00:00:00Z", amount=500.00,  vendor="Tectra Inc",               category="ENTERTAINMENT"),
        TransactionRecord(id="t8",  created_at="2025-11-23T00:00:00Z", amount=2078.50, vendor="AUTOMATIC PAYMENT - THANK",category="TRANSFER_OUT"),
        TransactionRecord(id="t9",  created_at="2025-11-23T00:00:00Z", amount=500.00,  vendor="KFC",                      category="FOOD_AND_DRINK"),
        TransactionRecord(id="t10", created_at="2025-11-23T00:00:00Z", amount=500.00,  vendor="Madison Bicycle Shop",     category="GENERAL_MERCHANDISE"),
        TransactionRecord(id="t11", created_at="2025-11-14T00:00:00Z", amount=25.00,   vendor="CREDIT CARD 3333 PAYMENT *//", category="LOAN_PAYMENTS"),
        TransactionRecord(id="t12", created_at="2025-11-14T00:00:00Z", amount=5.40,    vendor="Uber",                     category="TRANSPORTATION"),
        TransactionRecord(id="t13", created_at="2025-11-13T00:00:00Z", amount=1000.00, vendor="CD DEPOSIT .INITIAL.",     category="TRANSFER_OUT"),
        TransactionRecord(id="t14", created_at="2025-11-12T00:00:00Z", amount=78.50,   vendor="Touchstone Climbing",      category="PERSONAL_CARE"),
        TransactionRecord(id="t15", created_at="2025-11-12T00:00:00Z", amount=-500.00, vendor="United Airlines",          category="TRAVEL"),
        TransactionRecord(id="t16", created_at="2025-11-11T00:00:00Z", amount=12.00,   vendor="McDonald's",               category="FOOD_AND_DRINK"),
        TransactionRecord(id="t17", created_at="2025-11-11T00:00:00Z", amount=4.33,    vendor="Starbucks",                category="FOOD_AND_DRINK"),
        ]
    result = analyze_once(
        transactions=tx,
        forecast = [
        185.80, 201.47, 197.10, 173.21, 170.06, 182.23, 197.48, 192.60, 206.73, 201.85,
        188.24, 173.91, 187.90, 202.72, 196.35, 185.75, 184.50, 185.79, 201.48, 212.63,
        196.60, 175.44, 164.77, 186.32, 196.77, 192.71, 183.30, 177.75, 203.54, 208.33
        ],
        question="How am I doing and what should I improve this month?",
    )
    print(result["answer"])
