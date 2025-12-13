import pandas as pd
from google import genai
from . import API_Config as a

# Configure once at import
GEMINI_API_KEY = a.Config_API()
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Create a single client
client = genai.Client(api_key=GEMINI_API_KEY)


def _build_expense_summary(df: pd.DataFrame, max_rows: int = 30) -> str:
    """Turn recent rows into a compact text summary to send to Gemini."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    recent = df.tail(max_rows)

    lines = []
    for _, row in recent.iterrows():
        date_str = row["date"].date().isoformat()
        amt = float(row.get("expense", 0.0))
        cat = str(row.get("category", "Unknown"))
        is_fc = bool(row.get("is_forecast", False))
        tag = "forecast" if is_fc else "actual"
        lines.append(f"{date_str}: {amt:.2f} ({cat}, {tag})")

    return "\n".join(lines)

import re

def _to_plain_text(markdown_text: str) -> str:
    """
    Convert a bullet-heavy markdown answer from Gemini to plain text.
    - Removes leading *, -, **, ### etc.
    - Flattens multiple spaces and blank lines.
    """
    lines = markdown_text.splitlines()
    cleaned = []

    for line in lines:
        l = line.strip()
        if not l:
            cleaned.append("")          # keep paragraph breaks
            continue

        # remove common bullet markers at start
        l = re.sub(r"^[-*]\s+", "", l)  # leading "- " or "* "
        # remove surrounding bold **text**
        l = re.sub(r"\*\*(.*?)\*\*", r"\1", l)
        # remove leftover markdown headers
        l = re.sub(r"^#+\s*", "", l)

        cleaned.append(l)

    # collapse more than 2 blank lines
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def optimize_answer(df_context, question: str) -> str:
    """
    Use Gemini to answer optimization questions over the combined
    history+forecast dataframe.
    """
    if not question.strip():
        return "Please enter a question about your expenses or forecast."

    try:
        df = df_context.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Basic stats to prime the model
        last_date = df["date"].max()
        last_30 = df["date"] >= (last_date - pd.Timedelta(days=30))
        recent = df[last_30]

        total_spend = (
            recent["expense"].sum() if "expense" in recent.columns else float("nan")
        )
        top_cat = (
            recent.groupby("category")["expense"]
            .sum()
            .sort_values(ascending=False)
            .index[0]
            if {"category", "expense"}.issubset(recent.columns) and not recent.empty
            else None
        )

        summary_text = _build_expense_summary(df, max_rows=60)

        system_prompt = (
            "You are a personal finance optimization assistant.\n"
            "You receive a recent history of expenses with categories and a forecast flag.\n"
            "Use the data to give concrete, actionable suggestions (not generic advice).\n"
            "Keep answers short, No bullets no bold words(alphabets)"
        )

        context_stats = [
            f"Last date in data: {last_date.date().isoformat()}",
            f"Total spend in last 30 days: {total_spend:.2f}",
        ]
        if top_cat:
            context_stats.append(f"Top category in last 30 days: {top_cat}")
        stats_block = "\n".join(context_stats)

        prompt = f"""{system_prompt}

=== Aggregate stats ===
{stats_block}

=== Recent expenses (up to ~60 rows) ===
date: amount (category, actual/forecast)
{summary_text}

=== User question ===
{question}

Now answer the question using ONLY the patterns and numbers implied by this data.
"""

        # Call Gemini via the client
        resp = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )

        # google-genai returns resp.text for text output
        # _to_plain_text(raw)
        # return resp.text.strip() if getattr(resp, "text", None) else "No answer generated."
        return _to_plain_text(resp.text) if getattr(resp, "text", None) else "No answer generated."

    except Exception as e:
        return f"Could not analyze the data or call Gemini. Details: {e}"
