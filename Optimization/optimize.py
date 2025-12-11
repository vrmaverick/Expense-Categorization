import pandas as pd
def optimize_answer(df_context, question: str) -> str:
    """
    Given a dataframe with columns like ['date','expense','category','is_forecast', ...]
    and a user question, return a textual answer.

    Replace the body with your actual optimization / LLM call.
    """
    if not question.strip():
        return "Please enter a question about your expenses or forecast."

    # Very simple placeholder logic: summarize last month + hint.
    try:
        df = df_context.copy()
        df["date"] = pd.to_datetime(df["date"])
        last_date = df["date"].max()
        last_month_mask = df["date"] >= (last_date - pd.Timedelta(days=30))
        recent = df[last_month_mask]

        total_spend = recent["expense"].sum() if "expense" in recent.columns else float("nan")
        top_cat = (
            recent.groupby("category")["expense"].sum().sort_values(ascending=False).index[0]
            if {"category", "expense"}.issubset(recent.columns) and not recent.empty
            else None
        )

        answer = []
        answer.append(f"In the last 30 days (up to {last_date.date()}), your total spend is about {total_spend:,.2f}.")
        if top_cat:
            answer.append(f"Your highest-spend category is **{top_cat}**.")
        answer.append("You can ask targeted questions like: 'How can I reduce my food expenses?' or "
                      "'Is my transport spend increasing compared to last month?'")
        return "\n\n".join(answer)
    except Exception as e:
        return f"Could not analyze the data for this question. Details: {e}"
