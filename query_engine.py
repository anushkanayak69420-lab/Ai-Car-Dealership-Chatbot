"""
Query engine for the Car Dealership AI Chatbot.

Rule-based NL parsing (works with zero API key) + an optional Claude-powered
NL-to-pandas mode for more flexible, open-ended questions.
"""

import re
import os
import pandas as pd

MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]


def load_data(path="data/car_sales.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def _find_month(text: str):
    text = text.lower()
    for i, m in enumerate(MONTHS, start=1):
        if re.search(rf"\b{m}\b", text) or re.search(rf"\b{m[:3]}\b", text):
            return i, m.capitalize()
    return None, None


def _find_in_column(text: str, df: pd.DataFrame, column: str):
    text = text.lower()
    for val in df[column].dropna().unique():
        if re.search(rf"\b{re.escape(str(val).lower())}\b", text):
            return val
    return None


def answer_question(question: str, df: pd.DataFrame):
    """Rule-based NL understanding. Returns (answer_text, chart_data or None)."""
    q = question.lower()
    month_num, month_name = _find_month(q)
    brand = _find_in_column(q, df, "brand")
    model = _find_in_column(q, df, "model")
    branch = _find_in_column(q, df, "branch")
    body_type = _find_in_column(q, df, "body_type")

    filtered = df.copy()
    if month_num:
        filtered = filtered[filtered["month_num"] == month_num]
    if brand:
        filtered = filtered[filtered["brand"] == brand]
    if branch:
        filtered = filtered[filtered["branch"] == branch]
    if body_type:
        filtered = filtered[filtered["body_type"].str.lower() == body_type.lower()]

    if filtered.empty:
        return "I couldn't find any sales matching that filter.", None

    # "which month does <model> sell the most / sell out" -> trend across months
    if model and ("month" in q or "when" in q) and not month_num:
        trend = (
            df[df["model"] == model]
            .groupby("month_num").size()
            .reindex(range(1, 13), fill_value=0)
        )
        trend.index = [MONTHS[i - 1].capitalize() for i in trend.index]
        best_month = trend.idxmax()
        answer = (
            f"The **{model}** sells the most in {best_month}, with "
            f"{int(trend.max())} units sold that month. Here's the trend across the year:"
        )
        return answer, trend

    # inventory aging / slow movers — the standout feature tied to real dealership pain points
    if any(kw in q for kw in ["inventory", "days", "stuck", "aging", "slow", "longest", "quickest", "fastest"]):
        avg_days = filtered.groupby("model")["days_in_inventory"].mean().sort_values(ascending=False)
        if "fastest" in q or "quickest" in q:
            fastest = avg_days.sort_values()
            answer = (
                f"**{fastest.index[0]}** sells the fastest — averaging "
                f"{fastest.iloc[0]:.0f} days in inventory before selling."
            )
            return answer, fastest.head(6)
        if "average" in q or "avg" in q:
            answer = (
                f"**{avg_days.index[0]}** takes the longest to sell on average — "
                f"{avg_days.iloc[0]:.0f} days in inventory. "
                f"Fastest mover: **{avg_days.index[-1]}** at {avg_days.iloc[-1]:.0f} days."
            )
            return answer, avg_days.head(6)
        answer = (
            f"Slowest-moving model: **{avg_days.index[0]}**, averaging "
            f"{avg_days.iloc[0]:.0f} days on the lot before selling. "
            f"Consider a discount push on this model."
        )
        return answer, avg_days.head(6)

    # revenue / price questions
    if "revenue" in q or "price" in q or "expensive" in q or "worth" in q or "earn" in q:
        if "expensive" in q or "highest price" in q:
            top_row = filtered.sort_values("price", ascending=False).iloc[0]
            answer = f"The most expensive car sold was a **{top_row['model']}** for ₹{top_row['price']:,.0f}."
            return answer, None
        by_model = filtered.groupby("model")["price"].sum().sort_values(ascending=False)
        answer = f"**{by_model.index[0]}** generated the highest revenue: ₹{by_model.iloc[0]:,.0f}."
        return answer, by_model.head(5)

    # brand comparison
    if "brand" in q and not brand:
        by_brand = filtered.groupby("brand").size().sort_values(ascending=False)
        answer = f"Top brand: **{by_brand.index[0]}** with {int(by_brand.iloc[0])} units sold."
        return answer, by_brand

    # branch/showroom comparison
    if ("branch" in q or "showroom" in q or "outlet" in q) and not branch:
        by_branch = filtered.groupby("branch").size().sort_values(ascending=False)
        answer = f"Top-performing branch: **{by_branch.index[0]}** with {int(by_branch.iloc[0])} cars sold."
        return answer, by_branch

    # body type comparison (SUV vs sedan vs hatchback)
    if ("body" in q or "suv" in q or "sedan" in q or "hatchback" in q) and not body_type:
        by_body = filtered.groupby("body_type").size().sort_values(ascending=False)
        answer = f"Most popular body type: **{by_body.index[0]}** with {int(by_body.iloc[0])} units sold."
        return answer, by_body

    # least sold
    if "least" in q or "worst" in q or "lowest" in q:
        by_model = filtered.groupby("model").size().sort_values()
        answer = f"The slowest-selling model was **{by_model.index[0]}** with only {int(by_model.iloc[0])} units sold."
        return answer, by_model.head(5)

    # default / most common intent: "what sold out / sold the most in <month/brand/branch>"
    by_model = filtered.groupby("model").size().sort_values(ascending=False)
    top = by_model.index[0]
    scope = []
    if month_name:
        scope.append(month_name)
    if brand:
        scope.append(brand)
    if branch:
        scope.append(branch)
    scope_text = " / ".join(scope) if scope else "overall"
    answer = f"In {scope_text}, the **{top}** sold the most, with {int(by_model.iloc[0])} units sold."
    if len(by_model) > 1:
        answer += f" Runner-up: {by_model.index[1]} ({int(by_model.iloc[1])} units)."
    return answer, by_model.head(6)


def answer_with_llm(question: str, df: pd.DataFrame, api_key: str = None):
    """Optional LLM-powered NL-to-pandas path using the Claude API."""
    try:
        import anthropic
    except ImportError:
        return answer_question(question, df)

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return answer_question(question, df)

    client = anthropic.Anthropic(api_key=key)

    schema_desc = (
        "Columns: sale_id, date, month, month_num (1-12), brand, model, body_type, "
        "branch, color, price (INR), days_in_inventory. "
        f"Brands: {', '.join(df['brand'].unique())}. "
        f"Models: {', '.join(df['model'].unique())}. "
        f"Branches: {', '.join(df['branch'].unique())}."
    )

    prompt = f"""You are a data analyst for a car dealership. Given this pandas DataFrame `df` with schema:
{schema_desc}

Write ONE line of pandas code (assign the result to a variable called `result`)
that answers this question: "{question}"

Only output the code, nothing else. No explanations, no markdown fences."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        code = response.content[0].text.strip()

        if any(bad in code for bad in ["import", "__", "open(", "exec(", "eval(", "os.", "sys."]):
            raise ValueError("Unsafe code generated")

        local_vars = {"df": df, "pd": pd}
        exec(code, {}, local_vars)
        result = local_vars.get("result")

        explain_prompt = f"""The question was: "{question}"
The computed pandas result is:
{result}

Explain this result in one clear, friendly sentence for a dealership manager."""
        explain_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[{"role": "user", "content": explain_prompt}],
        )
        explanation = explain_response.content[0].text.strip()
        return explanation, result
    except Exception:
        return answer_question(question, df)
