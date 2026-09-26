"""
AI Car Dealership Inventory Chatbot
Ask plain-English questions about car sales & inventory and get real,
data-backed answers with auto-generated charts.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
from query_engine import load_data, answer_question, answer_with_llm

st.set_page_config(page_title="AI Car Dealership Chatbot", page_icon="🚗", layout="wide")

st.title("🚗 AI Car Dealership Inventory Chatbot")
st.caption(
    "Ask questions like *'What cars sold the most in October?'* or "
    "*'Which model takes the longest to sell?'*"
)

DATA_PATH = "data/car_sales.csv"
if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Run `python generate_data.py` first.")
    st.stop()

df = load_data(DATA_PATH)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    use_llm = st.toggle(
        "Use LLM mode (Claude API)",
        value=False,
        help="Requires ANTHROPIC_API_KEY in the environment. Off by default so the app works with zero setup.",
    )
    if use_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("No ANTHROPIC_API_KEY found. Falling back to rule-based mode.")

    st.divider()
    st.subheader("About this project")
    st.write(
        "Inspired by a real car dealership internship, this chatbot answers "
        "natural language questions over sales & inventory data — including "
        "which models are slow-moving stock, a real problem dealerships face."
    )
    st.divider()
    st.subheader("Try asking:")
    st.markdown(
        "- What cars sold the most in October?\n"
        "- When does the Creta sell the most?\n"
        "- Which brand sold the most?\n"
        "- Which branch sold the most cars?\n"
        "- Which model takes the longest to sell?\n"
        "- Which model sells fastest?\n"
        "- What was the most expensive car sold?"
    )

# --- Top-line KPI row (instant visual impact for anyone opening the app) ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cars Sold", f"{len(df):,}")
col2.metric("Total Revenue", f"₹{df['price'].sum():,.0f}")
col3.metric("Avg Days in Inventory", f"{df['days_in_inventory'].mean():.0f} days")
top_model = df.groupby("model").size().idxmax()
col4.metric("Best-Selling Model", top_model)

# --- Slow-moving stock alert panel (the standout feature) ---
with st.expander("⚠️ Slow-Moving Stock Alert (models aging longest in inventory)", expanded=False):
    slow = df.groupby("model")["days_in_inventory"].mean().sort_values(ascending=False).head(5)
    st.bar_chart(slow)
    st.caption("Models here sit longest before selling — candidates for discounts or promotions.")

with st.expander("Preview raw data"):
    st.dataframe(df.head(20), use_container_width=True)

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart") is not None:
            st.bar_chart(msg["chart"])

question = st.chat_input("Ask a question about car sales & inventory...")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "chart": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing dealership data..."):
            if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
                answer, chart_data = answer_with_llm(question, df)
            else:
                answer, chart_data = answer_question(question, df)

        st.markdown(answer)
        if chart_data is not None:
            try:
                st.bar_chart(chart_data)
            except Exception:
                pass

    st.session_state.messages.append({"role": "assistant", "content": answer, "chart": chart_data})
