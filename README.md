# 🚗 AI Car Dealership Inventory Chatbot

An AI-powered chatbot that answers natural language questions about car sales and inventory — *"What sold the most in October?"*, *"Which model takes the longest to sell?"* — with real, data-backed answers and charts.

**Inspired by a real car dealership internship**, where inventory management and slow-moving stock were everyday problems. This project turns that experience into a working analytics tool.

## Live Demo
_(Add your Streamlit Cloud link here once deployed)_

## What makes this stand out

Most beginner projects stop at "here's a chart." This one:
- Answers **open-ended natural language questions**, not just a fixed dashboard
- Flags **slow-moving inventory** (models sitting too long before selling) — a genuine business problem, not a toy metric
- Has a **rule-based mode that works with zero API cost**, plus an **optional LLM mode** (Claude API) that does true NL-to-pandas query generation
- Fails gracefully — if the LLM generates bad code or the API is unavailable, it safely falls back instead of crashing

## How it works

1. **Data layer** — synthetic but realistic dealership data (brand, model, body type, branch, price, days in inventory) with real seasonal patterns (festive season SUV demand, wedding-season MUV demand, year-end hatchback discounts)
2. **NL understanding** — extracts intent (model/brand/branch/month) from the question and filters + aggregates the data with pandas
3. **Optional LLM mode** — sends the question + schema to Claude, which writes and the app safely executes a pandas query, then explains the result in plain English
4. **KPIs + alerts** — a top-line metrics row and a "slow-moving stock" panel give instant visual insight before you even ask a question
5. **Interface** — Streamlit chat UI, fully demoable live

## Setup

```bash
pip install -r requirements.txt
python generate_data.py      # creates data/car_sales.csv
streamlit run app.py
```

Works immediately in rule-based mode. For LLM mode:
```bash
export ANTHROPIC_API_KEY="your-key-here"
streamlit run app.py   # then toggle "Use LLM mode" in the sidebar
```

## Project structure

```
car-inventory-chatbot/
├── generate_data.py    # synthetic dealership dataset generator
├── query_engine.py      # NL parsing + rule-based and LLM query logic
├── app.py                # Streamlit chat interface + KPI dashboard
├── requirements.txt
└── data/
    └── car_sales.csv     # generated dataset
```

## Example questions

- What cars sold the most in October?
- When does the Creta sell the most?
- Which brand sold the most?
- Which branch sold the most cars?
- Which model takes the longest to sell?
- Which model sells fastest?
- What was the most expensive car sold?
- What is the least sold model?

## Skills demonstrated

- Data cleaning, filtering & aggregation with Pandas
- Domain-driven feature design (days-in-inventory, seasonal demand) grounded in real business experience
- LLM API integration for NL-to-query generation
- Defensive coding — safe code execution, graceful fallback
- Dashboard design (KPIs + alerts + chat, not just a notebook)
- End-to-end deployment

## Possible extensions (great interview talking points)

- Connect to a real dealership CRM export instead of synthetic data
- Add demand forecasting (e.g., predict next month's top-selling model)
- Add a "reorder recommendation" feature based on sell-through rate
- Multi-turn conversation memory (follow-up questions like "and last month?")

## Resume bullet

> Built an AI-powered inventory analytics chatbot for a car dealership use case (drawn from internship experience), combining NL-to-pandas querying (Claude API) with inventory-aging analytics to flag slow-moving stock — deployed live via Streamlit.
