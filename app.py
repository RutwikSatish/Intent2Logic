import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from ucimlrepo import fetch_ucirepo

st.set_page_config(
    page_title="NeuroSymbolic Decision Translator",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NeuroSymbolic Decision Translator for Supply Chains")
st.caption("Natural language planning intent → structured logic → explainable recommendations")

# =============================================================================
# APP OVERVIEW
# =============================================================================
with st.expander("What this application does", expanded=True):
    st.markdown(
        """
### What this app does
This application demonstrates a **neurosymbolic supply chain decision-support system**.

It takes a planner-style instruction such as:

- *Reduce stockouts for high variability SKUs without increasing inventory too much*
- *Focus on top revenue items but control overstock risk*

Then it:
1. Interprets the request using **Groq LLM**
2. Converts that request into a structured planning intent
3. Applies **symbolic business rules**
4. Returns **explainable recommendations**

### What problem it solves
In real businesses, people describe planning goals in natural language, but systems require explicit rules and structured logic.

This creates a gap between:
- **what humans mean**
- **what systems can execute**

This demo bridges that gap.

### Demo dataset
This app uses the **UCI Online Retail dataset**, a public and credible retail transaction dataset frequently used for analytics and demand-pattern work.

### Important limitation
This is a **decision-support demo**, not a full ERP or MRP engine.  
The dataset does not include live on-hand inventory, supplier lead times, or formal service level history, so the recommendations are based on:
- demand variability
- revenue concentration
- velocity
- demand trend
- recency of sales
        """
    )

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(show_spinner=True)
def load_demo_data() -> pd.DataFrame:
    dataset = fetch_ucirepo(id=352)
    df = dataset.data.features.copy()

    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")

    df = df.dropna(subset=["InvoiceDate", "StockCode", "Description", "Quantity", "UnitPrice"])
    df["Description"] = df["Description"].astype(str).str.strip()
    df["Country"] = df["Country"].fillna("Unknown").astype(str).str.strip()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    df["is_cancellation"] = df["InvoiceNo"].str.startswith("C")
    df["line_revenue"] = df["Quantity"] * df["UnitPrice"]

    sales_df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    sales_df["month"] = sales_df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    return sales_df


@st.cache_data(show_spinner=False)
def build_sku_metrics(sales_df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        sales_df.groupby(["StockCode", "Description", "month"], as_index=False)
        .agg(
            qty=("Quantity", "sum"),
            revenue=("line_revenue", "sum"),
            orders=("InvoiceNo", "nunique"),
        )
    )

    sku_base = (
        sales_df.groupby(["StockCode", "Description"], as_index=False)
        .agg(
            total_qty=("Quantity", "sum"),
            total_revenue=("line_revenue", "sum"),
            avg_unit_price=("UnitPrice", "mean"),
            order_lines=("InvoiceNo", "count"),
            orders=("InvoiceNo", "nunique"),
            customers=("CustomerID", pd.Series.nunique),
            countries=("Country", pd.Series.nunique),
            first_sale=("InvoiceDate", "min"),
            last_sale=("InvoiceDate", "max"),
        )
    )

    monthly_stats = (
        monthly.groupby(["StockCode", "Description"], as_index=False)
        .agg(
            avg_monthly_qty=("qty", "mean"),
            std_monthly_qty=("qty", "std"),
            avg_monthly_revenue=("revenue", "mean"),
            active_months=("month", "nunique"),
        )
    )

    sku = sku_base.merge(monthly_stats, on=["StockCode", "Description"], how="left")
    sku["std_monthly_qty"] = sku["std_monthly_qty"].fillna(0)
    sku["avg_monthly_qty"] = sku["avg_monthly_qty"].fillna(0)

    sku["demand_cv"] = np.where(
        sku["avg_monthly_qty"] > 0,
        sku["std_monthly_qty"] / sku["avg_monthly_qty"],
        0,
    )
    sku["demand_cv"] = sku["demand_cv"].replace([np.inf, -np.inf], 0).fillna(0)

    latest_date = sales_df["InvoiceDate"].max()
    sku["days_since_last_sale"] = (latest_date - sku["last_sale"]).dt.days

    latest_month = sales_df["month"].max()
    recent_start = latest_month - pd.DateOffset(months=2)
    prev_start = latest_month - pd.DateOffset(months=5)
    prev_end = latest_month - pd.DateOffset(months=3)

    recent = monthly[(monthly["month"] >= recent_start) & (monthly["month"] <= latest_month)]
    prev = monthly[(monthly["month"] >= prev_start) & (monthly["month"] <= prev_end)]

    recent_agg = recent.groupby(["StockCode", "Description"], as_index=False)["qty"].sum().rename(
        columns={"qty": "recent_3m_qty"}
    )
    prev_agg = prev.groupby(["StockCode", "Description"], as_index=False)["qty"].sum().rename(
        columns={"qty": "prev_3m_qty"}
    )

    sku = sku.merge(recent_agg, on=["StockCode", "Description"], how="left")
    sku = sku.merge(prev_agg, on=["StockCode", "Description"], how="left")
    sku["recent_3m_qty"] = sku["recent_3m_qty"].fillna(0)
    sku["prev_3m_qty"] = sku["prev_3m_qty"].fillna(0)

    sku["trend_pct"] = np.where(
        sku["prev_3m_qty"] > 0,
        (sku["recent_3m_qty"] - sku["prev_3m_qty"]) / sku["prev_3m_qty"],
        np.where(sku["recent_3m_qty"] > 0, 1.0, 0.0),
    )

    total_revenue = max(sku["total_revenue"].sum(), 1.0)
    sku["revenue_share"] = sku["total_revenue"] / total_revenue

    sku["variability_band"] = pd.cut(
        sku["demand_cv"],
        bins=[-0.01, 0.25, 0.5, 1.0, 999],
        labels=["Low", "Moderate", "High", "Very High"],
    ).astype(str)

    sku["velocity_band"] = pd.cut(
        sku["avg_monthly_qty"],
        bins=[-0.01, 10, 50, 200, 999999],
        labels=["Low", "Moderate", "High", "Very High"],
    ).astype(str)

    sku["stockout_risk_score"] = (
        40 * np.clip(sku["demand_cv"], 0, 2)
        + 20 * np.clip(sku["trend_pct"], 0, 2)
        + 20 * np.where(sku["revenue_share"] >= sku["revenue_share"].quantile(0.8), 1, 0)
        + 20 * np.where(sku["days_since_last_sale"] <= 14, 1, 0)
    )
    sku["stockout_risk_score"] = np.clip(sku["stockout_risk_score"], 0, 100)

    sku["overstock_risk_score"] = (
        35 * np.where(sku["days_since_last_sale"] > 60, 1, 0)
        + 30 * np.where(sku["trend_pct"] < -0.30, 1, 0)
        + 20 * np.where(sku["avg_monthly_qty"] < 10, 1, 0)
        + 15 * np.where(sku["demand_cv"] < 0.25, 1, 0)
    )
    sku["overstock_risk_score"] = np.clip(sku["overstock_risk_score"], 0, 100)

    sku = sku.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    sku["cum_revenue_share"] = sku["total_revenue"].cumsum() / total_revenue
    sku["abc_class"] = np.select(
        [sku["cum_revenue_share"] <= 0.80, sku["cum_revenue_share"] <= 0.95],
        ["A", "B"],
        default="C",
    )

    return sku


# =============================================================================
# GROQ LLM LAYER
# =============================================================================
def groq_available() -> bool:
    return "GROQ_API_KEY" in st.secrets and bool(st.secrets["GROQ_API_KEY"])


def call_groq_intent_parser(user_text: str) -> Dict[str, Any]:
    """
    Uses Groq's OpenAI-compatible chat completions API.
    """
    api_key = st.secrets["GROQ_API_KEY"]
    model = st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    url = "https://api.groq.com/openai/v1/chat/completions"

    system_prompt = """
You are an expert supply chain planning intent parser.

Your job is to convert a user's natural-language planning request into strict JSON.

Return ONLY valid JSON with this exact schema:
{
  "objectives": [string],
  "constraints": [string],
  "target_segment": string,
  "priority": string
}

Allowed objective values:
- reduce_stockouts
- reduce_overstock
- protect_revenue
- manage_variability
- balanced_inventory

Allowed constraint values:
- limit_inventory_increase
- cost_sensitive
- urgent

Allowed target_segment values:
- all_skus
- high_variability_skus
- slow_moving_skus
- high_value_skus

Allowed priority values:
- service
- cost
- balanced

Rules:
- If the user mentions stockouts, availability, fill rate, service issues -> reduce_stockouts
- If the user mentions overstock, dead stock, excess inventory -> reduce_overstock
- If the user mentions revenue, top items, key products -> protect_revenue
- If the user mentions volatility, variability, unstable demand -> manage_variability
- If no objective is clear -> balanced_inventory

- If the user says don't increase inventory, control inventory, limit inventory -> limit_inventory_increase
- If the user mentions cost, budget, lower cost -> cost_sensitive
- If the user mentions urgent, critical, immediate -> urgent

- If the user mentions high variability, volatile, erratic, lumpy -> high_variability_skus
- If the user mentions slow moving, dead stock, low velocity -> slow_moving_skus
- If the user mentions top revenue, important items, A items -> high_value_skus
- Otherwise -> all_skus

- If the user emphasizes service or availability -> service
- If the user emphasizes cost -> cost
- Otherwise -> balanced
""".strip()

    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    # Defensive normalization
    return {
        "raw_text": user_text,
        "objectives": parsed.get("objectives", ["balanced_inventory"]),
        "constraints": parsed.get("constraints", []),
        "target_segment": parsed.get("target_segment", "all_skus"),
        "priority": parsed.get("priority", "balanced"),
    }


def fallback_parse_intent(user_text: str) -> Dict[str, Any]:
    """
    Rule-based fallback if Groq key is missing or request fails.
    """
    t = (user_text or "").lower().strip()

    objectives = []
    constraints = []
    target_segment = "all_skus"
    priority = "balanced"

    if any(x in t for x in ["stockout", "availability", "fill rate", "service"]):
        objectives.append("reduce_stockouts")
    if any(x in t for x in ["overstock", "dead stock", "excess inventory", "slow moving"]):
        objectives.append("reduce_overstock")
    if any(x in t for x in ["revenue", "top item", "important item", "a item"]):
        objectives.append("protect_revenue")
    if any(x in t for x in ["variability", "volatile", "erratic", "lumpy", "unstable"]):
        objectives.append("manage_variability")
    if not objectives:
        objectives = ["balanced_inventory"]

    if any(x in t for x in ["don't increase inventory", "do not increase inventory", "limit inventory", "inventory too much"]):
        constraints.append("limit_inventory_increase")
    if any(x in t for x in ["cost", "budget", "save money", "reduce cost"]):
        constraints.append("cost_sensitive")
    if any(x in t for x in ["urgent", "critical", "immediately", "asap"]):
        constraints.append("urgent")

    if any(x in t for x in ["high variability", "volatile", "erratic", "lumpy"]):
        target_segment = "high_variability_skus"
    elif any(x in t for x in ["slow moving", "dead stock", "low velocity"]):
        target_segment = "slow_moving_skus"
    elif any(x in t for x in ["top revenue", "important items", "a items", "high value"]):
        target_segment = "high_value_skus"

    if any(x in t for x in ["service first", "availability first"]):
        priority = "service"
    elif any(x in t for x in ["cost first", "budget first"]):
        priority = "cost"

    return {
        "raw_text": user_text,
        "objectives": objectives,
        "constraints": constraints,
        "target_segment": target_segment,
        "priority": priority,
    }


def parse_intent(user_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Returns intent + source label.
    """
    if groq_available():
        try:
            return call_groq_intent_parser(user_text), "Groq LLM"
        except Exception as e:
            st.warning(f"Groq parsing failed, using fallback rules. Error: {e}")
            return fallback_parse_intent(user_text), "Fallback rules"

    return fallback_parse_intent(user_text), "Fallback rules"


# =============================================================================
# SYMBOLIC LAYER
# =============================================================================
def choose_target_df(sku_df: pd.DataFrame, intent: Dict[str, Any]) -> pd.DataFrame:
    df = sku_df.copy()

    if intent["target_segment"] == "high_variability_skus":
        df = df[df["demand_cv"] >= 0.5]
    elif intent["target_segment"] == "slow_moving_skus":
        df = df[(df["avg_monthly_qty"] < 10) | (df["days_since_last_sale"] > 45)]
    elif intent["target_segment"] == "high_value_skus":
        df = df[(df["abc_class"] == "A") | (df["revenue_share"] >= df["revenue_share"].quantile(0.8))]

    return df.copy()


def symbolic_recommendations(target_df: pd.DataFrame, intent: Dict[str, Any]) -> Tuple[List[str], pd.DataFrame]:
    if target_df.empty:
        return ["No SKUs matched the selected planning intent."], pd.DataFrame()

    work = target_df.copy()
    actions: List[str] = []

    limit_inventory = "limit_inventory_increase" in intent["constraints"]
    cost_sensitive = "cost_sensitive" in intent["constraints"]
    service_mode = intent["priority"] == "service"
    cost_mode = intent["priority"] == "cost"

    if service_mode:
        safety_stock_pct = 18.0
        reorder_pct = 12.0
    elif cost_mode or cost_sensitive or limit_inventory:
        safety_stock_pct = 6.0
        reorder_pct = 4.0
    else:
        safety_stock_pct = 12.0
        reorder_pct = 8.0

    frames = []

    if "reduce_stockouts" in intent["objectives"]:
        stockout = work.sort_values(
            ["stockout_risk_score", "total_revenue"],
            ascending=[False, False],
        ).head(15).copy()
        stockout["recommended_action"] = "Increase safety stock / review reorder point"
        stockout["suggested_safety_stock_change_pct"] = safety_stock_pct
        stockout["suggested_reorder_point_change_pct"] = reorder_pct
        stockout["reason"] = np.where(
            stockout["demand_cv"] >= 0.5,
            "High demand variability",
            "High demand pressure or commercial importance",
        )
        actions.append(
            f"Raise protection selectively for the highest stockout-risk SKUs, capped at {safety_stock_pct:.1f}% safety stock increase."
        )
        frames.append(stockout)

    if "reduce_overstock" in intent["objectives"]:
        overstock = work.sort_values(
            ["overstock_risk_score", "days_since_last_sale"],
            ascending=[False, False],
        ).head(15).copy()
        overstock["recommended_action"] = "Lower reorder point / review replenishment frequency"
        overstock["suggested_safety_stock_change_pct"] = -max(reorder_pct, 5.0)
        overstock["suggested_reorder_point_change_pct"] = -max(reorder_pct, 5.0)
        overstock["reason"] = np.where(
            overstock["days_since_last_sale"] > 60,
            "Weak recency signal",
            "Demand slowing or low velocity",
        )
        actions.append("Reduce exposure on slow-moving or weakening-demand SKUs.")
        frames.append(overstock)

    if "protect_revenue" in intent["objectives"]:
        actions.append("Prioritize A-class and top-revenue SKUs when trade-offs must be made.")

    if "manage_variability" in intent["objectives"]:
        actions.append("Use demand variability as the main segmentation driver for policy changes.")

    if limit_inventory:
        actions.append("Inventory increase constraint detected: action sizes were intentionally capped.")

    if cost_sensitive:
        actions.append("Cost sensitivity detected: leaner recommendations were selected.")

    if not frames:
        generic = work.sort_values("total_revenue", ascending=False).head(15).copy()
        generic["recommended_action"] = "Review policy settings"
        generic["suggested_safety_stock_change_pct"] = 0.0
        generic["suggested_reorder_point_change_pct"] = 0.0
        generic["reason"] = "Balanced review based on current intent"
        frames.append(generic)

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["StockCode", "recommended_action"])

    keep = [
        "StockCode",
        "Description",
        "abc_class",
        "variability_band",
        "velocity_band",
        "total_revenue",
        "avg_monthly_qty",
        "demand_cv",
        "trend_pct",
        "days_since_last_sale",
        "stockout_risk_score",
        "overstock_risk_score",
        "recommended_action",
        "suggested_safety_stock_change_pct",
        "suggested_reorder_point_change_pct",
        "reason",
    ]
    return actions, result[keep]


# =============================================================================
# LOAD DATA
# =============================================================================
try:
    sales_df = load_demo_data()
    sku_df = build_sku_metrics(sales_df)
except Exception as e:
    st.error(f"Failed to load demo dataset: {e}")
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("Settings")
show_preview = st.sidebar.checkbox("Preview raw dataset", value=True)
top_n_chart = st.sidebar.slider("Top SKUs in revenue chart", 5, 30, 10)

st.sidebar.markdown("### LLM status")
if groq_available():
    st.sidebar.success("Groq connected")
    st.sidebar.caption(f'Model: `{st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")}`')
else:
    st.sidebar.info("No GROQ_API_KEY found. App will use fallback rule parsing.")

# =============================================================================
# DATASET OVERVIEW
# =============================================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(sales_df):,}")
c2.metric("Unique SKUs", f"{sales_df['StockCode'].nunique():,}")
c3.metric("Unique Customers", f"{sales_df['CustomerID'].nunique():,}")
c4.metric("Countries", f"{sales_df['Country'].nunique():,}")

st.markdown("### Demo dataset")
st.info("Source: UCI Online Retail dataset")

if show_preview:
    with st.expander("Preview demo data"):
        st.dataframe(sales_df.head(50), use_container_width=True)

# =============================================================================
# SNAPSHOT CHARTS
# =============================================================================
st.markdown("### Business snapshot")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Revenue", f"£{sales_df['line_revenue'].sum():,.0f}")
m2.metric("Average Order Value", f"£{sales_df.groupby('InvoiceNo')['line_revenue'].sum().mean():,.1f}")
m3.metric("Avg Monthly Qty / SKU", f"{sku_df['avg_monthly_qty'].mean():,.1f}")
m4.metric("Median Demand CV", f"{sku_df['demand_cv'].median():.2f}")

left, right = st.columns(2)

with left:
    top_revenue = sku_df.nlargest(top_n_chart, "total_revenue")[["Description", "total_revenue"]].sort_values(
        "total_revenue"
    )
    fig = px.bar(
        top_revenue,
        x="total_revenue",
        y="Description",
        orientation="h",
        title="Top SKUs by Revenue",
        labels={"total_revenue": "Revenue (£)", "Description": "SKU"},
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig2 = px.scatter(
        sku_df,
        x="demand_cv",
        y="avg_monthly_qty",
        size="total_revenue",
        color="abc_class",
        hover_data=["Description", "days_since_last_sale"],
        title="SKU Demand Risk Map",
        labels={
            "demand_cv": "Demand Variability (CV)",
            "avg_monthly_qty": "Avg Monthly Qty",
        },
    )
    st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# DECISION TRANSLATOR
# =============================================================================
st.markdown("### Decision Translator")

default_prompt = "Reduce stockouts for high variability SKUs without increasing inventory too much"

user_prompt = st.text_area(
    "Enter a planning request",
    value=default_prompt,
    height=120,
)

if st.button("Generate Decision Recommendation", type="primary"):
    intent, parser_source = parse_intent(user_prompt)
    target_df = choose_target_df(sku_df, intent)
    summary_actions, recommendations = symbolic_recommendations(target_df, intent)

    st.markdown("#### Interpreted intent")
    i1, i2 = st.columns([1, 1])

    with i1:
        st.json(
            {
                "objectives": intent["objectives"],
                "constraints": intent["constraints"],
                "target_segment": intent["target_segment"],
                "priority": intent["priority"],
            }
        )

    with i2:
        st.markdown("**Parsing source**")
        st.write(parser_source)
        st.markdown("**What happened**")
        st.write(
            "The app interpreted your free-text request, mapped it to structured planning logic, and then applied deterministic supply chain rules."
        )

    st.markdown("#### Recommendation summary")
    for item in summary_actions:
        st.write(f"- {item}")

    if recommendations.empty:
        st.warning("No recommendations were generated.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Recommended SKU actions", f"{len(recommendations)}")
        k2.metric("Avg Stockout Risk", f"{recommendations['stockout_risk_score'].mean():.1f}")
        k3.metric("Avg Overstock Risk", f"{recommendations['overstock_risk_score'].mean():.1f}")

        st.markdown("#### Recommended SKU actions")
        st.dataframe(
            recommendations.sort_values(
                ["stockout_risk_score", "total_revenue"],
                ascending=[False, False],
            ),
            use_container_width=True,
        )

        e1, e2 = st.columns(2)

        with e1:
            reasons = recommendations["reason"].value_counts().reset_index()
            reasons.columns = ["Reason", "Count"]
            fig3 = px.bar(reasons, x="Reason", y="Count", title="Top Recommendation Drivers")
            st.plotly_chart(fig3, use_container_width=True)

        with e2:
            mix = recommendations["recommended_action"].value_counts().reset_index()
            mix.columns = ["Action", "Count"]
            fig4 = px.pie(mix, names="Action", values="Count", title="Recommendation Mix")
            st.plotly_chart(fig4, use_container_width=True)

# =============================================================================
# SKU EXPLORER
# =============================================================================
st.markdown("### SKU Explorer")

f1, f2, f3 = st.columns(3)

selected_abc = f1.multiselect(
    "ABC class",
    options=sorted(sku_df["abc_class"].dropna().unique()),
    default=sorted(sku_df["abc_class"].dropna().unique()),
)

selected_var = f2.multiselect(
    "Variability band",
    options=[x for x in sku_df["variability_band"].dropna().unique().tolist() if x != "nan"],
    default=[x for x in sku_df["variability_band"].dropna().unique().tolist() if x != "nan"],
)

min_revenue = f3.number_input("Minimum revenue (£)", min_value=0.0, value=0.0, step=100.0)

explorer = sku_df.copy()
explorer = explorer[explorer["abc_class"].isin(selected_abc)]
explorer = explorer[explorer["variability_band"].isin(selected_var)]
explorer = explorer[explorer["total_revenue"] >= min_revenue]

st.dataframe(
    explorer[
        [
            "StockCode",
            "Description",
            "abc_class",
            "variability_band",
            "velocity_band",
            "total_revenue",
            "avg_monthly_qty",
            "demand_cv",
            "trend_pct",
            "days_since_last_sale",
            "stockout_risk_score",
            "overstock_risk_score",
        ]
    ].sort_values("total_revenue", ascending=False),
    use_container_width=True,
)

st.markdown("---")
st.caption(
    "For production use, connect this to ERP inventory, supplier lead times, service levels, and policy constraints."
)
