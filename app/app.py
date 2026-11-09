
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import gradio as gr


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uganda_fuel_prices_master.csv"
)

FORECAST_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "latest_fuel_forecast.csv"
)

PROMOTION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "model_promotion_history.csv"
)

PIPELINE_LOG = (
    PROJECT_ROOT
    / "outputs"
    / "pipeline_runs.log"
)

CURRENT_MONTH = pd.Timestamp.now().to_period("M")

NAVY = "#0B1220"
PANEL = "#111C2E"
BLUE = "#3B82F6"
CYAN = "#22D3EE"
GREEN = "#22C55E"
RED = "#F87171"
TEXT = "#E5E7EB"
MUTED = "#94A3B8"


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_master():

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {MASTER_FILE}"
        )

    df = pd.read_csv(MASTER_FILE)

    required = [
        "month",
        "petrol_shs_per_litre",
        "diesel_shs_per_litre"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Master dataset is missing columns: {missing}"
        )

    df["month"] = pd.to_datetime(
        df["month"], errors="coerce"
    )

    for column in [
        "petrol_shs_per_litre",
        "diesel_shs_per_litre",
        "paraffin_shs_per_litre"
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    df = df.dropna(subset=["month"])

    df = (
        df.sort_values("month")
        .drop_duplicates(subset=["month"])
        .reset_index(drop=True)
    )

    return df


def load_forecast():

    if not FORECAST_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(FORECAST_FILE)

    if df.empty:
        return df

    required = [
        "Fuel",
        "Forecast_Month",
        "Latest_Actual_Price",
        "Predicted_Price"
    ]

    for column in required:

        if column not in df.columns:
            raise ValueError(
                f"Forecast file is missing {column}"
            )

    df["Forecast_Month"] = (
        df["Forecast_Month"].astype(str)
    )

    df["Latest_Actual_Month"] = (
        df["Latest_Actual_Month"].astype(str)
    )

    return df


def load_promotion_history():

    if not PROMOTION_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(PROMOTION_FILE)

    return df


def load_pipeline_logs():

    if not PIPELINE_LOG.exists():
        return ["No pipeline logs available."]

    with open(
        PIPELINE_LOG,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        lines = file.readlines()

    return [
        line.rstrip()
        for line in lines[-40:]
    ]


# ============================================================
# 3. FORMATTING HELPERS
# ============================================================

def format_ugx(value):

    if pd.isna(value):
        return "N/A"

    return f"UGX {value:,.0f}"


def format_percentage(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:+.2f}%"


def make_card(title, value, subtitle, accent=BLUE):

    return f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtitle">{subtitle}</div>
        <div class="metric-accent"
             style="background:{accent};"></div>
    </div>
    """


def dark_layout(fig, title=None):

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=NAVY,
        plot_bgcolor=PANEL,
        font=dict(
            family="Inter, Arial, sans-serif",
            color=TEXT,
            size=12
        ),
        margin=dict(
            l=45,
            r=25,
            t=65,
            b=45
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_color=TEXT
        )
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False
    )

    fig.update_yaxes(
        gridcolor="#263449",
        zeroline=False,
        title="UGX per litre"
    )

    return fig


# ============================================================
# 4. KPI CARDS
# ============================================================

def build_kpis(master, forecasts):

    latest = master.iloc[-1]

    latest_month = latest["month"].strftime("%b %Y")

    petrol = latest["petrol_shs_per_litre"]
    diesel = latest["diesel_shs_per_litre"]

    petrol_forecast = forecasts[
        forecasts["Fuel"].str.lower() == "petrol"
    ]

    diesel_forecast = forecasts[
        forecasts["Fuel"].str.lower() == "diesel"
    ]

    petrol_prediction = (
        petrol_forecast.iloc[-1]["Predicted_Price"]
        if not petrol_forecast.empty else np.nan
    )

    diesel_prediction = (
        diesel_forecast.iloc[-1]["Predicted_Price"]
        if not diesel_forecast.empty else np.nan
    )

    cards = [
        make_card(
            "LATEST PETROL PRICE",
            format_ugx(petrol),
            f"Latest actual · {latest_month}",
            BLUE
        ),
        make_card(
            "LATEST DIESEL PRICE",
            format_ugx(diesel),
            f"Latest actual · {latest_month}",
            CYAN
        ),
        make_card(
            "PETROL MODEL ESTIMATE",
            format_ugx(petrol_prediction),
            "Next unobserved month in dataset",
            GREEN
        ),
        make_card(
            "DIESEL MODEL ESTIMATE",
            format_ugx(diesel_prediction),
            "Next unobserved month in dataset",
            GREEN
        )
    ]

    return (
        '<div class="metric-grid">'
        + "".join(cards)
        + "</div>"
    )


# ============================================================
# 5. MARKET HISTORY CHART
# ============================================================

def build_market_chart(master):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=master["month"],
            y=master["petrol_shs_per_litre"],
            name="Petrol",
            mode="lines",
            line=dict(
                color=BLUE,
                width=3
            ),
            connectgaps=False,
            hovertemplate=(
                "Month: %{x|%b %Y}<br>"
                "Petrol: UGX %{y:,.0f}"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=master["month"],
            y=master["diesel_shs_per_litre"],
            name="Diesel",
            mode="lines",
            line=dict(
                color=CYAN,
                width=3
            ),
            connectgaps=False,
            hovertemplate=(
                "Month: %{x|%b %Y}<br>"
                "Diesel: UGX %{y:,.0f}"
                "<extra></extra>"
            )
        )
    )

    if "paraffin_shs_per_litre" in master.columns:

        if master["paraffin_shs_per_litre"].notna().any():

            fig.add_trace(
                go.Scatter(
                    x=master["month"],
                    y=master["paraffin_shs_per_litre"],
                    name="Paraffin",
                    mode="lines",
                    line=dict(
                        color="#F59E0B",
                        width=2
                    ),
                    connectgaps=False
                )
            )

    fig.update_layout(
        hovermode="x unified",
        height=470
    )

    return dark_layout(
        fig,
        "Historical National Fuel Prices"
    )


# ============================================================
# 6. MONTHLY PRICE MOVEMENT
# ============================================================

def build_movement_chart(master):

    df = master.copy()

    df["Petrol_Change"] = (
        df["petrol_shs_per_litre"].diff()
    )

    df["Diesel_Change"] = (
        df["diesel_shs_per_litre"].diff()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["month"],
            y=df["Petrol_Change"],
            name="Petrol change",
            marker_color=BLUE
        )
    )

    fig.add_trace(
        go.Bar(
            x=df["month"],
            y=df["Diesel_Change"],
            name="Diesel change",
            marker_color=CYAN
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=MUTED
    )

    fig.update_layout(
        barmode="group",
        height=390,
        yaxis_title="Monthly price change (UGX)"
    )

    return dark_layout(
        fig,
        "Month-to-Month Price Changes"
    )


# ============================================================
# 7. FORECAST CHART
# ============================================================

def build_forecast_chart(master, forecasts):

    fig = go.Figure()

    fuel_config = {
        "Petrol": {
            "column": "petrol_shs_per_litre",
            "color": BLUE
        },
        "Diesel": {
            "column": "diesel_shs_per_litre",
            "color": CYAN
        }
    }

    for fuel, config in fuel_config.items():

        column = config["column"]

        fig.add_trace(
            go.Scatter(
                x=master["month"],
                y=master[column],
                name=f"{fuel} actual",
                mode="lines",
                line=dict(
                    color=config["color"],
                    width=2.5
                ),
                connectgaps=False
            )
        )

        selected = forecasts[
            forecasts["Fuel"].str.lower() == fuel.lower()
        ]

        if selected.empty:
            continue

        prediction = selected.iloc[-1]

        forecast_date = pd.to_datetime(
            prediction["Forecast_Month"] + "-01"
        )

        latest_date = master["month"].max()

        fig.add_trace(
            go.Scatter(
                x=[latest_date, forecast_date],
                y=[
                    prediction["Latest_Actual_Price"],
                    prediction["Predicted_Price"]
                ],
                name=f"{fuel} estimate",
                mode="lines+markers",
                line=dict(
                    color=config["color"],
                    width=3,
                    dash="dash"
                ),
                marker=dict(size=10),
                hovertemplate=(
                    "%{x|%b %Y}<br>"
                    "Price: UGX %{y:,.0f}"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        height=470,
        hovermode="x unified"
    )

    return dark_layout(
        fig,
        "Actual Prices and Model Estimates"
    )


# ============================================================
# 8. MODEL PERFORMANCE
# ============================================================

def build_model_performance(history):

    if history.empty:

        return (
            go.Figure().update_layout(
                template="plotly_dark",
                title="No promotion history available"
            ),
            pd.DataFrame()
        )

    latest = history.copy()

    if "timestamp" in latest.columns:

        latest["timestamp"] = pd.to_datetime(
            latest["timestamp"],
            errors="coerce"
        )

        latest = latest.sort_values("timestamp")

    if "fuel" in latest.columns:

        latest = latest.drop_duplicates(
            subset=["fuel"],
            keep="last"
        )

    metric_columns = [
        "fuel",
        "candidate_mae",
        "baseline_mae",
        "improvement_percent",
        "identical_to_production",
        "decision",
        "reason"
    ]

    available_columns = [
        column
        for column in metric_columns
        if column in latest.columns
    ]

    display = latest[available_columns].copy()

    fig = go.Figure()

    for _, row in latest.iterrows():

        fuel = str(row.get("fuel", "Unknown")).title()

        candidate = row.get("candidate_mae", np.nan)
        baseline = row.get("baseline_mae", np.nan)

        if pd.notna(candidate) and pd.notna(baseline):

            fig.add_trace(
                go.Bar(
                    x=["Candidate", "Persistence baseline"],
                    y=[candidate, baseline],
                    name=fuel,
                    text=[
                        f"{candidate:.2f}",
                        f"{baseline:.2f}"
                    ],
                    textposition="outside"
                )
            )

    fig.update_layout(
        barmode="group",
        height=390,
        yaxis_title="Rolling validation MAE (UGX)"
    )

    fig = dark_layout(
        fig,
        "Candidate Model vs Persistence Baseline"
    )

    return fig, display


# ============================================================
# 9. DATA QUALITY
# ============================================================

def build_data_quality(master):

    expected = pd.period_range(
        start=master["month"].min().to_period("M"),
        end=master["month"].max().to_period("M"),
        freq="M"
    )

    observed = set(
        master["month"].dt.to_period("M")
    )

    missing = [
        str(month)
        for month in expected
        if month not in observed
    ]

    latest_month = master["month"].max().to_period("M")

    records = [
        {
            "Check": "Historical records",
            "Result": len(master),
            "Status": "Available"
        },
        {
            "Check": "First observation",
            "Result": str(master["month"].min().to_period("M")),
            "Status": "Available"
        },
        {
            "Check": "Latest observation",
            "Result": str(latest_month),
            "Status": "Available"
        },
        {
            "Check": "Missing months",
            "Result": ", ".join(missing) if missing else "None",
            "Status": "Review" if missing else "Passed"
        },
        {
            "Check": "Duplicate months",
            "Result": int(master["month"].duplicated().sum()),
            "Status": "Review" if master["month"].duplicated().any() else "Passed"
        },
        {
            "Check": "Missing petrol values",
            "Result": int(master["petrol_shs_per_litre"].isna().sum()),
            "Status": "Review" if master["petrol_shs_per_litre"].isna().any() else "Passed"
        },
        {
            "Check": "Missing diesel values",
            "Result": int(master["diesel_shs_per_litre"].isna().sum()),
            "Status": "Review" if master["diesel_shs_per_litre"].isna().any() else "Passed"
        }
    ]

    if "paraffin_shs_per_litre" in master.columns:

        records.append({
            "Check": "Missing paraffin values",
            "Result": int(master["paraffin_shs_per_litre"].isna().sum()),
            "Status": "Information"
        })

    return pd.DataFrame(records)


# ============================================================
# 10. FORECAST STATUS
# ============================================================

def build_forecast_status(forecasts):

    if forecasts.empty:

        return (
            "## Forecast status\n"
            "No forecast output is available."
        )

    messages = []

    for _, row in forecasts.iterrows():

        forecast_period = pd.Period(
            row["Forecast_Month"],
            freq="M"
        )

        if forecast_period < CURRENT_MONTH:

            status = (
                "Historical-month estimate — "
                "actual observation not yet verified"
            )

        elif forecast_period == CURRENT_MONTH:

            status = "Current-month estimate"

        else:

            status = "Future-month forecast"

        messages.append(
            f"**{row['Fuel']} — {row['Forecast_Month']}**  \n"
            f"{status}"
        )

    return "\n\n".join(messages)


# ============================================================
# 11. DASHBOARD DATA REFRESH
# ============================================================

def refresh_dashboard():

    master = load_master()
    forecasts = load_forecast()
    history = load_promotion_history()

    kpis = build_kpis(master, forecasts)

    market = build_market_chart(master)

    movement = build_movement_chart(master)

    forecast_chart = build_forecast_chart(
        master, forecasts
    )

    model_chart, model_table = (
        build_model_performance(history)
    )

    quality = build_data_quality(master)

    logs = "\n".join(load_pipeline_logs())

    forecast_status = build_forecast_status(
        forecasts
    )

    forecast_table = (
        forecasts if not forecasts.empty
        else pd.DataFrame(
            columns=[
                "Fuel",
                "Forecast_Month",
                "Predicted_Price"
            ]
        )
    )

    return (
        kpis,
        market,
        movement,
        forecast_chart,
        forecast_table,
        forecast_status,
        model_chart,
        model_table,
        quality,
        logs
    )


# ============================================================
# 12. DASHBOARD STYLING
# ============================================================

CSS = """
body {
    background: #0B1220 !important;
}

.gradio-container {
    background: #0B1220 !important;
    color: #E5E7EB !important;
    max-width: 1600px !important;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 15px;
    margin: 12px 0 22px 0;
}

.metric-card {
    position: relative;
    background: #111C2E;
    border: 1px solid #263449;
    border-radius: 13px;
    padding: 21px;
    min-height: 125px;
    overflow: hidden;
}

.metric-label {
    color: #94A3B8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

.metric-value {
    color: #F8FAFC;
    font-size: 27px;
    font-weight: 750;
    margin-top: 13px;
}

.metric-subtitle {
    color: #94A3B8;
    font-size: 11px;
    margin-top: 8px;
}

.metric-accent {
    position: absolute;
    height: 3px;
    bottom: 0;
    left: 0;
    right: 0;
}

.section-heading {
    color: #F8FAFC;
    font-size: 21px;
    font-weight: 700;
    margin: 12px 0;
}

.app-header {
    background: linear-gradient(120deg, #111C2E, #172A46);
    border: 1px solid #263449;
    border-radius: 15px;
    padding: 24px;
    margin-bottom: 18px;
}

.app-header h1 {
    color: #F8FAFC;
    font-size: 29px;
    margin: 0;
}

.app-header p {
    color: #94A3B8;
    margin: 8px 0 0 0;
    letter-spacing: 1px;
    font-size: 11px;
}

@media(max-width: 850px) {
    .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media(max-width: 500px) {
    .metric-grid {
        grid-template-columns: 1fr;
    }
}
"""


# ============================================================
# 13. GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate"
    ),
    css=CSS,
    title="Ugandan Fuel Prediction Intel"
) as demo:

    gr.HTML("""
    <div class="app-header">
        <h1>UGANDAN FUEL PREDICTION INTEL</h1>
        <p>
            NATIONAL FUEL MARKET ANALYTICS
            &nbsp; | &nbsp;
            FORECASTING INTELLIGENCE
            &nbsp; | &nbsp;
            DATA MONITORING
        </p>
    </div>
    """)

    with gr.Row():

        gr.Markdown(
            "**MARKET INTELLIGENCE / OVERVIEW**"
        )

        refresh_button = gr.Button(
            "↻ Refresh Dashboard",
            variant="primary"
        )

    with gr.Tab("Overview"):

        kpi_output = gr.HTML()

        with gr.Row():

            overview_market = gr.Plot(
                label="Historical Fuel Prices"
            )

        with gr.Row():

            overview_movement = gr.Plot(
                label="Monthly Price Movement"
            )

    with gr.Tab("Market Analysis"):

        gr.Markdown(
            "## Historical Fuel Market"
        )

        market_chart = gr.Plot()

        movement_chart = gr.Plot()

    with gr.Tab("Forecast Intelligence"):

        gr.Markdown(
            "## Fuel Price Forecasts"
        )

        forecast_status = gr.Markdown()

        forecast_chart = gr.Plot()

        forecast_table = gr.Dataframe(
            label="Latest Forecast Records",
            interactive=False
        )

    with gr.Tab("Model Performance"):

        gr.Markdown(
            "## Candidate Model Evaluation"
        )

        gr.Markdown(
            "MAE comparisons below represent rolling "
            "validation results recorded during candidate "
            "model evaluation. They are not a substitute "
            "for independent production-model backtesting."
        )

        model_chart = gr.Plot()

        model_table = gr.Dataframe(
            interactive=False,
            label="Latest Model Promotion Decisions"
        )

    with gr.Tab("Data & System Health"):

        gr.Markdown(
            "## Master Dataset Quality"
        )

        quality_table = gr.Dataframe(
            interactive=False,
            label="Data Quality Checks"
        )

        gr.Markdown(
            "## Pipeline Execution Logs"
        )

        pipeline_logs = gr.Textbox(
            lines=12,
            interactive=False,
            label="Recent Pipeline Activity"
        )

    outputs = [
        kpi_output,
        overview_market,
        overview_movement,
        forecast_chart,
        forecast_table,
        forecast_status,
        model_chart,
        model_table,
        quality_table,
        pipeline_logs
    ]

    demo.load(
        fn=refresh_dashboard,
        inputs=[],
        outputs=outputs
    )

    refresh_button.click(
        fn=refresh_dashboard,
        inputs=[],
        outputs=outputs
    )


# ============================================================
# 14. APPLICATION LAUNCH
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("UGANDAN FUEL PREDICTION INTEL")
    print("=" * 60)

    print("Starting Gradio dashboard...")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
