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

NAVY = "#0B1220"
PANEL = "#111C2E"

UGANDA_BLACK = "#000000"
UGANDA_YELLOW = "#FCDC04"
UGANDA_RED = "#D90000"

PETROL_COLOR = UGANDA_RED
DIESEL_COLOR = UGANDA_YELLOW
PARAFFIN_COLOR = "#E5E7EB"

GREEN = "#22C55E"
RED = UGANDA_RED

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
        df["month"],
        errors="coerce"
    )

    for column in [
        "petrol_shs_per_litre",
        "diesel_shs_per_litre",
        "paraffin_shs_per_litre"
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
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

    if "Latest_Actual_Month" in df.columns:

        df["Latest_Actual_Month"] = (
            df["Latest_Actual_Month"].astype(str)
        )

    df["Latest_Actual_Price"] = pd.to_numeric(
        df["Latest_Actual_Price"],
        errors="coerce"
    )

    df["Predicted_Price"] = pd.to_numeric(
        df["Predicted_Price"],
        errors="coerce"
    )

    return df


def load_promotion_history():

    if not PROMOTION_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(PROMOTION_FILE)


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


def make_card(
    title,
    value,
    subtitle,
    accent=PETROL_COLOR
):

    return f"""
    <div class="metric-card">

        <div class="metric-label">
            {title}
        </div>

        <div class="metric-value">
            {value}
        </div>

        <div class="metric-subtitle">
            {subtitle}
        </div>

        <div class="metric-accent"
             style="background:{accent};">
        </div>

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

        autosize=True,
        height=470,

        margin=dict(
            l=55,
            r=25,
            t=75,
            b=55
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
        zeroline=False,
        automargin=True
    )

    fig.update_yaxes(
        gridcolor="#263449",
        zeroline=False,
        title="UGX per litre",
        automargin=True
    )

    return fig


# ============================================================
# 4. KPI CARDS
# ============================================================

def build_kpis(master, forecasts):

    latest = master.iloc[-1]

    latest_month = latest["month"].strftime("%b %Y")

    latest_period = master["month"].max().to_period("M")

    current_period = pd.Timestamp.now().to_period("M")

    petrol = latest["petrol_shs_per_litre"]

    diesel = latest["diesel_shs_per_litre"]

    def get_estimate(fuel):

        selected = (
            forecasts[
                forecasts["Fuel"].astype(str).str.lower() == fuel
            ]
            if not forecasts.empty
            else pd.DataFrame()
        )

        if selected.empty:
            return np.nan, "No estimate available"

        row = selected.iloc[-1]

        forecast_period = pd.Period(
            str(row["Forecast_Month"]),
            freq="M"
        )

        month_label = forecast_period.strftime("%b %Y")

        if forecast_period < current_period:

            label = (
                f"Historical estimate · {month_label} · unverified"
            )

        elif latest_period < current_period - 1:

            label = (
                f"Estimate · {month_label} · data is outdated"
            )

        else:

            label = f"Model estimate · {month_label}"

        return row["Predicted_Price"], label

    petrol_prediction, petrol_label = get_estimate("petrol")

    diesel_prediction, diesel_label = get_estimate("diesel")

    cards = [

        make_card(
            "LATEST PETROL PRICE",
            format_ugx(petrol),
            f"Latest actual · {latest_month}",
            PETROL_COLOR
        ),

        make_card(
            "LATEST DIESEL PRICE",
            format_ugx(diesel),
            f"Latest actual · {latest_month}",
            DIESEL_COLOR
        ),

        make_card(
            "PETROL MODEL ESTIMATE",
            format_ugx(petrol_prediction),
            petrol_label,
            PETROL_COLOR
        ),

        make_card(
            "DIESEL MODEL ESTIMATE",
            format_ugx(diesel_prediction),
            diesel_label,
            DIESEL_COLOR
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
                color=PETROL_COLOR,
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
                color=DIESEL_COLOR,
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
                        color=PARAFFIN_COLOR,
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

            marker_color=PETROL_COLOR

        )
    )

    fig.add_trace(
        go.Bar(

            x=df["month"],

            y=df["Diesel_Change"],

            name="Diesel change",

            marker_color=DIESEL_COLOR

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
            "color": PETROL_COLOR
        },

        "Diesel": {
            "column": "diesel_shs_per_litre",
            "color": DIESEL_COLOR
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
            forecasts["Fuel"].astype(str).str.lower()
            == fuel.lower()
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

                marker=dict(
                    size=10,
                    symbol="diamond"
                ),

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

        fuel = str(
            row.get("fuel", "Unknown")
        ).title()

        candidate = row.get(
            "candidate_mae",
            np.nan
        )

        baseline = row.get(
            "baseline_mae",
            np.nan
        )

        if pd.notna(candidate) and pd.notna(baseline):

            fig.add_trace(
                go.Bar(

                    x=[
                        "Candidate",
                        "Persistence baseline"
                    ],

                    y=[
                        candidate,
                        baseline
                    ],

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

    latest_month = (
        master["month"].max().to_period("M")
    )

    records = [

        {
            "Check": "Historical records",
            "Result": len(master),
            "Status": "Available"
        },

        {
            "Check": "First observation",
            "Result": str(
                master["month"].min().to_period("M")
            ),
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
            "Result": int(
                master["month"].duplicated().sum()
            ),
            "Status": (
                "Review"
                if master["month"].duplicated().any()
                else "Passed"
            )
        },

        {
            "Check": "Missing petrol values",
            "Result": int(
                master["petrol_shs_per_litre"].isna().sum()
            ),
            "Status": (
                "Review"
                if master["petrol_shs_per_litre"].isna().any()
                else "Passed"
            )
        },

        {
            "Check": "Missing diesel values",
            "Result": int(
                master["diesel_shs_per_litre"].isna().sum()
            ),
            "Status": (
                "Review"
                if master["diesel_shs_per_litre"].isna().any()
                else "Passed"
            )
        }

    ]

    if "paraffin_shs_per_litre" in master.columns:

        records.append({

            "Check": "Missing paraffin values",

            "Result": int(
                master["paraffin_shs_per_litre"].isna().sum()
            ),

            "Status": "Information"

        })

    return pd.DataFrame(records)


# ============================================================
# 10. FORECAST INTELLIGENCE
# ============================================================

def build_forecast_status(master, forecasts):

    if forecasts.empty:

        return """
        <div class="forecast-empty">
            No forecast output is available.
        </div>
        """

    current_period = pd.Timestamp.now().to_period("M")

    cards = []

    for _, row in forecasts.iterrows():

        fuel = str(row["Fuel"]).strip().title()

        actual = pd.to_numeric(
            row["Latest_Actual_Price"],
            errors="coerce"
        )

        predicted = pd.to_numeric(
            row["Predicted_Price"],
            errors="coerce"
        )

        forecast_period = pd.Period(
            str(row["Forecast_Month"]),
            freq="M"
        )

        month_label = forecast_period.strftime("%B %Y")

        # ----------------------------------------------------
        # CALCULATE PRICE MOVEMENT
        # ----------------------------------------------------

        if (
            pd.isna(actual)
            or pd.isna(predicted)
            or actual == 0
        ):

            change = np.nan
            percentage = np.nan

            direction = "Unavailable"
            arrow = "—"

            movement_class = "neutral"

        else:

            change = predicted - actual

            percentage = (
                change / actual
            ) * 100

            if change > 0:

                direction = "Increase"
                arrow = "↑"

                movement_class = "increase"

            elif change < 0:

                direction = "Decrease"
                arrow = "↓"

                movement_class = "decrease"

            else:

                direction = "No change"
                arrow = "→"

                movement_class = "neutral"

        # ----------------------------------------------------
        # FORECAST STATUS
        # ----------------------------------------------------

        if forecast_period < current_period:

            status = "HISTORICAL ESTIMATE"

            explanation = (
                "Historical model estimate — actual market "
                "price has not been verified."
            )

        elif forecast_period == current_period:

            status = "CURRENT-MONTH ESTIMATE"

            explanation = (
                "Model estimate — not an observed market price."
            )

        else:

            status = "FUTURE FORECAST"

            explanation = (
                "Predicted future market price — "
                "not an observed price."
            )

        fuel_class = fuel.lower()

        # ----------------------------------------------------
        # FORECAST CARD
        # ----------------------------------------------------

        cards.append(f"""

        <div class="forecast-card {fuel_class}">

            <div class="forecast-card-header">

                <div>

                    <div class="forecast-fuel">
                        {fuel}
                    </div>

                    <div class="forecast-month">
                        {month_label}
                    </div>

                </div>

                <span class="forecast-status-badge">
                    {status}
                </span>

            </div>


            <div class="forecast-prices">

                <div class="forecast-price-box actual-price">

                    <div class="forecast-label">
                        LATEST ACTUAL PRICE
                    </div>

                    <div class="forecast-price">
                        {format_ugx(actual)}
                    </div>

                </div>


                <div class="forecast-price-box estimated-price">

                    <div class="forecast-label">
                        MODEL ESTIMATE
                    </div>

                    <div class="forecast-price">
                        {format_ugx(predicted)}
                    </div>

                </div>

            </div>


            <div class="forecast-change {movement_class}">

                <div class="change-direction">

                    <span class="change-arrow">
                        {arrow}
                    </span>

                    <span>
                        {direction}
                    </span>

                </div>


                <div class="change-values">

                    <span class="change-amount">
                        {format_ugx(change)}
                    </span>

                    <span class="change-percentage">
                        ({format_percentage(percentage)})
                    </span>

                </div>

            </div>


            <div class="forecast-explanation">
                {explanation}
            </div>

        </div>

        """)

    return (
        '<div class="forecast-intelligence-grid">'
        + "".join(cards)
        + "</div>"
    )


# ============================================================
# 11. DASHBOARD DATA REFRESH
# ============================================================

def refresh_dashboard():

    master = load_master()

    forecasts = load_forecast()

    history = load_promotion_history()

    # --------------------------------------------------------
    # BUILD DASHBOARD COMPONENTS
    # --------------------------------------------------------

    kpis = build_kpis(
        master,
        forecasts
    )

    market = build_market_chart(
        master
    )

    movement = build_movement_chart(
        master
    )

    forecast_chart = build_forecast_chart(
        master,
        forecasts
    )

    forecast_status = build_forecast_status(
        master,
        forecasts
    )

    model_chart, model_table = build_model_performance(
        history
    )

    quality = build_data_quality(
        master
    )

    logs = "\n".join(
        load_pipeline_logs()
    )

    # --------------------------------------------------------
    # FORECAST TABLE WITH AUTOMATIC CHANGES
    # --------------------------------------------------------

    if not forecasts.empty:

        forecast_table = forecasts.copy()

        forecast_table["Price_Change_UGX"] = (
            forecast_table["Predicted_Price"]
            - forecast_table["Latest_Actual_Price"]
        )

        forecast_table["Percentage_Change"] = np.where(

            forecast_table["Latest_Actual_Price"] != 0,

            (
                forecast_table["Price_Change_UGX"]
                / forecast_table["Latest_Actual_Price"]
            ) * 100,

            np.nan

        )

        forecast_table["Movement"] = np.select(

            [
                forecast_table["Price_Change_UGX"] > 0,
                forecast_table["Price_Change_UGX"] < 0
            ],

            [
                "Increase",
                "Decrease"
            ],

            default="No change"

        )

        current_period = pd.Timestamp.now().to_period("M")

        forecast_table["Forecast_Status"] = (
            forecast_table["Forecast_Month"].apply(

                lambda value: (

                    "Historical estimate — unverified"

                    if pd.Period(
                        str(value),
                        freq="M"
                    ) < current_period

                    else (

                        "Current-month estimate — not actual"

                        if pd.Period(
                            str(value),
                            freq="M"
                        ) == current_period

                        else "Future forecast — not actual"

                    )

                )

            )
        )

    else:

        forecast_table = pd.DataFrame(

            columns=[

                "Fuel",
                "Forecast_Month",
                "Latest_Actual_Price",
                "Predicted_Price",
                "Price_Change_UGX",
                "Percentage_Change",
                "Movement",
                "Forecast_Status"

            ]

        )

    # --------------------------------------------------------
    # RETURN ALL DASHBOARD OUTPUTS
    # --------------------------------------------------------

    return (

        kpis,

        market,

        movement,

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
    overflow-x: hidden !important;
}

.plot-container,
.js-plotly-plot,
.plotly,
.svg-container {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
}

.tabs,
.tabitem,
.gradio-container,
.wrap {
    min-width: 0 !important;
}


/* ============================================================
   KPI CARDS
   ============================================================ */

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


/* ============================================================
   GENERAL HEADINGS
   ============================================================ */

.section-heading {
    color: #F8FAFC;
    font-size: 21px;
    font-weight: 700;
    margin: 12px 0;
}

.app-header {
    background: linear-gradient(
        120deg,
        #111C2E,
        #172A46
    );

    border: 1px solid #263449;
    border-radius: 15px;
    padding: 24px;
    margin-bottom: 18px;

    position: relative;
    overflow: hidden;
}

/* Uganda flag stripe */

.app-header::after {
    content: "";
    display: block;
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 6px;

    background: linear-gradient(
        to right,
        #000000 0%,
        #000000 33.33%,
        #FCDC04 33.33%,
        #FCDC04 66.66%,
        #D90000 66.66%,
        #D90000 100%
    );
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


/* ============================================================
   FORECAST INTELLIGENCE
   ============================================================ */

.forecast-intelligence-grid {

    display: grid;

    grid-template-columns: repeat(
        2,
        minmax(0, 1fr)
    );

    gap: 18px;

    margin: 15px 0 25px 0;

}

.forecast-card {

    background: #111C2E;

    border: 1px solid #263449;

    border-radius: 15px;

    padding: 22px;

    position: relative;

    overflow: hidden;

}

.forecast-card.petrol {

    border-top: 4px solid #D90000;

}

.forecast-card.diesel {

    border-top: 4px solid #FCDC04;

}

.forecast-card-header {

    display: flex;

    justify-content: space-between;

    align-items: flex-start;

    gap: 12px;

    margin-bottom: 22px;

}

.forecast-fuel {

    font-size: 22px;

    font-weight: 800;

    color: #F8FAFC;

}

.forecast-month {

    color: #94A3B8;

    font-size: 13px;

    margin-top: 4px;

}

.forecast-status-badge {

    background: #FCDC04;

    color: #000000;

    font-size: 10px;

    font-weight: 800;

    padding: 7px 10px;

    border-radius: 5px;

    text-align: center;

}

.forecast-prices {

    display: grid;

    grid-template-columns: 1fr 1fr;

    gap: 12px;

}

.forecast-price-box {

    padding: 15px;

    border-radius: 9px;

    background: #0B1220;

    min-width: 0;

}

.actual-price {

    border: 1px solid #475569;

}

.estimated-price {

    border: 1px dashed #FCDC04;

}

.forecast-label {

    font-size: 10px;

    font-weight: 800;

    color: #94A3B8;

    letter-spacing: 0.7px;

    margin-bottom: 10px;

}

.forecast-price {

    font-size: clamp(17px, 2vw, 23px);

    font-weight: 800;

    color: #F8FAFC;

    overflow-wrap: anywhere;

}


/* ============================================================
   FORECAST PRICE MOVEMENT
   ============================================================ */

.forecast-change {

    margin-top: 18px;

    padding: 15px;

    border-radius: 9px;

    display: flex;

    justify-content: space-between;

    align-items: center;

    flex-wrap: wrap;

    gap: 10px;

}

.forecast-change.increase {

    background: rgba(217, 0, 0, 0.13);

    border: 1px solid #D90000;

    color: #FF7777;

}

.forecast-change.decrease {

    background: rgba(34, 197, 94, 0.10);

    border: 1px solid #22C55E;

    color: #4ADE80;

}

.forecast-change.neutral {

    background: rgba(148, 163, 184, 0.10);

    border: 1px solid #64748B;

    color: #CBD5E1;

}

.change-direction {

    font-weight: 800;

    font-size: 15px;

}

.change-arrow {

    font-size: 22px;

    margin-right: 5px;

}

.change-values {

    text-align: right;

    font-weight: 800;

}

.change-amount {

    font-size: 17px;

}

.change-percentage {

    display: block;

    font-size: 12px;

    margin-top: 3px;

}

.forecast-explanation {

    color: #94A3B8;

    font-size: 11px;

    margin-top: 15px;

}

.forecast-empty {

    background: #111C2E;

    border: 1px solid #263449;

    border-radius: 12px;

    padding: 20px;

    color: #FCDC04;

}


/* ============================================================
   RESPONSIVE DESIGN
   ============================================================ */

@media(max-width: 850px) {

    .metric-grid {

        grid-template-columns: repeat(
            2,
            minmax(0, 1fr)
        );

    }

}

@media(max-width: 700px) {

    .forecast-intelligence-grid {

        grid-template-columns: 1fr;

    }

    .forecast-card-header {

        flex-direction: column;

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
    title="Ugandan Fuel Prediction"
) as demo:

    # --------------------------------------------------------
    # APPLICATION HEADER
    # --------------------------------------------------------

    gr.HTML("""

    <div class="app-header">

        <h1>
            UGANDAN FUEL PREDICTION
        </h1>

        <p>
            NATIONAL FUEL MARKET ANALYTICS AND FORECASTING INTELLIGENCE
        </p>

    </div>

    """)

    # --------------------------------------------------------
    # DASHBOARD CONTROLS
    # --------------------------------------------------------

    with gr.Row():

        gr.Markdown(
            "**MARKET INTELLIGENCE / OVERVIEW**"
        )

        refresh_button = gr.Button(
            "↻ Refresh Dashboard",
            variant="primary"
        )

    # ========================================================
    # OVERVIEW TAB
    # ========================================================

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

    # ========================================================
    # MARKET ANALYSIS TAB
    # ========================================================

    with gr.Tab("Market Analysis"):

        gr.Markdown(
            "## Historical Fuel Market"
        )

        market_chart = gr.Plot()

        movement_chart = gr.Plot()

    # ========================================================
    # FORECAST INTELLIGENCE TAB
    # ========================================================

    with gr.Tab("Forecast Intelligence"):

        gr.Markdown(
            "## Fuel Price Forecasts"
        )

        # HTML allows custom forecast cards and indicators.

        forecast_status = gr.HTML()

        forecast_chart = gr.Plot()

        forecast_table = gr.Dataframe(

            label="Latest Forecast Records",

            interactive=False

        )

    # ========================================================
    # MODEL PERFORMANCE TAB
    # ========================================================

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

    # ========================================================
    # DATA AND SYSTEM HEALTH TAB
    # ========================================================

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

    # --------------------------------------------------------
    # OUTPUT COMPONENTS
    # --------------------------------------------------------

    outputs = [

        kpi_output,

        overview_market,

        overview_movement,

        market_chart,

        movement_chart,

        forecast_chart,

        forecast_table,

        forecast_status,

        model_chart,

        model_table,

        quality_table,

        pipeline_logs

    ]

    # --------------------------------------------------------
    # INITIAL DASHBOARD LOAD
    # --------------------------------------------------------

    demo.load(

        fn=refresh_dashboard,

        inputs=[],

        outputs=outputs

    )

    # --------------------------------------------------------
    # MANUAL REFRESH
    # --------------------------------------------------------

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

    print("UGANDAN FUEL PREDICTION")

    print("=" * 60)

    print("Starting Gradio dashboard...")

    demo.launch(

        server_name="127.0.0.1",

        share=False,

        show_error=True,

        theme=gr.themes.Soft(

            primary_hue="red",

            secondary_hue="yellow",

            neutral_hue="slate"

        ),

        css=CSS

    )