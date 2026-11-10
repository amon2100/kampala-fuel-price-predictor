# Kampala Fuel Price Predictor

An analytics and forecasting project for monitoring retail fuel prices in
Kampala, Uganda. It combines historical fuel prices with relevant market
indicators, produces petrol and diesel forecasts, and presents the results in
an interactive dashboard.

The project is intended as a practical, data-driven view of fuel-market trends.
Forecasts are estimates and should not be treated as guaranteed prices or
financial advice.

## What it does

- Maintains a processed master dataset of historical fuel prices.
- Uses source data on Brent crude prices, Uganda fuel prices, fuel taxes, and
  USD/UGX exchange rates.
- Checks incoming data for validity and historical revisions before updating
  the master dataset.
- Trains candidate petrol and diesel models when the master dataset changes and
  evaluates candidates before promoting models.
- Generates forecasts from the production models and verifies the forecast
  output.
- Provides a Gradio dashboard for price trends, forecasts, data quality,
  forecast status, and model-promotion history.

## Project structure

```text
app/                 Gradio dashboard
data/
  incoming/          Data files awaiting processing
  processed/         Master and engineered datasets
  raw/               Source datasets
models/              Production and candidate models
notebooks/           Exploratory analysis and outputs
outputs/             Forecasts, promotion history, and pipeline logs
src/                 Data, modeling, forecasting, and pipeline modules
requirements.txt     Python dependencies
```

## Data sources

The repository includes datasets covering:

- Weekly Brent crude oil prices
- Uganda fuel prices published by UBOS
- Uganda fuel taxes
- USD/UGX exchange rates

## Getting started

Install the dependencies using the Python interpreter you intend to use:

```powershell
python -m pip install -r requirements.txt
```

### Run the dashboard

From the repository root, start the dashboard with:

```powershell
python .\app\app.py
```

Open the local URL printed in the terminal. If the default port is busy,
Gradio will try another available port.

### Run the data and forecasting pipeline

From the repository root, run:

```powershell
python .\src\pipeline.py
```

The pipeline checks for incoming data, validates and updates the master
dataset, conditionally trains and evaluates candidate models when data changes,
verifies the production models, and generates and checks the latest forecast.
Execution details are written to `outputs/pipeline_runs.log`.

### Explore the notebook

Open `notebooks/kampala-fuel-price-predictor.ipynb` in Jupyter or Visual Studio
Code. Ensure the notebook uses the same Python environment where the project
dependencies were installed.
