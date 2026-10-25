# Healthcare Affordability Dashboard · Streamlit

An interactive presentation of the saved results in `DBA3803_RFModel.ipynb`.
The dashboard never executes, retrains or edits the notebook.

## Start

Open the VS Code terminal in the repository root (`DBA3803-Grp12`) and run:

```sh
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser if it does not open automatically.
Keep the terminal running while using the dashboard. Press **Ctrl+C** to stop it.
Use the same Python environment for both commands. If the dependencies are
already installed, only the second command is necessary.

The dashboard dependencies are separate from the original notebook's
`requirements.txt`; the notebook's model environment and pinned requirements
are unchanged. For an isolated installation, use a separate environment:

```sh
python -m venv .venv-dashboard
source .venv-dashboard/bin/activate
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

## Explore

- **Outreach overview:** choose a contact capacity and model; see case counts,
  recall, precision, lift and interactive charts.
- **Model comparison:** compare methods at the same capacity, each model's
  separate F2 rule, predictor importance and overall metrics.
- **Population profiles:** inspect both profiles within the fixed 20% Random
  Forest outreach scenario.
- **About the analysis:** review evaluation years, limitations, weighted
  prevalence and source consistency checks.
- **Export all capacity results:** download all four methods at all six evaluated
  capacities as CSV.

## Refresh results

Save the notebook and click **Reload saved notebook** or refresh the browser.
The app reads its existing saved outputs directly; no notebook execution,
HTML build or manual export is needed. Updated file contents invalidate the
results cache automatically.

The loader requires saved execution records and no saved errors, reconciles
case and contact counts and displayed metrics, verifies that Parts 8.1 and 8.3
agree on counts and overall forest metrics, compares fixed cutoffs, verifies the
profile pool size and requires all notebook comparison checks to pass.
Incomplete or inconsistent saved results produce a clear error instead of a
stale dashboard.

Execution counters that are out of notebook order are disclosed under **About
the analysis → Source and consistency checks**. The current saved sequence is
34, 2–33; it can be displayed when the result checks pass. This does not prove a
clean sequential run. Missing or duplicate counters and saved execution errors
still block loading; no counters or notebook outputs are changed.

Only aggregate results are read. Capacity recall, precision and lift are derived
from saved integer counts; other statistics use displayed notebook precision.
The source fingerprint and load time appear under **About the analysis**.

Profiles stay at the 20% Random Forest scenario when sidebar controls change.
Characteristic percentages retain missing responses in the profile denominator,
matching the notebook annotations. The selected model name and cutoffs come
from the saved selection output; the current run retains the original one-hot
reference, so saved original/revised F2 labels describe the same specification.
Random targeting uses the notebook's fixed random tie order; expected random
yield is labelled separately. Model evaluation is unweighted; population
prevalence is explicitly survey weighted. The evaluation is described as
out-of-time, with the notebook's prior examination of 2025 acknowledged.

## Files

- `app.py`: Streamlit layout, controls, charts and CSV download.
- `notebook_data.py`: read-only table extraction and consistency checks.
- `requirements.txt`: dashboard dependencies.
- `../.streamlit/config.toml`: dashboard theme and disabled usage telemetry.

The previous static HTML dashboard and its publishing configuration were removed.
