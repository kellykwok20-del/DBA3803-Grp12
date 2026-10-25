"""Read and validate saved notebook tables without executing or editing the notebook."""

from datetime import datetime
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parent / "DBA3803_RFModel.ipynb"


class TableReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = None
        self.section = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = {"head": [], "body": []}
        elif self.table is not None:
            if tag in ("thead", "tbody"):
                self.section = "head" if tag == "thead" else "body"
            elif tag == "tr":
                self.row = []
            elif tag in ("td", "th") and self.row is not None:
                self.cell = [tag, ""]

    def handle_data(self, text):
        if self.cell is not None:
            self.cell[1] += text

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append((self.cell[0], self.cell[1].strip()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.section:
                self.table[self.section].append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None
            self.section = None


def scalar(text):
    cleaned = text.replace(",", "").strip()
    if cleaned in ("NaN", "nan", "", "—"):
        return None
    if cleaned in ("True", "False"):
        return cleaned == "True"
    try:
        if cleaned.endswith("%"):
            return float(cleaned[:-1]) / 100
        number = float(cleaned)
        return int(number) if number.is_integer() else number
    except ValueError:
        return text


def cell_tables(cells, cell_id):
    result = []
    for output in cells[cell_id].get("outputs", []):
        markup = output.get("data", {}).get("text/html")
        if not markup:
            continue
        parser = TableReader()
        parser.feed("".join(markup))
        for table in parser.tables:
            headers = [text for _, text in table["head"][0] if text]
            if table["body"]:
                width = sum(tag == "td" for tag, _ in table["body"][0])
                if len(headers) == width + 1:
                    # A named DataFrame index is a row heading, not a value column.
                    headers = headers[1:]
            records = []
            for row in table["body"]:
                values = [scalar(text) for tag, text in row if tag == "td"]
                if len(values) != len(headers):
                    raise ValueError(f"Unexpected table layout in {cell_id}.")
                records.append(dict(zip(headers, values)))
            result.append(records)
    return result


def select_table(tables, *columns):
    matches = [rows for rows in tables if rows and set(columns) <= rows[0].keys()]
    if len(matches) != 1:
        raise ValueError(f"Expected one saved table with {columns}; found {len(matches)}.")
    return matches[0]


def extract(raw):
    notebook = json.loads(raw)
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code]
    if any(type(count) is not int or count <= 0 for count in counts):
        raise ValueError("The notebook has cells without saved execution records.")
    if len(counts) != len(set(counts)):
        raise ValueError("The notebook has duplicate execution counters; review the saved results.")
    # Counters alone cannot prove that saved tables are consistent. For example,
    # rerunning imports moves their counter after the results without changing
    # those results. Report ordering separately and reconcile the saved tables.
    if any(out.get("output_type") == "error" for cell in code for out in cell.get("outputs", [])):
        raise ValueError("The notebook contains a saved execution error. Review the saved results before displaying them.")
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    evaluation_tables = cell_tables(cells, "54ee7e43f757")
    benchmark_tables = cell_tables(cells, "rfcomp-later-year-comparison")
    sample = select_table(evaluation_tables, "adults", "reported_cases")[0]
    return raw, notebook, cells, evaluation_tables, benchmark_tables, sample


def validate_counts(row, adults, cases, contact_key):
    """Check saved integer counts and metrics, allowing displayed rounding."""
    keys = ["cases_found", "false_positives", "missed_cases", "true_negatives", contact_key]
    if any(type(row.get(key)) is not int or row[key] < 0 for key in keys):
        raise ValueError("Saved outreach counts must be non-negative integers.")
    tp, fp, fn, tn, contacts = (row[key] for key in keys)
    if tp + fn != cases or tp + fp != contacts or tp + fp + fn + tn != adults:
        raise ValueError("Saved outreach counts do not reconcile.")
    denominator = 5 * tp + 4 * fn + fp
    metrics = {"precision": tp / contacts if contacts else 0,
               "recall": tp / cases if cases else 0,
               "F2": 5 * tp / denominator if denominator else 0}
    for key, value in metrics.items():
        # Saved percentages have one decimal place; other metrics have four.
        if key in row and abs(row[key] - value) > 0.0006:
            raise ValueError(f"Saved {key} does not match the outreach counts.")


def load_results(raw=None):
    raw = NOTEBOOK.read_bytes() if raw is None else raw
    raw, notebook, cells, evaluation_tables, benchmark_tables, sample = extract(raw)
    capacity_tables = [rows for rows in benchmark_tables if rows and
                       {"method", "capacity", "true_negatives"} <= rows[0].keys()]
    capacities = max(capacity_tables, key=len)
    expected = next(rows for rows in evaluation_tables if rows and
                    rows[0].get("method") == "Random selection: expected")
    overall = select_table(benchmark_tables, "method", "AP", "Brier")
    policies = select_table(evaluation_tables, "procedure", "flagged")
    fixed = select_table(benchmark_tables, "method", "cutoff", "selected_share")
    checks = select_table(benchmark_tables, "check", "passed")
    if not all(row["passed"] is True for row in checks):
        raise ValueError("A notebook comparison consistency check failed.")

    n, cases = sample["adults"], sample["reported_cases"]
    methods = {
        "Existing random forest": "rf",
        "Full-feature logistic": "lr",
        "Insurance only": "insurance",
        "Constant score / random ties": "random",
    }
    for row in capacities:
        row["id"] = methods[row["method"]]
        validate_counts(row, n, cases, "contacts")
        tp = row["cases_found"]
        row.update(recall=tp / cases, precision=tp / row["contacts"],
                   lift=(tp / row["contacts"]) / (cases / n))
    fractions = sorted(set(row["capacity"] for row in capacities))
    for fraction in fractions:
        group = [row for row in capacities if row["capacity"] == fraction]
        if len(group) != 4 or len({row["contacts"] for row in group}) != 1:
            raise ValueError("Each capacity must compare four methods at equal contact counts.")
    for table in capacity_tables:
        for row in table:
            validate_counts(row, n, cases, "contacts")
            comparison = next(r for r in capacities if r["method"] == row["method"]
                              and r["capacity"] == row["capacity"])
            if any(row[key] != comparison[key] for key in
                   ["contacts", "cases_found", "missed_cases", "false_positives", "true_negatives"]):
                raise ValueError("The saved benchmark capacity tables disagree.")
    original_capacities = next(rows for rows in evaluation_tables if rows and
                               rows[0].get("method") == "Random forest" and "capacity" in rows[0])
    original_methods = {"Random forest": "rf", "Insurance only": "insurance",
                        "Constant score / random tie order": "random"}
    for row in original_capacities:
        if row["method"] in original_methods:
            comparison = next(r for r in capacities if r["id"] == original_methods[row["method"]]
                              and r["capacity"] == row["capacity"])
            if any(row[k] != comparison[k] for k in ["contacts", "cases_found", "missed_cases", "false_positives"]):
                raise ValueError("The saved capacity results in Parts 8.1 and 8.3 disagree.")
    forest_metrics = next(row for row in overall if row["method"] == "Existing random forest")
    original_metrics = next(row for row in select_table(evaluation_tables, "method", "AP", "Brier")
                            if row["method"] == "Random forest")
    if any(abs(forest_metrics[key] - original_metrics[key]) > 0.000001
           for key in ["AP", "ROC_AUC", "Brier"]):
        raise ValueError("The saved overall forest metrics in Parts 8.1 and 8.3 disagree.")
    for row in policies + fixed:
        validate_counts(row, n, cases, "flagged")

    stream = "".join("".join(o.get("text", [])) for o in cells["2a670956"].get("outputs", []))
    selected = re.search(r"Selected forest: (.+)", stream).group(1)
    cutoff = float(re.search(r"Recall-first cutoff: ([\d.eE+-]+)", stream).group(1))
    conservative = float(re.search(r"Conservative cutoff: ([\d.eE+-]+)", stream).group(1))
    rf_fixed = next(row for row in fixed if row["method"] == "Existing random forest")
    if abs(rf_fixed["cutoff"] - cutoff) > 0.000001:
        raise ValueError("Development and evaluation cutoffs disagree.")
    rf_policy = next(row for row in policies if row["procedure"] == "Revised forest: F2 cutoff")
    if any(rf_policy[k] != rf_fixed[k] for k in ["cases_found", "missed_cases", "false_positives", "flagged"]):
        raise ValueError("The two final-evaluation policy tables disagree.")
    reference_retained = selected == "Original forest reference"
    if reference_retained:
        original_policy = next(row for row in policies if row["procedure"] == "Original forest: F2 cutoff")
        if any(original_policy[key] != rf_policy[key] for key in
               ["cases_found", "missed_cases", "false_positives", "true_negatives", "flagged"]):
            raise ValueError("The retained reference and selected forest policies disagree.")
    for row in overall + fixed:
        row["id"] = methods[row["method"]]

    profiles = select_table(cell_tables(cells, "31ba23cf98a9"), "profile", "median_age", "adults")
    if sum(row["adults"] for row in profiles) != next(r["contacts"] for r in capacities if r["id"] == "rf" and r["capacity"] == 0.2):
        raise ValueError("Profile population does not match the 20% outreach population.")
    counts = [cell["execution_count"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    return {
        "source": NOTEBOOK.name,
        "source_sha256": sha256(raw).hexdigest(),
        "loaded_at": datetime.now(ZoneInfo("Asia/Singapore")).strftime("%d %b %Y, %H:%M SGT"),
        "sample": sample, "capacities": capacities, "fractions": fractions,
        "random_expected": expected, "overall": overall, "policies": policies,
        "fixed": fixed, "selected_forest": selected, "cutoff": cutoff,
        "reference_retained": reference_retained,
        "execution_in_order": counts == sorted(counts),
        "execution_counts": counts,
        "conservative_cutoff": conservative, "profiles": profiles,
        "importance": select_table(cell_tables(cells, "28b3974f53af"), "predictor", "validation_AP_drop"),
        "prevalence": select_table(cell_tables(cells, "d9c0c6db01b0"), "year", "weighted_rate"),
        "checks": checks,
    }


if __name__ == "__main__":
    data = load_results()
    print(f"Verified {len(data['capacities'])} capacity rows from {data['source']}.")
    print(f"Notebook SHA-256: {data['source_sha256']}")
