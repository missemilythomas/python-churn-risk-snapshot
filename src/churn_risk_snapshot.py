from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "users.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def to_snake_case(name: str) -> str:
    """Convert column names to snake_case."""
    name = name.strip().lower()
    name = re.sub(r"[^\w\s]", "", name)     # remove punctuation/brackets
    name = re.sub(r"\s+", "_", name)        # spaces -> underscores
    return name


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]
    return df


def map_churn_flag(df: pd.DataFrame, churn_col: str = "churn_status_yesno") -> pd.DataFrame:
    """Map Yes/No churn column to 1/0."""
    df = df.copy()

    if churn_col not in df.columns:
        raise KeyError(
            f"Expected churn column '{churn_col}' not found. "
            f"Columns available: {list(df.columns)[:15]}..."
        )

    df[churn_col] = df[churn_col].astype(str).str.strip().str.lower()
    df["churn_flag"] = df[churn_col].map({"yes": 1, "no": 0})

    # Basic validation
    if df["churn_flag"].isna().any():
        bad_values = df.loc[df["churn_flag"].isna(), churn_col].value_counts().head(10)
        raise ValueError(
            "Found unexpected churn values after mapping. Examples:\n"
            f"{bad_values}"
        )

    return df


def churn_by_payment(df: pd.DataFrame, payment_col: str = "payment_history_ontimedelayed") -> pd.DataFrame:
    if payment_col not in df.columns:
        raise KeyError(
            f"Expected payment column '{payment_col}' not found. "
            f"Columns available: {list(df.columns)[:15]}..."
        )

    df = df.copy()
    df[payment_col] = df[payment_col].astype(str).str.strip()

    summary = (
        df.groupby(payment_col, dropna=False)
        .agg(
            users=("churn_flag", "size"),
            churned=("churn_flag", "sum"),
        )
        .reset_index()
        .rename(columns={payment_col: "payment_status"})
    )
    summary["churn_rate"] = (summary["churned"] / summary["users"]).round(3)
    summary = summary.sort_values("churn_rate", ascending=False)

    return summary


def save_chart(summary: pd.DataFrame, outpath: Path) -> None:
    """Bar chart of churn rate by payment status."""
    plt.figure()
    plt.bar(summary["payment_status"], summary["churn_rate"])
    plt.title("Churn rate by payment status")
    plt.xlabel("Payment status")
    plt.ylabel("Churn rate")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found at {DATA_PATH}. "
            "Place your dataset in the data/ folder and name it users.csv"
        )

    df = pd.read_csv(DATA_PATH)

    # Cleaning / standardisation
    df = standardise_columns(df)

    # Map churn values
    df = map_churn_flag(df, churn_col="churn_status_yesno")

    # KPI table
    summary = churn_by_payment(df, payment_col="payment_history_ontimedelayed")

    # Save outputs
    summary.to_csv(OUTPUTS_DIR / "churn_by_payment.csv", index=False)
    save_chart(summary, OUTPUTS_DIR / "churn_by_payment.png")

    print("Saved outputs:")
    print(f"- {OUTPUTS_DIR / 'churn_by_payment.csv'}")
    print(f"- {OUTPUTS_DIR / 'churn_by_payment.png'}")
    print("\nPreview:")
    print(summary)


if __name__ == "__main__":
    main()
