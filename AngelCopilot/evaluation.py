from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

OVERALL_COL = "Overall Score (Weighted Average)"


def set_angelcopilot_style():
    sns.set_theme(
        style="whitegrid",
        context="talk",  # big enough for blog, not monstrous
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "regular",
            "axes.grid.axis": "y",  # only horizontal gridlines
            "grid.alpha": 0.2,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
        },
    )

    # Slightly smaller font than default 'talk'
    sns.set_context("talk", font_scale=0.9)


@dataclass
class TierComparisonResult:
    mean_a: float
    mean_b: float
    sd_a: float
    sd_b: float
    n_a: int
    n_b: int
    diff: float
    t_stat: float
    df: float
    p_value: float
    cohen_d: float


def load_evaluations(path: Path | str) -> pd.DataFrame:
    """
    Load evaluation data from CSV or Excel and do basic cleaning.

    Expected columns:
      - 'Tier'
      - 'Company Name'
      - 'Iteration'
      - OVERALL_COL (Overall Score (Weighted Average))
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=0)
    else:
        raise ValueError(f"Unsupported file extension for {path}")

    # Forward-fill Tier and Company Name so all iterations are labeled
    df["Tier"] = df["Tier"].ffill()
    df["Company Name"] = df["Company Name"].ffill()

    return df


def compute_company_level_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute company-level mean, std, and count for the overall score.

    Returns a DataFrame with:
      - Tier
      - Company Name
      - overall_mean
      - overall_std
      - n_evals
    """
    scored = df.dropna(subset=[OVERALL_COL]).copy()

    grouped = (
        scored.groupby(["Tier", "Company Name"])[OVERALL_COL]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "overall_mean",
                "std": "overall_std",
                "count": "n_evals",
            }
        )
    )

    return grouped


def compare_tiers(company_scores: pd.DataFrame) -> TierComparisonResult:
    """
    Run Welch's t-test comparing Tier A vs Tier B company-level means,
    and compute Cohen's d effect size.
    """
    a = company_scores.loc[company_scores["Tier"] == "A", "overall_mean"].to_numpy()
    b = company_scores.loc[company_scores["Tier"] == "B", "overall_mean"].to_numpy()

    if len(a) == 0 or len(b) == 0:
        raise ValueError("Need at least one Tier A and one Tier B company.")

    mean_a, mean_b = a.mean(), b.mean()
    sd_a, sd_b = a.std(ddof=1), b.std(ddof=1)
    n_a, n_b = len(a), len(b)
    diff = mean_a - mean_b

    # Welch t-test
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    # Effective degrees of freedom for Welch's t-test
    var_a, var_b = sd_a**2, sd_b**2
    se_a2, se_b2 = var_a / n_a, var_b / n_b
    df_num = (se_a2 + se_b2) ** 2
    df_den = (se_a2**2) / (n_a - 1) + (se_b2**2) / (n_b - 1)
    df = df_num / df_den

    # Cohen's d (pooled SD)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    cohen_d = diff / np.sqrt(pooled_var)

    return TierComparisonResult(
        mean_a=mean_a,
        mean_b=mean_b,
        sd_a=sd_a,
        sd_b=sd_b,
        n_a=n_a,
        n_b=n_b,
        diff=diff,
        t_stat=t_stat,
        df=df,
        p_value=p_value,
        cohen_d=cohen_d,
    )


# ---------- Plotting utilities ---------- #


def plot_tier_distributions(
    company_scores: pd.DataFrame,
) -> plt.Axes:
    """
    Overlapping histogram of company-level mean scores for Tier A and Tier B.
    Returns the matplotlib Axes object.
    """
    fig, ax = plt.subplots()

    a = company_scores.loc[company_scores["Tier"] == "A", "overall_mean"].to_numpy()
    b = company_scores.loc[company_scores["Tier"] == "B", "overall_mean"].to_numpy()

    all_scores = np.concatenate([a, b])
    bins = np.linspace(all_scores.min() - 0.05, all_scores.max() + 0.05, 10)

    ax.hist(a, bins=bins, alpha=0.6, density=True, label="Tier A")
    ax.hist(b, bins=bins, alpha=0.6, density=True, label="Tier B")

    ax.set_xlabel("Overall score (company-level mean, 1–5 scale)")
    ax.set_ylabel("Density")
    ax.set_title("AngelCopilot Score Distribution")
    ax.legend()

    fig.savefig("tier_distribution.png", dpi=300, bbox_inches="tight")

    return ax


def plot_robustness(
    company_scores: pd.DataFrame,
    min_evals: int = 5,
) -> plt.Axes | None:
    """
    Plot robustness: per-company mean ± 1 SD for companies
    with at least `min_evals` evaluations.

    Uses the output of `compute_company_level_scores`:
      - overall_mean
      - overall_std
      - n_evals
    """
    subset = company_scores[company_scores["n_evals"] >= min_evals].copy()
    if subset.empty:
        print(f"[robustness] No companies with >= {min_evals} evaluations.")
        return None

    subset = subset.sort_values("overall_mean", ascending=False)

    x = np.arange(len(subset))
    labels = subset["Company Name"].tolist()
    means = subset["overall_mean"].to_numpy()
    sds = subset["overall_std"].to_numpy()
    tiers = subset["Tier"].to_numpy()

    colors = np.where(tiers == "A", "tab:blue", "tab:orange")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(x, means, yerr=sds, fmt="none", ecolor="gray", capsize=4)
    ax.scatter(x, means, c=colors)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Overall score (mean ± 1 SD)")
    ax.set_title(f"Robustness of AngelCopilot scores (n ≥ {min_evals} evals/company)")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="tab:blue", label="Tier A"),
        plt.Line2D([0], [0], marker="o", linestyle="", color="tab:orange", label="Tier B"),
    ]
    ax.legend(handles=handles)

    fig.savefig("robustness_errorbars.png", dpi=300, bbox_inches="tight")

    return ax


def plot_sorted_company_scores(
    company_scores: pd.DataFrame,
) -> plt.Axes:
    """
    Plot all companies sorted by mean score, colored by tier.
    Horizontal bar chart with score annotations.
    """
    df = company_scores.copy()
    df = df.sort_values("overall_mean", ascending=True)

    df["pretty_name"] = df["Company Name"].str.replace("_", " ")

    means = df["overall_mean"].to_numpy()
    tiers = df["Tier"].to_numpy()
    labels = df["pretty_name"].tolist()

    colors = np.where(tiers == "A", "tab:blue", "tab:orange")

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, means, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Overall score (company-level mean)")
    ax.set_ylabel("")
    ax.set_title("Company-level AngelCopilot scores (sorted)")

    # Annotate scores
    for bar, value in zip(bars, means):
        ax.text(
            value + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    # Legend outside the plot on the right
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color="tab:blue", label="Tier A"),
        plt.Line2D([0], [0], marker="s", linestyle="", color="tab:orange", label="Tier B"),
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # ← fixed name
        frameon=True,
    )

    plt.tight_layout()
    fig.savefig("company_scores_sorted.png", dpi=300, bbox_inches="tight")

    return ax


def summarize_sd(robustness_df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Return (median_sd, mean_sd, max_sd) across companies with non-null SD.
    Useful for one-line reporting in the notebook.
    """
    sd_series = robustness_df["sd_score"].dropna()
    if sd_series.empty:
        return (float("nan"), float("nan"), float("nan"))

    return (
        float(sd_series.median()),
        float(sd_series.mean()),
        float(sd_series.max()),
    )
