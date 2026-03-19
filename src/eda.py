"""
eda.py
======
Exploratory Data Analysis for Customer Churn Prediction dataset.
Generates 4 publication-quality visualisations.

Author : Sourav Nayak
GitHub : github.com/SOURAV033
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

BG      = "#F8F9FA"
CHURN   = "#E05B5B"
STAY    = "#0A7B8C"
PURPLE  = "#5B3FD4"
GOLD    = "#F5A623"
GRID    = "#E8E8E8"


def load_and_clean(path="data/telco_churn.csv"):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn_Flag"]   = (df["Churn"] == "Yes").astype(int)
    return df


def plot_eda_overview(df):
    """Fig 1 — overview: churn rate, tenure dist, monthly charges by churn."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # 1a — Churn Donut
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["Churn"].value_counts()
    colors = [STAY, CHURN]
    wedges, texts, autotexts = ax1.pie(
        counts, labels=None, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.7,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2))
    for at in autotexts:
        at.set_fontsize(13); at.set_fontweight("bold"); at.set_color("white")
    ax1.legend(["Retained", "Churned"], loc="lower center",
               ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.08))
    ax1.set_title("Overall Churn Rate", fontweight="bold", fontsize=11, pad=10)

    # 1b — Tenure distribution by churn
    ax2 = fig.add_subplot(gs[0, 1])
    for label, color in [("No", STAY), ("Yes", CHURN)]:
        subset = df[df["Churn"] == label]["tenure"]
        ax2.hist(subset, bins=24, alpha=0.65, color=color,
                 label=f"{'Retained' if label=='No' else 'Churned'}", edgecolor="white")
    ax2.set_xlabel("Tenure (months)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Tenure Distribution by Churn", fontweight="bold", fontsize=11)
    ax2.legend(fontsize=9); ax2.set_facecolor(BG)
    ax2.grid(axis="y", color=GRID, linewidth=0.7)
    ax2.spines[["top","right"]].set_visible(False)

    # 1c — Monthly charges violin
    ax3 = fig.add_subplot(gs[0, 2])
    sns.violinplot(data=df, x="Churn", y="MonthlyCharges",
                   palette={"No": STAY, "Yes": CHURN},
                   inner="quartile", ax=ax3, linewidth=1.2)
    ax3.set_xticklabels(["Retained","Churned"])
    ax3.set_title("Monthly Charges by Churn", fontweight="bold", fontsize=11)
    ax3.set_xlabel(""); ax3.set_ylabel("Monthly Charges (₹)", fontsize=10)
    ax3.set_facecolor(BG); ax3.spines[["top","right"]].set_visible(False)
    ax3.grid(axis="y", color=GRID, linewidth=0.7)

    # 1d — Contract type churn rate
    ax4 = fig.add_subplot(gs[1, 0])
    ct = df.groupby("Contract")["Churn_Flag"].mean().reset_index()
    ct["pct"] = ct["Churn_Flag"] * 100
    bars = ax4.bar(ct["Contract"], ct["pct"],
                   color=[CHURN, GOLD, STAY], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, ct["pct"]):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax4.set_title("Churn Rate by Contract Type", fontweight="bold", fontsize=11)
    ax4.set_ylabel("Churn Rate (%)", fontsize=10)
    ax4.set_xticklabels(ct["Contract"], rotation=12, fontsize=9)
    ax4.set_facecolor(BG); ax4.grid(axis="y", color=GRID, linewidth=0.7)
    ax4.spines[["top","right"]].set_visible(False)

    # 1e — Internet service churn
    ax5 = fig.add_subplot(gs[1, 1])
    it = df.groupby("InternetService")["Churn_Flag"].mean().reset_index()
    it["pct"] = it["Churn_Flag"] * 100
    bars2 = ax5.bar(it["InternetService"], it["pct"],
                    color=[PURPLE, CHURN, STAY], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars2, it["pct"]):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax5.set_title("Churn Rate by Internet Service", fontweight="bold", fontsize=11)
    ax5.set_ylabel("Churn Rate (%)", fontsize=10)
    ax5.set_facecolor(BG); ax5.grid(axis="y", color=GRID, linewidth=0.7)
    ax5.spines[["top","right"]].set_visible(False)

    # 1f — Payment method churn
    ax6 = fig.add_subplot(gs[1, 2])
    pm = df.groupby("PaymentMethod")["Churn_Flag"].mean().reset_index().sort_values("Churn_Flag")
    pm["pct"] = pm["Churn_Flag"] * 100
    colors6 = [STAY, PURPLE, GOLD, CHURN]
    bars3 = ax6.barh(pm["PaymentMethod"], pm["pct"],
                     color=colors6, edgecolor="white", linewidth=1)
    for bar, val in zip(bars3, pm["pct"]):
        ax6.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax6.set_title("Churn Rate by Payment Method", fontweight="bold", fontsize=11)
    ax6.set_xlabel("Churn Rate (%)", fontsize=10)
    ax6.set_facecolor(BG); ax6.grid(axis="x", color=GRID, linewidth=0.7)
    ax6.spines[["top","right"]].set_visible(False)
    ax6.set_yticklabels(pm["PaymentMethod"], fontsize=8)

    fig.suptitle("Customer Churn EDA — Key Insights", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.savefig("outputs/eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] outputs/eda_overview.png")


def plot_correlation_heatmap(df):
    """Fig 2 — correlation heatmap of numeric features."""
    num_df = df.select_dtypes(include="number").drop(columns=["Churn_Flag"], errors="ignore")
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", linewidths=0.5,
                cmap=sns.diverging_palette(250, 10, as_cmap=True),
                center=0, square=True, mask=False,
                annot_kws={"size": 9})
    ax.set_title("Correlation Heatmap — Numeric Features",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] outputs/correlation_heatmap.png")


def print_summary(df):
    print("\n── Dataset Summary ──────────────────────────────")
    print(f"  Rows            : {len(df):,}")
    print(f"  Columns         : {df.shape[1]}")
    print(f"  Churn rate      : {df['Churn_Flag'].mean()*100:.1f}%")
    print(f"  Avg tenure      : {df['tenure'].mean():.1f} months")
    print(f"  Avg monthly     : ₹{df['MonthlyCharges'].mean():.2f}")
    print(f"  Avg total       : ₹{df['TotalCharges'].mean():.2f}")
    print(f"  Missing values  : {df.isnull().sum().sum()}")
    print("─────────────────────────────────────────────────\n")


if __name__ == "__main__":
    df = load_and_clean()
    print_summary(df)
    plot_eda_overview(df)
    plot_correlation_heatmap(df)
    print("[DONE] EDA complete — check outputs/ folder")
