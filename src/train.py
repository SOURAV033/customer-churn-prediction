"""
train.py
========
Train, evaluate, and compare multiple ML models for churn prediction.
Models: Logistic Regression, Random Forest, Gradient Boosting, SVM
Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC

Author : Sourav Nayak
GitHub : github.com/SOURAV033
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os, pickle
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

from preprocess import load_data, clean_data, engineer_features, encode_features

# ── Colour palette ─────────────────────────────────────────────
PALETTE = {
    "Logistic Regression":    "#5B3FD4",
    "Random Forest":          "#0A7B8C",
    "Gradient Boosting":      "#E05B5B",
    "SVM":                    "#F5A623",
}
BG      = "#F8F9FA"
GRID    = "#E0E0E0"

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42,
            class_weight="balanced", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42),
        "SVM": SVC(
            kernel="rbf", probability=True,
            random_state=42, class_weight="balanced"),
    }


def evaluate_model(model, X_test, y_test, name):
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),    4),
        "Precision": round(precision_score(y_test, y_pred),   4),
        "Recall":    round(recall_score(y_test, y_pred),      4),
        "F1 Score":  round(f1_score(y_test, y_pred),          4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob),     4),
    }, y_pred, y_prob


def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    x       = np.arange(len(metrics))
    width   = 0.18
    models  = results_df["Model"].tolist()

    for i, model_name in enumerate(models):
        vals   = results_df[results_df["Model"] == model_name][metrics].values.flatten()
        offset = (i - len(models)/2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=model_name,
                        color=PALETTE[model_name], alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, color="#333333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.60, 1.02)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison — Customer Churn Prediction",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.spines[["top","right","left"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] outputs/model_comparison.png")


def plot_confusion_matrices(models_dict, X_test, y_test):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.02)

    for ax, (name, model) in zip(axes, models_dict.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap=sns.light_palette(PALETTE[name], as_cmap=True),
                    linewidths=1, linecolor="white",
                    xticklabels=["No Churn","Churn"],
                    yticklabels=["No Churn","Churn"],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(name, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual",    fontsize=9)
        ax.set_facecolor(BG)

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] outputs/confusion_matrices.png")


def plot_roc_curves(models_dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.grid(color=GRID, linewidth=0.8)

    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc  = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=PALETTE[name], linewidth=2.2,
                label=f"{name}  (AUC = {auc:.3f})")

    ax.plot([0,1],[0,1],"--", color="#AAAAAA", linewidth=1.5, label="Random Classifier")
    ax.fill_between([0,1],[0,1], alpha=0.04, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] outputs/roc_curves.png")


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    colors = [PALETTE["Random Forest"] if v > fi_df["Importance"].median()
              else "#B0C4DE" for v in fi_df["Importance"]]
    bars = ax.barh(fi_df["Feature"], fi_df["Importance"],
                   color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, fi_df["Importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8, color="#333333")

    ax.set_xlabel("Feature Importance Score", fontsize=11)
    ax.set_title("Top 15 Feature Importances — Random Forest",
                 fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.spines[["top","right","bottom"]].set_visible(False)

    high = mpatches.Patch(color=PALETTE["Random Forest"], label="Above median importance")
    low  = mpatches.Patch(color="#B0C4DE",                label="Below median importance")
    ax.legend(handles=[high, low], fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] outputs/feature_importance.png")


def main():
    print("=" * 55)
    print(" Customer Churn Prediction — ML Training Pipeline")
    print("=" * 55)

    # Load & preprocess
    df      = load_data("data/telco_churn.csv")
    df      = clean_data(df)
    df      = engineer_features(df)
    X_train, X_test, y_train, y_test, features, scaler = encode_features(df)

    # Train all models
    models  = get_models()
    results = []
    trained = {}

    for name, model in models.items():
        print(f"\n[TRAIN] {name}...")
        model.fit(X_train, y_train)
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        trained[name] = model
        print(f"        F1={metrics['F1 Score']:.4f}  ROC-AUC={metrics['ROC-AUC']:.4f}")

    # Results table
    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    print("\n" + "=" * 55)
    print(" MODEL COMPARISON RESULTS")
    print("=" * 55)
    print(results_df.to_string(index=False))

    best_name  = results_df.iloc[0]["Model"]
    best_model = trained[best_name]
    print(f"\n[BEST] {best_name}  —  ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, best_model.predict(X_test),
                                 target_names=["No Churn","Churn"]))

    # Save plots
    plot_model_comparison(results_df)
    plot_confusion_matrices(trained, X_test, y_test)
    plot_roc_curves(trained, X_test, y_test)

    rf_model = trained["Random Forest"]
    plot_feature_importance(rf_model, features)

    # Save best model
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler,
                     "features": features, "model_name": best_name}, f)
    print(f"[SAVED] models/best_model.pkl")

    # Save results CSV
    results_df.to_csv("outputs/model_results.csv", index=False)
    print("[SAVED] outputs/model_results.csv")
    print("\n[DONE] All training complete!")
    return results_df, trained, X_test, y_test, features


if __name__ == "__main__":
    main()
