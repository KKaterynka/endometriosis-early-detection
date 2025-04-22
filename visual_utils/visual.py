"""
This module contains functions to visualize model performance and feature importance.
"""

import seaborn as sns
import matplotlib.pyplot as plt


def plot_model_performance_vs_top_features(
    lr_f1s,
    rf_f1s,
    xgb_f1s,
    ada_f1s,
    mlp_f1s,
    tabpfn_f1s,
    model_importances,
    min_features=7,
):
    """Plots F1 scores of multiple models across different numbers of top features.

    Visualizes how model performance (F1 score) changes as the number of top features
    used for training increases. Highlights the optimal number of features for each model.

    Args:
        lr_f1s: F1 scores for Logistic Regression across feature counts
        rf_f1s: F1 scores for Random Forest across feature counts
        xgb_f1s: F1 scores for XGBoost across feature counts
        ada_f1s: F1 scores for ADABoost across feature counts
        mlp_f1s: F1 scores for MLP across feature counts
        tabpfn_f1s: F1 scores for TabPFN across feature counts
        model_importances: Name of the model used for SHAP feature importance
        min_features (optional): Minimum number of features to start from.
    """
    plt.figure(figsize=(30, 11))
    sns.set_style("white")

    num_features_range = range(min_features, len(lr_f1s) + min_features)
    colors = {
        "Random Forest": "#1f77b4",
        "XGBoost": "#2ca02c",
        "ADABoost": "#d62728",
        "Logistic Regression": "#9467bd",
        "MLP": "#ff7f0e",
        "TabPFN": "#17becf",
    }

    def plot_model_results(f1s, label):
        if not f1s:
            return

        best_idx = f1s.index(max(f1s))
        best_feature_count = num_features_range[best_idx]

        plt.plot(
            num_features_range,
            f1s,
            label=f"{label}",
            color=colors[label],
            linestyle="-",
            linewidth=2,
        )
        plt.scatter(
            best_feature_count,
            max(f1s),
            color=colors[label],
            s=100,
            edgecolor="black",
            zorder=5,
        )

    model_results = {
        "Random Forest": rf_f1s,
        "XGBoost": xgb_f1s,
        "ADABoost": ada_f1s,
        "Logistic Regression": lr_f1s,
        "MLP": mlp_f1s,
        "TabPFN": tabpfn_f1s,
    }

    for model, f1s in model_results.items():
        plot_model_results(f1s, model)

    plt.xlabel("Number of Top Features", fontsize=30)
    plt.ylabel("F1 Score", fontsize=30)
    plt.title(
        f"Model Performance vs. Top Features\n(SHAP by {model_importances})",
        fontsize=30,
        fontweight="bold",
    )
    plt.legend(loc="best", fontsize=25, frameon=True, fancybox=True)
    plt.grid(True, linestyle="--", alpha=0.3)

    ax = plt.gca()

    ax.tick_params(axis="x", labelsize=23)

    ax.tick_params(axis="y", labelsize=23)

    plt.show()


def visualize_normalized_shap_table(merged_table):
    """
    Visualizes the normalized SHAP feature importance scores using a heatmap.

    Args:
        merged_table: A dataframe containing normalized SHAP importance scores
                                     across multiple models, with features as the index.

    Displays:
        A heatmap showing feature importance scores for different models, with annotations.
    """

    plt.figure(figsize=(10, 14))

    ax = sns.heatmap(
        merged_table,
        annot=True,
        cmap=sns.cubehelix_palette(as_cmap=True),
        linewidths=0.5,
        fmt=".2f",
        annot_kws={"size": 8},
    )

    ax.set_yticklabels(merged_table.index, rotation=0, ha="right", fontsize=10)
    ax.set_ylabel("Feature", fontsize=12, labelpad=4)

    ax.set_xticklabels(
        ["Random Forest", "XGBoost", "AdaBoost", "MLP", "TabPFN", "Mean"],
        ha="center",
        fontsize=10,
    )
    ax.xaxis.set_ticks_position("top")

    plt.title("SHAP Feature Importance Across Models", fontsize=14)
    plt.show()
