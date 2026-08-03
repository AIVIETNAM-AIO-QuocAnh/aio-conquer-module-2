import ast

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", palette="pastel")

from src.config import load_config, read_results, resolve_path

cfg = load_config()
FIGURES_DIR = resolve_path(cfg, "figures")

DPI = 300

SCALING_PIPELINE_ORDER = ["raw", "scaled"]
SCALING_PIPELINE_PALETTE = {
    "raw": "#D13A3F",
    "scaled": "#3EC35D",
}

PCA_PIPELINE_ORDER = ["no_pca", "pca_90", "pca_75", "pca_50"]
PCA_PIPELINE_PALETTE = {
    "no_pca": "#3973D0",
    "pca_50": "#E47B3E",
    "pca_75": "#3EC35D",
    "pca_90": "#D13A3F",
}

MODEL_PALETTE = {
    "KNeighborsClassifier": "#3973D0",
    "LogisticRegression": "#E47B3E",
    "RandomForestClassifier": "#3EC35D",
}
MODEL_ORDER = list(MODEL_PALETTE.keys())


def normalize_pipeline_label(value):
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            flag_scale, flag_pca, n_components = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        if not flag_pca or n_components is None:
            return "no_pca"
        return f"pca_{round(float(n_components) * 100)}"
    return value

# 1. Boxplot: model x pipeline, tứ phân vị của metric qua các block
def draw_boxplot_by_pipeline(
    df,
    pipeline_order,
    pipeline_palette,
    metric="macro_f1",
    title="",
    figsize=(9, 5),
):
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    n_pipelines = len(pipeline_order)
    width = 0.8 / n_pipelines
    x_base = np.arange(len(MODEL_ORDER))

    for i, pipeline in enumerate(pipeline_order):
        data = [
            df[(df["model"] == model) & (df["pipeline"] == pipeline)][metric].values
            for model in MODEL_ORDER
        ]
        positions = x_base + (i - (n_pipelines - 1) / 2) * width
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            manage_ticks=False,
        )
        for box in bp["boxes"]:
            box.set_facecolor(pipeline_palette[pipeline])
            box.set_alpha(0.8)
        ax.plot([], [], color=pipeline_palette[pipeline], label=pipeline, linewidth=8)

    # Vẽ đường gạch dọc ở giữa các vị trí model
    for x in x_base[:-1]:
        ax.axvline(
            x + 0.5, 
            color="gray", 
            linestyle="--", 
            linewidth=1, 
            alpha=0.6
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Distribution of {metric} across blocks — by model x pipeline")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, title="Pipeline")

    fig.tight_layout()
    return plt

# 2. Paired line plot:
def draw_paired_line_by_pipeline(
    df,
    pipeline_order,
    pipeline_palette,
    metric="accuracy",
    figsize=(18, 5),
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(MODEL_ORDER),
        figsize=figsize,
        sharex=True,
        sharey=True,
        dpi=DPI,
    )
    if len(MODEL_ORDER) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODEL_ORDER):
        model_df = df[df["model"] == model]

        for i, pipeline in enumerate(pipeline_order):
            line_df = model_df[model_df["pipeline"] == pipeline].sort_values("block_id")
            if line_df.empty:
                continue
            ax.plot(
                line_df["block_id"],
                line_df[metric],
                marker="o",
                color=pipeline_palette[pipeline],
                linestyle="-" if i == 0 else "--",
                label=pipeline,
            )

        ax.set_title(model)
        ax.set_xlabel("Block")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend(fontsize=8)

    plt.tight_layout()
    return plt

# 3. mean +- std của metric theo pipeline
def draw_performance_bar(
    df,
    pipeline_order,
    metric="macro_f1",
    title="",
    xlabel="Pipeline",
    figsize=(9, 5),
):
    summary = (
        df.groupby(["pipeline", "model"], observed=True)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    x_base = np.arange(len(pipeline_order))
    width = 0.8 / len(MODEL_ORDER)

    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    for i, model in enumerate(MODEL_ORDER):
        rows = summary[summary["model"] == model].set_index("pipeline")
        means = [rows.loc[p, "mean"] if p in rows.index else np.nan for p in pipeline_order]
        stds = [rows.loc[p, "std"] if p in rows.index else 0 for p in pipeline_order]
        positions = x_base + (i - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(
            positions,
            means,
            width=width * 0.9,
            yerr=stds,
            capsize=3,
            label=model,
            color=MODEL_PALETTE[model],
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(pipeline_order)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.set_title(title or f"Performance by pipeline — mean ± std across blocks ({metric})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    return plt

# 4. Bubble plot: metric vs runtime theo pipeline, size bubble = log-scaled runtime
def draw_bubble_runtime(
    df,
    pipeline_order,
    runtime_col="fit_runtime_sec",
    metric="macro_f1",
    title=None,
    figsize=(11, 5.5),
):
    summary = (
        df.groupby(["pipeline", "model"], observed=True)
        .agg(metric_mean=(metric, "mean"), runtime_mean=(runtime_col, "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    x_positions = {p: i for i, p in enumerate(pipeline_order)}

    # 1. BIẾN ĐỔI LOGARITHM
    log_runtime = np.log1p(summary["runtime_mean"])
    d_min, d_max = 12, 45
    log_min, log_max = log_runtime.min(), log_runtime.max()

    if log_max > log_min:
        summary["diameter"] = d_min + (log_runtime - log_min) / (log_max - log_min) * (d_max - d_min)
    else:
        summary["diameter"] = (d_min + d_max) / 2

    summary["size"] = summary["diameter"] ** 2

    for model in MODEL_PALETTE:
        sub = summary[summary["model"] == model]
        if sub.empty:
            continue

        xs = [x_positions[p] for p in sub["pipeline"]]

        ax.scatter(
            xs,
            sub["metric_mean"],
            s=sub["size"],
            color=MODEL_PALETTE[model],
            alpha=0.55,
            edgecolors="white",
            linewidth=1,
        )

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()))
    ax.set_xlabel("Pipeline")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs {runtime_col} by pipeline (bubble size = log-scaled runtime)")

    # Giữ khoảng đệm biên chuẩn
    ax.set_xlim(-0.5, len(pipeline_order) - 0.5)
    y_min, y_max = summary["metric_mean"].min(), summary["metric_mean"].max()
    y_pad = (y_max - y_min) * 0.15 if y_max > y_min else 0.05
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # 1. LEGEND MODEL
    model_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8)
        for c in MODEL_PALETTE.values()
    ]
    # Lề dưới của box legend bằng đúng đáy trục Ox
    leg_model = ax.legend(
        model_handles,
        list(MODEL_PALETTE.keys()),
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        fontsize=8,
        title="Model",
        borderaxespad=0,
    )

    # 2. THƯỚC ĐO DIAMETER
    r_min = summary["runtime_mean"].min()
    r_max = summary["runtime_mean"].max()

    legend_vals = np.array(
        [r_min, r_min + (r_max - r_min) * 0.1, r_min + (r_max - r_min) * 0.4, r_max]
    )

    log_vals = np.log1p(legend_vals)
    x_coords = (
        (log_vals - log_min) / (log_max - log_min) if log_max > log_min else np.linspace(0, 1, 4)
    )

    ax_leg = fig.add_axes([0.80, 0.35, 0.14, 0.06])
    ax_leg.axis("off")

    y_line = 0.5
    ax_leg.plot([0, 1], [y_line, y_line], color="gray", lw=1)

    for x, val in zip(x_coords, legend_vals):
        ax_leg.plot([x, x], [y_line - 0.15, y_line + 0.15], color="gray", lw=1)
        ax_leg.text(x, y_line - 0.35, f"{val:.1f}", ha="center", va="top", fontsize=7, color="#333333")

    ax_leg.text(0.0, y_line + 0.3, "Diameter (log scale)", ha="left", va="bottom", fontsize=8, color="#333333")
    ax_leg.text(0.5, y_line - 0.7, f"{runtime_col} (s)", ha="center", va="top", fontsize=7, color="#333333")

    ax_leg.set_xlim(-0.05, 1.05)
    ax_leg.set_ylim(-0.2, 1.2)

    # Dành lề bên phải rộng đủ để chú thích không bị đè hay tràn viền
    fig.subplots_adjust(right=0.78, bottom=0.15)

    return plt


# Save
def save_fig(plt_obj, file_name, dpi=DPI):
    path = resolve_path(cfg, "figures") / file_name
    plt_obj.savefig(path, dpi=dpi)
    plt_obj.close("all") if hasattr(plt_obj, "close") else plt.close("all")
    print(f"Saved: {path}")


def main():
    # Scaling
    df_scaling = read_results(cfg, "Scaling.csv")

    save_fig(
        draw_boxplot_by_pipeline(
            df_scaling,
            SCALING_PIPELINE_ORDER,
            SCALING_PIPELINE_PALETTE,
            title="Effect of Feature Scaling on Model Performance",
            figsize=(8, 5),
        ),
        "scaling_boxplot.png",
    )

    save_fig(
        draw_bubble_runtime(df_scaling, SCALING_PIPELINE_ORDER, runtime_col="fit_runtime_sec"),
        "scaling_fit_runtime_plot.png",
    )

    save_fig(
        draw_bubble_runtime(df_scaling, SCALING_PIPELINE_ORDER, runtime_col="inference_runtime_sec"),
        "scaling_infer_plot.png",
    )

    save_fig(
        draw_paired_line_by_pipeline(df_scaling, SCALING_PIPELINE_ORDER, SCALING_PIPELINE_PALETTE),
        "scaling_paired_plot.png",
    )

    # PCA
    df_pca = read_results(cfg, "PCA.csv")
    df_pca["pipeline"] = df_pca["pipeline"].apply(normalize_pipeline_label)

    save_fig(
        draw_boxplot_by_pipeline(
            df_pca,
            PCA_PIPELINE_ORDER,
            PCA_PIPELINE_PALETTE,
            title="Distribution of macro_f1 across blocks — by model x PCA level",
            figsize=(11, 5),
        ),
        "pca_boxplot.png",
    )

    save_fig(
        draw_paired_line_by_pipeline(df_pca, PCA_PIPELINE_ORDER, PCA_PIPELINE_PALETTE),
        "pca_paired_plot.png",
    )

    save_fig(
        draw_performance_bar(
            df_pca,
            PCA_PIPELINE_ORDER,
            xlabel="Pipeline (PCA level)",
            title="Performance by PCA level — mean ± std across 10 blocks",
        ),
        "pca_performance.png",
    )

    save_fig(
        draw_bubble_runtime(df_pca, PCA_PIPELINE_ORDER, runtime_col="fit_runtime_sec"),
        "pca_fit_runtime_plot.png",
    )

    save_fig(
        draw_bubble_runtime(df_pca, PCA_PIPELINE_ORDER, runtime_col="inference_runtime_sec"),
        "pca_infer_runtime_plot.png",
    )


if __name__ == "__main__":
    main()