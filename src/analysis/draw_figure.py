import ast

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
sns.set_theme(style="ticks", palette="pastel")

from src.config import load_config, read_results, resolve_path
cfg = load_config()
FIGURES_DIR = resolve_path(cfg, 'figures')

# Scaling
def draw_scaling_boxplot(df_scaling):

    # Description:
    # Vẽ 1 plot bao gồm 3 models, mỗi model gồm 2 boxplot thể hiện tứ phân vị của f1-score của 10 block kết hợp lại
    # Mục đích là để xem giữa raw và scaled có sự thay đổi đáng kể nào về f1-score của các models không
    sns.boxplot(
        data=df_scaling,
        x="model",
        y="macro_f1",
        hue="pipeline",
        palette={
            "raw": "#FF0000",
            "scaled": "#00FF00"
        },
        width=0.6
    )

    plt.xlabel("Model")
    plt.ylabel("Macro F1")
    plt.title("Effect of Feature Scaling on Model Performance")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    return plt
def draw_scaling_time(df_scaling, time_tag):
    
    # Description: 
    # Vẽ 3 plot chỗ mỗi model, so sánh xem thời gian fit(infer) giữa raw và scaled của các models có bị ảnh hưởng mạnh không
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5),
        sharex=True,
        sharey=False
    )
    models = df_scaling["model"].unique()

    for ax, model in zip(axes, models):

        model_df = df_scaling[df_scaling["model"] == model]
        
        raw_df = model_df[model_df["pipeline"] == "raw"]
        scaled_df = model_df[model_df["pipeline"] == "scaled"]

        ax.plot(
            raw_df["block_id"],
            raw_df[time_tag],
            marker="o",
            color="red",
            label="Raw"
        )
        ax.plot(
            scaled_df["block_id"],
            scaled_df[time_tag],
            marker="s",
            color="green",
            label="Scaled"
        )
        
        ax.set_title(model)
        ax.set_xlabel("Block")
        ax.set_ylabel("fit_run_time (s)")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    return plt

# PCA
# ------------------------------------------------------------------
# Cấu hình chung cho các plot PCA
# ------------------------------------------------------------------
PCA_PIPELINE_ORDER = ["no_pca", "pca_50", "pca_75", "pca_90"]
PCA_PIPELINE_PALETTE = {
    "no_pca": "#4C72B0",
    "pca_50": "#DD8452",
    "pca_75": "#55A868",
    "pca_90": "#C44E52",
}
MODEL_PALETTE = {
    "KNeighborsClassifier": "#4C72B0",
    "LogisticRegression": "#DD8452",
    "RandomForestClassifier": "#55A868",
}
MODEL_ORDER = list(MODEL_PALETTE.keys())


def normalize_pipeline_label(value):
    # Cột 'pipeline' có thể đã là string label sẵn (vd 'no_pca', 'pca_50', ...)
    # hoặc là string dạng list cũ "[flag_scale, flag_pca, n_components]" nếu
    # build_pipeline.py chưa đổi sang string label. Hàm này xử lý được cả 2.
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            flag_scale, flag_pca, n_components = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        if not flag_pca or n_components is None:
            return "no_pca"
        return f"pca_{round(float(n_components) * 100)}"
    return value


def plot_boxplot(df, metric="macro_f1", save_path=FIGURES_DIR / "pca_boxplot.png"):
    # Description:
    # Vẽ boxplot thủ công bằng ax.boxplot (thay cho sns.boxplot) để kiểm soát
    # màu/độ rộng box tốt hơn: 3 models, mỗi model 4 boxplot (no_pca, pca_50,
    # pca_75, pca_90) thể hiện tứ phân vị của macro_f1 qua 10 block.
    fig, ax = plt.subplots(figsize=(11, 5))

    n_pipelines = len(PCA_PIPELINE_ORDER)
    width = 0.8 / n_pipelines
    x_base = np.arange(len(MODEL_ORDER))

    for i, pipeline in enumerate(PCA_PIPELINE_ORDER):
        data = []
        for model in MODEL_ORDER:
            vals = df[(df["model"] == model) & (df["pipeline"] == pipeline)][metric]
            data.append(vals.values)
        positions = x_base + (i - (n_pipelines - 1) / 2) * width
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            manage_ticks=False,
        )
        for box in bp["boxes"]:
            box.set_facecolor(PCA_PIPELINE_PALETTE[pipeline])
            box.set_alpha(0.8)
        # legend proxy
        ax.plot([], [], color=PCA_PIPELINE_PALETTE[pipeline], label=pipeline, linewidth=8)

    ax.set_xticks(x_base)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel(metric)
    ax.set_title(f"Distribution of {metric} across 10 blocks — by model x pipeline")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, title="Pipeline")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def draw_pca_paired_line(df_pca, metric="macro_f1"):
    # Description:
    # Vẽ 3 plot mỗi model, mỗi plot gồm 4 đường (no_pca, pca_50, pca_75, pca_90)
    # nối qua 10 block -- để xem PCA có làm thay đổi hiệu năng 1 cách nhất quán
    # qua các block hay không (paired theo block_id).
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5),
        sharex=True,
        sharey=True,
    )
    models = df_pca["model"].unique()

    for ax, model in zip(axes, models):
        model_df = df_pca[df_pca["model"] == model]

        for pipeline in PCA_PIPELINE_ORDER:
            line_df = model_df[model_df["pipeline"] == pipeline].sort_values("block_id")
            if line_df.empty:
                continue
            ax.plot(
                line_df["block_id"],
                line_df[metric],
                marker="o",
                color=PCA_PIPELINE_PALETTE[pipeline],
                linestyle="-" if pipeline == "no_pca" else "--",
                label=pipeline,
            )

        ax.set_title(model)
        ax.set_xlabel("Block")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend(fontsize=8)

    plt.tight_layout()
    return plt


def draw_pca_performance(df_pca, metric="macro_f1"):
    # Description:
    # Grouped bar chart: trục x = mức PCA (no_pca/50/75/90), mỗi nhóm gồm 3 cột
    # cạnh nhau (1 cột / model), giá trị = mean +- std của metric qua 10 block.
    summary = (
        df_pca.groupby(["pipeline", "model"], observed=True)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    models = list(MODEL_PALETTE.keys())
    x_base = np.arange(len(PCA_PIPELINE_ORDER))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        rows = summary[summary["model"] == model].set_index("pipeline")
        means = [rows.loc[p, "mean"] if p in rows.index else np.nan for p in PCA_PIPELINE_ORDER]
        stds = [rows.loc[p, "std"] if p in rows.index else 0 for p in PCA_PIPELINE_ORDER]
        positions = x_base + (i - (len(models) - 1) / 2) * width
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
    ax.set_xticklabels(PCA_PIPELINE_ORDER)
    ax.set_xlabel("Pipeline (PCA level)")
    ax.set_ylabel(metric)
    ax.set_title("Performance by PCA level — mean ± std across 10 blocks")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    return plt


def _nice_round_seconds(value):
    # Làm tròn giây về mốc dễ đọc, tổng quát (không lấy lẻ)
    steps = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600]
    for s in steps:
        if value <= s:
            return s
    return int(round(value / 60)) * 60


def _nice_legend_values(series):
    percentiles = np.percentile(series, [25, 60, 95])
    return sorted({_nice_round_seconds(v) for v in percentiles})


def _draw_diameter_legend(fig, values, unit_label, rect):
    """
    Vẽ 1 legend kiểu "Diameter" (giống hình ConvNeXt paper): các vòng tròn
    tiếp tuyến chung 1 đường đáy, kích thước tăng dần trái->phải, có tick
    mark + nhãn giá trị bên dưới mỗi vòng tròn. Vẽ trên 1 inset axes riêng
    (toạ độ theo fraction của figure) để không đè lên data chính.

    rect: (left, bottom, width, height) theo toạ độ fraction của figure.
    """
    legend_ax = fig.add_axes(rect)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    values = sorted(values)
    max_val = values[-1]
    max_radius = 0.30  # bán kính (fraction of axes) của vòng tròn lớn nhất
    radii = [max_radius * np.sqrt(v / max_val) for v in values]

    baseline_y = 0.16
    xs = np.linspace(0.20, 0.85, len(values))

    for x, r, v in zip(xs, radii, values):
        circle = plt.Circle(
            (x, baseline_y + r), r,
            transform=legend_ax.transAxes,
            fill=False, edgecolor="gray", linewidth=1.2,
        )
        legend_ax.add_patch(circle)
        legend_ax.plot(
            [x, x], [baseline_y - 0.05, baseline_y],
            color="gray", linewidth=1, transform=legend_ax.transAxes,
        )
        legend_ax.text(
            x, baseline_y - 0.09, f"{v:g}",
            ha="center", va="top", fontsize=7, transform=legend_ax.transAxes,
        )

    legend_ax.plot(
        [0.08, 0.92], [baseline_y, baseline_y],
        color="gray", linewidth=1, transform=legend_ax.transAxes,
    )
    legend_ax.text(
        0.08, baseline_y + max_radius * 2 + 0.10, unit_label,
        ha="left", va="bottom", fontsize=7, transform=legend_ax.transAxes,
    )


def draw_pca_bubble_runtime(df_pca, runtime_col="fit_runtime_sec", metric="macro_f1"):
    summary = (
        df_pca.groupby(["pipeline", "model"], observed=True)
        .agg(metric_mean=(metric, "mean"), runtime_mean=(runtime_col, "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = {p: i for i, p in enumerate(PCA_PIPELINE_ORDER)}

    # 1. BIẾN ĐỔI LOGARITHM ĐỂ GIỮ TỶ LỆ CHUẨN XÁC
    # Offset tránh log(0)
    log_runtime = np.log1p(summary["runtime_mean"])

    # Scale diameter theo log
    d_min, d_max = 12, 45
    log_min, log_max = log_runtime.min(), log_runtime.max()

    if log_max > log_min:
        summary["diameter"] = d_min + (log_runtime - log_min) / (log_max - log_min) * (d_max - d_min)
    else:
        summary["diameter"] = (d_min + d_max) / 2

    # Diện tích = (đường kính)^2 (Matplotlib)
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
        for x, y in zip(xs, sub["metric_mean"]):
            ax.annotate(
                model.replace("Classifier", "").replace("Regression", ""),
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
            )

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()))
    ax.set_xlabel("Pipeline")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {runtime_col} by pipeline (bubble size = log-scaled runtime)")

    # Legend Model - Góc dưới trái
    model_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8)
        for c in MODEL_PALETTE.values()
    ]
    ax.legend(
        model_handles,
        list(MODEL_PALETTE.keys()),
        loc="lower left",
        bbox_to_anchor=(0.02, 0.15),
        fontsize=7,
        title="Model",
        labelspacing=0.8,
        borderpad=0.8,
    )

    plt.tight_layout()

    # 2. VẼ THƯỚC ĐO DIAMETER THEO THANG LOG (GÓC DƯỚI PHẢI)
    r_min = summary["runtime_mean"].min()
    r_max = summary["runtime_mean"].max()

    # Tạo 4 mốc thực tế đại diện (ví dụ: min, 10%, 50%, max)
    legend_vals = np.array([r_min, r_min + (r_max - r_min) * 0.1, r_min + (r_max - r_min) * 0.4, r_max])

    # Tương ứng với vị trí x (0 đến 1) trên thước đo log
    log_vals = np.log1p(legend_vals)
    x_coords = (log_vals - log_min) / (log_max - log_min) if log_max > log_min else np.linspace(0, 1, 4)

    ax_leg = fig.add_axes([0.65, 0.12, 0.28, 0.08])
    ax_leg.axis("off")

    y_line = 0.5
    ax_leg.plot([0, 1], [y_line, y_line], color="gray", lw=1)

    for x, val in zip(x_coords, legend_vals):
        ax_leg.plot([x, x], [y_line - 0.15, y_line + 0.15], color="gray", lw=1)
        ax_leg.text(x, y_line - 0.35, f"{val:.1f}", ha="center", va="top", fontsize=7, color="#333333")

    ax_leg.text(0.0, y_line + 0.3, "Diameter (log scale)", ha="left", va="bottom", fontsize=8, color="#333333")
    ax_leg.text(1.05, y_line - 0.35, f"{runtime_col} (s)", ha="left", va="top", fontsize=7, color="#333333")

    ax_leg.set_xlim(-0.05, 1.25)
    ax_leg.set_ylim(0, 1)

    return plt

# Save 
def save_fig(plt,file_name):
    path = resolve_path(cfg, 'figures') / file_name
    plt.savefig(path)

def main():
    df_scaling = read_results(cfg, 'Scaling.csv')

    plt_boxplot = draw_scaling_boxplot(df_scaling)
    save_fig(plt_boxplot, 'scaling_boxplot.png')

    plt_fit = draw_scaling_time(df_scaling,'fit_runtime_sec')
    save_fig(plt_fit, 'scaling_runtime_plot.png')

    plt_infer = draw_scaling_time(df_scaling,'inference_runtime_sec')
    save_fig(plt_infer, 'scaling_infer_plot.png')

    df_pca = read_results(cfg, 'PCA.csv')
    df_pca["pipeline"] = df_pca["pipeline"].apply(normalize_pipeline_label)

    plot_boxplot(df_pca, metric="macro_f1")

    plt_pca_paired = draw_pca_paired_line(df_pca, metric="macro_f1")
    save_fig(plt_pca_paired, 'pca_paired_plot.png')

    plt_pca_performance = draw_pca_performance(df_pca, metric="macro_f1")
    save_fig(plt_pca_performance, 'pca_performance.png')

    plt_pca_bubble = draw_pca_bubble_runtime(df_pca, runtime_col="fit_runtime_sec")
    save_fig(plt_pca_bubble, 'pca_runtime_plot.png')

if __name__ == "__main__":
    main()