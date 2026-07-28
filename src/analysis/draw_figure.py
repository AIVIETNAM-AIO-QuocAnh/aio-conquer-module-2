import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
sns.set_theme(style="ticks", palette="pastel")

from src.config import load_config, read_results, resolve_path
cfg = load_config()

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
    # CODE GO HERE
def draw_pca_performance(df_pca):
    # Description:
    return 0



# Save 
def save_fig(plt,file_name):
    path = resolve_path(cfg, 'results') / file_name
    plt.savefig(path)

def main():
    df_scaling = read_results(cfg, 'Scaling.csv')

    plt_boxplot = draw_scaling_boxplot(df_scaling)
    save_fig(plt_boxplot, 'scaling_boxplot.png')

    plt_fit = draw_scaling_time(df_scaling,'fit_runtime_sec')
    save_fig(plt_fit, 'scaling_runtime_plot.png')

    plt_infer = draw_scaling_time(df_scaling,'inference_runtime_sec')
    save_fig(plt_infer, 'scaling_infer_plot.png')
        
if __name__ == "__main__":
    main()