"""
Kiểm định thống kê cho Thí nghiệm Scaling (Raw vs Scaled).

Đặt file này ở: src/analysis/stats_tests.py (tạo mới thư mục src/analysis/
nếu chưa có, kèm __init__.py trống).

Chạy như module từ project root:
    python -m src.analysis.stats_tests

Yêu cầu: Scaling.csv đã tồn tại trong thư mục 'results' (config['paths']['results']),
với các cột: block_id, pipeline, model, accuracy, macro_f1
(pipeline chỉ có 2 giá trị: "raw" và "scaled" -- đúng như build_pipeline.py/config.save() tạo ra).

Cấu trúc: 3 hàm thủ công tách riêng, mỗi hàm làm đúng 1 việc:
    1. test_normality()  -- Shapiro-Wilk trên paired differences
    2. paired_ttest()    -- paired t-test (luôn có CI closed-form, không cần bootstrap)
    3. wilcoxon_test()   -- Wilcoxon signed-rank; tham số `bootstrap`:
         - bootstrap=True  -> chạy thêm bootstrap CI cho MEDIAN (khớp với đại
                              lượng Wilcoxon thực sự kiểm định)
         - bootstrap=False -> chạy Wilcoxon bình thường, KHÔNG tính CI (ci_low/
                              ci_high = None), nhanh hơn, dùng khi chỉ cần p-value

Hàm `analyze_scaling()` điều phối 3 hàm trên cho từng (model, metric), quyết
định dùng paired_ttest hay wilcoxon_test dựa trên kết quả test_normality().
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, shapiro

from src.config import load_config, resolve_path

DEFAULT_METRIC_CANDIDATES = ["accuracy", "macro_f1"]
ALPHA = 0.05
NORMALITY_ALPHA = 0.05  # ngưỡng Shapiro-Wilk để quyết định paired t-test hay Wilcoxon
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Hàm 1: kiểm tra phân phối chuẩn của paired differences
# ---------------------------------------------------------------------------
def test_normality(differences: np.ndarray, alpha: float = NORMALITY_ALPHA) -> dict:
    """
    Shapiro-Wilk test trên vector paired differences (scaled - raw).

    Lưu ý: với n nhỏ (vd 10 block), power của Shapiro-Wilk thấp -> không phát
    hiện được các sai lệch nhẹ khỏi phân phối chuẩn. Kết quả nên được diễn
    giải thận trọng, không dùng làm căn cứ duy nhất và tuyệt đối.

    Trả về dict: {statistic, p_value, is_normal}
    """
    statistic, p_value = shapiro(differences)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "is_normal": p_value >= alpha,  # True: không bác bỏ giả thuyết chuẩn
    }


# ---------------------------------------------------------------------------
# Hàm 2: paired t-test
# ---------------------------------------------------------------------------
def paired_ttest(raw_scores: np.ndarray, scaled_scores: np.ndarray,
                  alpha: float = ALPHA) -> dict:
    """
    Paired t-test (scaled vs raw). CI luôn là closed-form (không cần bootstrap
    vì scipy.stats.ttest_rel đã cung cấp sẵn qua .confidence_interval()).

    Trả về dict: {statistic, p_value, ci_low, ci_high}
    """
    result = ttest_rel(scaled_scores, raw_scores)
    ci = result.confidence_interval(confidence_level=1 - alpha)
    return {
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "ci_low": ci.low,
        "ci_high": ci.high,
    }


# ---------------------------------------------------------------------------
# Hàm 3: Wilcoxon signed-rank test
# ---------------------------------------------------------------------------
def wilcoxon_test(raw_scores: np.ndarray, scaled_scores: np.ndarray,
                   bootstrap: bool = False, n_bootstrap: int = N_BOOTSTRAP,
                   seed: int = BOOTSTRAP_SEED, confidence: float = 1 - ALPHA) -> dict:
    """
    Wilcoxon signed-rank test (scaled vs raw).

    bootstrap=False (mặc định): chỉ chạy Wilcoxon bình thường, KHÔNG tính CI
        -> ci_low/ci_high trả về None.
    bootstrap=True: chạy thêm bootstrap percentile CI cho MEDIAN của paired
        differences (khớp với đại lượng Wilcoxon thực sự đánh giá -- không
        bootstrap trên mean vì sẽ không nhất quán với kiểm định).

    Trả về dict: {statistic, p_value, ci_low, ci_high} (ci_* = None nếu bootstrap=False)
    """
    result = wilcoxon(scaled_scores, raw_scores)

    ci_low, ci_high = None, None
    if bootstrap:
        differences = scaled_scores - raw_scores
        rng = np.random.default_rng(seed)
        n = len(differences)
        boot_medians = np.array([
            np.median(rng.choice(differences, size=n, replace=True))
            for _ in range(n_bootstrap)
        ])
        alpha_tail = (1 - confidence) / 2 * 100
        ci_low, ci_high = np.percentile(boot_medians, [alpha_tail, 100 - alpha_tail])

    return {
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# ---------------------------------------------------------------------------
# Ghép cặp dữ liệu theo block_id
# ---------------------------------------------------------------------------
def _get_paired_vectors(df: pd.DataFrame, model: str, metric: str):
    """
    Lấy raw_scores và scaled_scores CÙNG THỨ TỰ block_id cho 1 model + 1 metric.
    Trả về (block_ids, raw_scores, scaled_scores) dạng np.ndarray.
    """
    sub = df[df["model"] == model]

    raw = (
        sub[sub["pipeline"] == "raw"]
        .sort_values("block_id")
        .set_index("block_id")[metric]
    )
    scaled = (
        sub[sub["pipeline"] == "scaled"]
        .sort_values("block_id")
        .set_index("block_id")[metric]
    )

    common_blocks = raw.index.intersection(scaled.index)
    if len(common_blocks) != len(raw) or len(common_blocks) != len(scaled):
        missing_raw = set(scaled.index) - set(raw.index)
        missing_scaled = set(raw.index) - set(scaled.index)
        raise ValueError(
            f"[{model}/{metric}] Thiếu block để paired: "
            f"missing_in_raw={missing_raw}, missing_in_scaled={missing_scaled}"
        )

    common_blocks = sorted(common_blocks)
    raw_scores = raw.loc[common_blocks].to_numpy()
    scaled_scores = scaled.loc[common_blocks].to_numpy()

    return np.array(common_blocks), raw_scores, scaled_scores


def _resolve_metrics(df: pd.DataFrame, metrics):
    if metrics is not None:
        return list(metrics)
    return [m for m in DEFAULT_METRIC_CANDIDATES if m in df.columns]


# ---------------------------------------------------------------------------
# Điều phối: chọn hàm 2 hoặc hàm 3 dựa trên kết quả hàm 1
# ---------------------------------------------------------------------------
def analyze_scaling(df_scaling: pd.DataFrame, models=None, metrics=None,
                     bootstrap: bool = False) -> pd.DataFrame:
    """
    bootstrap: truyền thẳng xuống wilcoxon_test() khi kiểm định được chọn là
    Wilcoxon. Không ảnh hưởng đến paired t-test (vốn luôn có CI closed-form).
    """
    if models is None:
        models = sorted(df_scaling["model"].unique())
    metrics = _resolve_metrics(df_scaling, metrics)

    rows = []

    for model in models:
        for metric in metrics:
            block_ids, raw_scores, scaled_scores = _get_paired_vectors(
                df_scaling, model, metric
            )
            differences = scaled_scores - raw_scores

            # Trường hợp biên: tất cả các cặp giống hệt nhau -> Wilcoxon sẽ lỗi,
            # kết luận "không khác biệt" là hiển nhiên, bỏ qua kiểm định.
            if np.allclose(differences, 0):
                rows.append({
                    "model": model, "metric": metric, "n_blocks": len(block_ids),
                    "raw_mean": raw_scores.mean(), "raw_std": raw_scores.std(ddof=1),
                    "scaled_mean": scaled_scores.mean(), "scaled_std": scaled_scores.std(ddof=1),
                    "mean_diff": 0.0, "median_diff": 0.0,
                    "ci_95_low": 0.0, "ci_95_high": 0.0, "ci_target": "mean/median (both = 0)",
                    "test_used": "N/A (all paired differences = 0)",
                    "shapiro_p_on_diff": np.nan, "statistic": np.nan, "p_value": 1.0,
                    "significant_at_0.05": False,
                    "conclusion": "Chưa đủ bằng chứng để kết luận có khác biệt",
                })
                continue

            normality = test_normality(differences)

            if normality["is_normal"]:
                test_result = paired_ttest(raw_scores, scaled_scores)
                test_name = "paired t-test (ttest_rel)"
                ci_target = "mean"
            else:
                test_result = wilcoxon_test(raw_scores, scaled_scores, bootstrap=bootstrap)
                test_name = "Wilcoxon signed-rank (wilcoxon)"
                ci_target = "median" if bootstrap else "N/A (bootstrap=False)"

            significant = test_result["p_value"] < ALPHA

            rows.append({
                "model": model,
                "metric": metric,
                "n_blocks": len(block_ids),
                "raw_mean": raw_scores.mean(),
                "raw_std": raw_scores.std(ddof=1),
                "scaled_mean": scaled_scores.mean(),
                "scaled_std": scaled_scores.std(ddof=1),
                "mean_diff": differences.mean(),
                "median_diff": np.median(differences),
                "ci_95_low": test_result["ci_low"],
                "ci_95_high": test_result["ci_high"],
                "ci_target": ci_target,
                "test_used": test_name,
                "shapiro_p_on_diff": normality["p_value"],
                "statistic": test_result["statistic"],
                "p_value": test_result["p_value"],
                "significant_at_0.05": significant,
                "conclusion": (
                    "Có bằng chứng thống kê Raw và Scaled khác nhau"
                    if significant
                    else "Chưa đủ bằng chứng để kết luận có khác biệt"
                ),
            })

    return pd.DataFrame(rows)


def print_report(results_df: pd.DataFrame):
    for _, r in results_df.iterrows():
        print("=" * 70)
        print(f"Model: {r['model']} | Metric: {r['metric']} | n_blocks={r['n_blocks']}")
        print(f"  Raw:    {r['raw_mean']:.4f} +/- {r['raw_std']:.4f}")
        print(f"  Scaled: {r['scaled_mean']:.4f} +/- {r['scaled_std']:.4f}")
        print(f"  Mean paired difference (scaled - raw):   {r['mean_diff']:.4f}")
        print(f"  Median paired difference (scaled - raw): {r['median_diff']:.4f}")
        if r["ci_95_low"] is None or (isinstance(r["ci_95_low"], float) and np.isnan(r["ci_95_low"])):
            print(f"  95% CI: không tính (bootstrap=False)")
        else:
            print(f"  95% CI (for {r['ci_target']}): [{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}]")
        print(f"  Shapiro-Wilk p-value on differences: {r['shapiro_p_on_diff']}"
              " (n nhỏ -> power thấp, diễn giải thận trọng)")
        print(f"  Test used: {r['test_used']}")
        print(f"  Statistic: {r['statistic']} | p-value: {r['p_value']:.4f}")
        print(f"  Significant at alpha=0.05: {r['significant_at_0.05']}")
        print(f"  Conclusion: {r['conclusion']}")
    print("=" * 70)


def main(bootstrap: bool = False):
    cfg = load_config()
    df_scaling = pd.read_csv(resolve_path(cfg, "results") / "Scaling.csv")

    results_df = analyze_scaling(df_scaling, bootstrap=bootstrap)
    print_report(results_df)

    out_path = resolve_path(cfg, "results") / "scaling_stats_report.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    # Đổi thành main(bootstrap=True) nếu muốn có CI cho cả trường hợp Wilcoxon.
    main(bootstrap=False)
