"""
Kiểm định thống kê cho Thí nghiệm Scaling (Raw vs Scaled) và Thí nghiệm PCA
(No-PCA vs các mức phương sai giữ lại).

Chạy như module từ project root: python -m src.analysis.stats_tests

Yêu cầu:
  - Scaling.csv đã tồn tại trong thư mục 'results' (config['paths']['results']),
    với các cột: block_id, pipeline, model, accuracy, macro_f1,
    fit_runtime_sec, inference_runtime_sec (pipeline chỉ có 2 giá trị: "raw"
    và "scaled" -- đúng như build_pipeline.py/config.save() tạo ra).
  - PCA.csv cùng thư mục, với cùng bộ cột, pipeline gồm "no_pca" và các mức
    phương sai giữ lại dạng "pca_XX" (vd "pca_50", "pca_75", "pca_90", hoặc
    "pca_90", "pca_95" tùy cấu hình n_components trong experiment.py).

Cấu trúc: các hàm thủ công tách riêng, mỗi hàm làm đúng 1 việc.

--- Nhóm hàm kiểm định lõi (dùng chung cho mọi phép so sánh) ---
    1. test_normality()  -- Shapiro-Wilk trên paired differences
    2. paired_ttest()    -- paired t-test (luôn có CI closed-form, không cần bootstrap)
    3. wilcoxon_test()   -- Wilcoxon signed-rank; tham số `bootstrap`:
         - bootstrap=True  -> chạy thêm bootstrap CI cho MEDIAN (khớp với đại
                              lượng Wilcoxon thực sự kiểm định)
         - bootstrap=False -> chạy Wilcoxon bình thường, KHÔNG tính CI (ci_low/
                              ci_high = None), nhanh hơn, dùng khi chỉ cần p-value

--- Nhóm hàm ghép cặp dữ liệu ---
    4. _get_paired_vectors()  -- ghép cặp "raw"/"scaled" theo block_id (giữ lại
                                  để tương thích ngược).
    5. _get_paired_values()   -- TỔNG QUÁT: ghép cặp HAI pipeline bất kỳ theo
                                  block_id (dùng được cho cả Scaling và PCA).

--- Thí nghiệm 1: Scaling (raw vs scaled) ---
    6. analyze_scaling()  -- điều phối kiểm định cho từng (model, metric);
                              nay là wrapper mỏng gọi analyze_pipeline_effect().

--- Thí nghiệm 2: PCA (no_pca vs từng mức phương sai) ---
    7. analyze_pipeline_effect() -- phiên bản TỔNG QUÁT của analyze_scaling():
                                     kiểm định paired differences giữa HAI
                                     pipeline bất kỳ. Dùng chung cho cả Scaling
                                     và PCA (đây cũng là lõi của "Thí nghiệm 3:
                                     Kiểm định thống kê" trong đề bài).
    8. _detect_pca_variance_pipelines() -- tự động phát hiện các pipeline PCA
                                            (vd "pca_50", "pca_90", "pca_95", ...)
                                            có trong PCA.csv.
    9. analyze_pca()       -- chạy analyze_pipeline_effect() cho từng mức PCA
                               so với baseline "no_pca", gộp kết quả lại
                               (tương ứng d_90, d_95, ... trong đề bài).
   10. summarize_pca_config() -- tổng hợp cấu hình PCA thực tế: số principal
                                  components, tỷ lệ explained variance, thời
                                  gian fit PCA -- nếu các cột này có trong dữ
                                  liệu (bỏ qua cột nào không tồn tại).

--- So sánh chéo hai file kết quả (không phải kiểm định thống kê) ---
   11. compare_no_pca_vs_scaled()   -- Yêu cầu (1): tính chênh lệch accuracy,
                                        macro_f1, fit_runtime_sec,
                                        inference_runtime_sec giữa pipeline
                                        "no_pca" (trong PCA.csv) và pipeline
                                        "scaled" (trong Scaling.csv) trên CÙNG
                                        block_id/model -- vì hai pipeline này
                                        cùng input và cùng chỉ áp dụng
                                        StandardScaler (không PCA), nên đây là
                                        kiểm tra tính nhất quán giữa hai lần
                                        chạy độc lập, KHÔNG phải hiệu ứng của
                                        một biến thí nghiệm.
   12. summarize_no_pca_vs_scaled()  -- tổng hợp (mean/std/min/max) của chênh
                                        lệch trên cho từng (model, metric).

--- In & lưu kết quả ---
   13. print_report()               -- in kết quả Thí nghiệm 1 (giữ nguyên).
   14. print_pipeline_effect_report()-- bản in TỔNG QUÁT cho mọi phép so sánh
                                        (dùng cho cả Scaling và PCA).
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, shapiro

from src.config import load_config, resolve_path

DEFAULT_METRIC_CANDIDATES = ["accuracy", "macro_f1"]
# Các metric dùng cho so sánh chéo no_pca vs scaled (yêu cầu 1): accuracy,
# macro F1, thời gian huấn luyện và thời gian suy luận -- đúng 4 đại lượng
# được yêu cầu.
DIFF_METRIC_CANDIDATES = ["accuracy", "macro_f1", "fit_runtime_sec", "inference_runtime_sec"]
# Các cột (nếu có) mô tả cấu hình PCA thực tế của từng lần chạy.
PCA_CONFIG_METRIC_CANDIDATES = ["n_features_after", "explained_variance_ratio", "pca_fit_time_sec"]

ALPHA = 0.05
NORMALITY_ALPHA = 0.05  # ngưỡng Shapiro-Wilk để quyết định paired t-test hay Wilcoxon
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Hàm 1: kiểm tra phân phối chuẩn của paired differences
# ---------------------------------------------------------------------------
def test_normality(differences: np.ndarray, alpha: float = NORMALITY_ALPHA) -> dict:
    """
    Shapiro-Wilk test trên vector paired differences (treatment - baseline).

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
def paired_ttest(baseline_scores: np.ndarray, treatment_scores: np.ndarray,
                  alpha: float = ALPHA) -> dict:
    """
    Paired t-test (treatment vs baseline). CI luôn là closed-form (không cần
    bootstrap vì scipy.stats.ttest_rel đã cung cấp sẵn qua .confidence_interval()).

    baseline_scores/treatment_scores: vd (raw, scaled) cho Thí nghiệm 1, hoặc
    (no_pca, pca_90) / (no_pca, pca_95) cho Thí nghiệm 2.

    Trả về dict: {statistic, p_value, ci_low, ci_high}
    """
    result = ttest_rel(treatment_scores, baseline_scores)
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
def wilcoxon_test(baseline_scores: np.ndarray, treatment_scores: np.ndarray,
                   bootstrap: bool = False, n_bootstrap: int = N_BOOTSTRAP,
                   seed: int = BOOTSTRAP_SEED, confidence: float = 1 - ALPHA) -> dict:
    """
    Wilcoxon signed-rank test (treatment vs baseline).

    bootstrap=False (mặc định): chỉ chạy Wilcoxon bình thường, KHÔNG tính CI
        -> ci_low/ci_high trả về None.
    bootstrap=True: chạy thêm bootstrap percentile CI cho MEDIAN của paired
        differences (khớp với đại lượng Wilcoxon thực sự đánh giá -- không
        bootstrap trên mean vì sẽ không nhất quán với kiểm định).

    Trả về dict: {statistic, p_value, ci_low, ci_high} (ci_* = None nếu bootstrap=False)
    """
    result = wilcoxon(treatment_scores, baseline_scores)

    ci_low, ci_high = None, None
    if bootstrap:
        differences = treatment_scores - baseline_scores
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
# Hàm 4: Ghép cặp dữ liệu theo block_id -- CỨNG cho "raw"/"scaled"
# (giữ lại để tương thích ngược với code cũ; ưu tiên dùng _get_paired_values
# cho code mới vì nó tổng quát cho mọi cặp pipeline)
# ---------------------------------------------------------------------------
def _get_paired_vectors(df: pd.DataFrame, model: str, metric: str):
    """
    Lấy raw_scores và scaled_scores CÙNG THỨ TỰ block_id cho 1 model + 1 metric.
    Trả về (block_ids, raw_scores, scaled_scores) dạng np.ndarray.
    """
    block_ids, raw_scores, scaled_scores = _get_paired_values(
        df, model, metric, pipeline_a="raw", pipeline_b="scaled"
    )
    return block_ids, raw_scores, scaled_scores


# ---------------------------------------------------------------------------
# Hàm 5: Ghép cặp dữ liệu theo block_id -- TỔNG QUÁT cho HAI pipeline bất kỳ
# ---------------------------------------------------------------------------
def _get_paired_values(df: pd.DataFrame, model: str, metric: str,
                        pipeline_a: str, pipeline_b: str):
    """
    Lấy giá trị của `metric` cho pipeline_a và pipeline_b, GHÉP CẶP theo
    block_id, cho một model. Đây là bản tổng quát của _get_paired_vectors
    (vốn chỉ cứng cho "raw"/"scaled"), dùng được cho mọi cặp pipeline, kể cả
    các pipeline PCA ("no_pca", "pca_50", "pca_90", "pca_95", ...) hoặc để
    ghép chéo giữa hai DataFrame khác nhau (sau khi đã pd.concat lại).

    Trả về (block_ids, values_a, values_b) dạng np.ndarray, cùng thứ tự block_id.
    """
    sub = df[df["model"] == model]

    a = (
        sub[sub["pipeline"] == pipeline_a]
        .sort_values("block_id")
        .set_index("block_id")[metric]
    )
    b = (
        sub[sub["pipeline"] == pipeline_b]
        .sort_values("block_id")
        .set_index("block_id")[metric]
    )

    common_blocks = a.index.intersection(b.index)
    if len(common_blocks) != len(a) or len(common_blocks) != len(b):
        missing_a = set(b.index) - set(a.index)
        missing_b = set(a.index) - set(b.index)
        raise ValueError(
            f"[{model}/{metric}] Thiếu block để ghép cặp giữa "
            f"'{pipeline_a}' và '{pipeline_b}': "
            f"missing_in_{pipeline_a}={missing_a}, missing_in_{pipeline_b}={missing_b}"
        )

    common_blocks = sorted(common_blocks)
    values_a = a.loc[common_blocks].to_numpy()
    values_b = b.loc[common_blocks].to_numpy()

    return np.array(common_blocks), values_a, values_b


def _resolve_metrics(df: pd.DataFrame, metrics):
    if metrics is not None:
        return list(metrics)
    return [m for m in DEFAULT_METRIC_CANDIDATES if m in df.columns]


def _dedupe_pipeline_runs(df: pd.DataFrame, subset=("block_id", "pipeline", "model"),
                           keep: str = "last", verbose: bool = True) -> pd.DataFrame:
    """
    Một số file kết quả (vd PCA.csv khi bị chạy lại experiment.py nhiều lần)
    có thể chứa nhiều dòng cho CÙNG một (block_id, pipeline, model), vì
    BuildPipeline.log() ghi file ở chế độ append và không tự dọn các lần
    chạy cũ. Nếu không xử lý, một block_id sẽ xuất hiện nhiều lần trong cùng
    một pipeline, khiến việc ghép cặp theo block_id giữa hai pipeline
    (_get_paired_values) bị lệch số lượng dòng và báo lỗi sai lệch.

    Loại bỏ các dòng trùng lặp theo `subset`, giữ lại dòng cuối cùng
    (keep="last", tức lần chạy gần nhất được ghi vào file) theo mặc định.
    In cảnh báo (nếu verbose=True và có trùng lặp) cho biết đã loại bao
    nhiêu dòng. Không sửa đổi df gốc.
    """
    dup_mask = df.duplicated(subset=list(subset), keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows == 0:
        return df

    deduped = df.drop_duplicates(subset=list(subset), keep=keep)
    if verbose:
        n_dropped = len(df) - len(deduped)
        print(
            f"[stats_tests] Cảnh báo: phát hiện {n_dup_rows} dòng trùng lặp theo "
            f"{list(subset)} (thường do file kết quả bị ghi append nhiều lần); "
            f"đã loại {n_dropped} dòng cũ, giữ lại lần chạy gần nhất (keep='{keep}')."
        )
    return deduped


# ---------------------------------------------------------------------------
# Hàm 6/7: Điều phối kiểm định TỔNG QUÁT -- lõi của Thí nghiệm 3
# (chọn paired_ttest hay wilcoxon_test dựa trên test_normality),
# dùng chung cho Thí nghiệm 1 (Scaling) và Thí nghiệm 2 (PCA)
# ---------------------------------------------------------------------------
def analyze_pipeline_effect(df: pd.DataFrame, pipeline_baseline: str, pipeline_treatment: str,
                             models=None, metrics=None, bootstrap: bool = False,
                             effect_label: str = None) -> pd.DataFrame:
    """
    Phiên bản TỔNG QUÁT của analyze_scaling(): kiểm định paired differences
    (treatment - baseline) giữa HAI pipeline bất kỳ trong cùng một DataFrame,
    cho từng (model, metric). Dùng chung cho:
      - Thí nghiệm 1 (Scaling): pipeline_baseline="raw", pipeline_treatment="scaled".
      - Thí nghiệm 2 (PCA):     pipeline_baseline="no_pca",
                                 pipeline_treatment="pca_90" / "pca_95" / ...

    Quy trình cho mỗi (model, metric) -- đúng "Thí nghiệm 3: Kiểm định thống
    kê" trong đề bài:
      1. Ghép N paired differences d_i = treatment_i - baseline_i theo block_id
         (N = số block, ví dụ 10).
      2. Báo cáo mean và std của d_i.
      3. Shapiro-Wilk trên d_i quyết định dùng paired t-test (nếu differences
         tương đối đối xứng / không có outlier nghiêm trọng, tức không bác bỏ
         giả thuyết chuẩn) hay Wilcoxon signed-rank (ngược lại).
      4. CHỈ dùng một kiểm định chính cho mỗi phép so sánh (không chạy cả hai
         rồi chọn kết quả đẹp hơn).
      5. Mức ý nghĩa alpha = 0.05.

    Tham số:
      effect_label: nhãn mô tả phép so sánh (vd "scaled_vs_raw",
                     "pca_90_vs_no_pca"); mặc định tự sinh từ tên hai pipeline.

    Trả về DataFrame với 1 dòng cho mỗi (model, metric), gồm cột 'comparison'
    để phân biệt khi gộp nhiều phép so sánh lại (xem analyze_pca()).
    """
    df = _dedupe_pipeline_runs(df)
    if models is None:
        models = sorted(df["model"].unique())
    metrics = _resolve_metrics(df, metrics)
    label = effect_label or f"{pipeline_treatment}_vs_{pipeline_baseline}"

    rows = []

    for model in models:
        for metric in metrics:
            block_ids, baseline_scores, treatment_scores = _get_paired_values(
                df, model, metric, pipeline_baseline, pipeline_treatment
            )
            differences = treatment_scores - baseline_scores

            # Trường hợp biên: tất cả các cặp giống hệt nhau -> kiểm định sẽ
            # lỗi (Wilcoxon không chấp nhận all-zero), kết luận "không khác
            # biệt" là hiển nhiên, bỏ qua kiểm định.
            if np.allclose(differences, 0):
                rows.append({
                    "comparison": label, "model": model, "metric": metric,
                    "n_blocks": len(block_ids),
                    "baseline_pipeline": pipeline_baseline,
                    "treatment_pipeline": pipeline_treatment,
                    "baseline_mean": baseline_scores.mean(), "baseline_std": baseline_scores.std(ddof=1),
                    "treatment_mean": treatment_scores.mean(), "treatment_std": treatment_scores.std(ddof=1),
                    "mean_diff": 0.0, "std_diff": 0.0, "median_diff": 0.0,
                    "ci_95_low": 0.0, "ci_95_high": 0.0, "ci_target": "mean/median (both = 0)",
                    "test_used": "N/A (all paired differences = 0)",
                    "shapiro_p_on_diff": np.nan, "statistic": np.nan, "p_value": 1.0,
                    "significant_at_0.05": False,
                    "conclusion": "Chưa đủ bằng chứng để kết luận có khác biệt",
                })
                continue

            normality = test_normality(differences)

            if normality["is_normal"]:
                test_result = paired_ttest(baseline_scores, treatment_scores)
                test_name = "paired t-test (ttest_rel)"
                ci_target = "mean"
            else:
                test_result = wilcoxon_test(baseline_scores, treatment_scores, bootstrap=bootstrap)
                test_name = "Wilcoxon signed-rank (wilcoxon)"
                ci_target = "median" if bootstrap else "N/A (bootstrap=False)"

            significant = test_result["p_value"] < ALPHA

            rows.append({
                "comparison": label,
                "model": model,
                "metric": metric,
                "n_blocks": len(block_ids),
                "baseline_pipeline": pipeline_baseline,
                "treatment_pipeline": pipeline_treatment,
                "baseline_mean": baseline_scores.mean(),
                "baseline_std": baseline_scores.std(ddof=1),
                "treatment_mean": treatment_scores.mean(),
                "treatment_std": treatment_scores.std(ddof=1),
                "mean_diff": differences.mean(),
                "std_diff": differences.std(ddof=1),
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
                    f"Có bằng chứng thống kê '{pipeline_treatment}' và "
                    f"'{pipeline_baseline}' khác nhau"
                    if significant
                    else "Chưa đủ bằng chứng để kết luận có khác biệt"
                ),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Thí nghiệm 1 (Scaling): wrapper mỏng giữ nguyên chữ ký/format cột cũ
# ---------------------------------------------------------------------------
def analyze_scaling(df_scaling: pd.DataFrame, models=None, metrics=None,
                     bootstrap: bool = False) -> pd.DataFrame:
    """
    Thí nghiệm 1 (Scaling): so sánh "raw" vs "scaled" cho từng (model, metric).

    Giữ lại để tương thích ngược với code/script cũ đang gọi analyze_scaling();
    nội bộ chỉ gọi analyze_pipeline_effect() rồi đổi tên cột về đúng format cũ
    (raw_mean/raw_std/scaled_mean/scaled_std) để print_report() không cần sửa.

    bootstrap: truyền thẳng xuống wilcoxon_test() khi kiểm định được chọn là
    Wilcoxon. Không ảnh hưởng đến paired t-test (vốn luôn có CI closed-form).
    """
    result = analyze_pipeline_effect(
        df_scaling,
        pipeline_baseline="raw",
        pipeline_treatment="scaled",
        models=models,
        metrics=metrics,
        bootstrap=bootstrap,
        effect_label="scaled_vs_raw",
    )
    return result.rename(columns={
        "baseline_mean": "raw_mean", "baseline_std": "raw_std",
        "treatment_mean": "scaled_mean", "treatment_std": "scaled_std",
    })


# ---------------------------------------------------------------------------
# Thí nghiệm 2 (PCA)
# ---------------------------------------------------------------------------
def _detect_pca_variance_pipelines(df_pca: pd.DataFrame) -> list:
    """
    Tự động phát hiện các pipeline PCA (khác baseline 'no_pca') có trong
    PCA.csv, vd 'pca_50', 'pca_75', 'pca_90' (dữ liệu hiện có) hoặc 'pca_90',
    'pca_95' (theo đề bài, nếu experiment.py được cấu hình lại với
    explain_var 90%/95%). Sắp xếp theo % phương sai tăng dần để kết quả in ra
    có thứ tự dễ đọc.
    """
    pipelines = [p for p in df_pca["pipeline"].unique() if p != "no_pca"]

    def _variance_key(p):
        try:
            return int(str(p).split("_")[-1])
        except (ValueError, IndexError):
            return 0

    return sorted(pipelines, key=_variance_key)


def analyze_pca(df_pca: pd.DataFrame, models=None, metrics=None,
                 variance_pipelines=None, baseline_pipeline: str = "no_pca",
                 bootstrap: bool = False) -> pd.DataFrame:
    """
    Thí nghiệm 2 (PCA): so sánh từng mức phương sai giữ lại (tự động phát
    hiện, vd "pca_90", "pca_95", hoặc "pca_50"/"pca_75"/"pca_90" tùy dữ liệu
    thực tế trong PCA.csv) với baseline không PCA ("no_pca" -- theo thiết kế
    của build_pipeline.py/preprocess.py, case flag_pca=True, n_components=None
    vẫn được StandardScaler áp dụng, nên baseline này về bản chất được "tái
    sử dụng" từ cùng loại pipeline chỉ-scaling như trong Thí nghiệm 1; xem
    compare_no_pca_vs_scaled() để kiểm tra tính nhất quán này).

    Với mỗi mô hình, mỗi metric và mỗi mức PCA, hiệu ứng tại block i là:
        d_i = s^{PCA}_i - s^{NoPCA}_i
    (tương ứng d_90, d_95, ... trong đề bài). Kiểm định thống kê thực hiện
    giống hệt quy trình của analyze_pipeline_effect() (Thí nghiệm 3: Shapiro-
    Wilk quyết định paired t-test hay Wilcoxon, alpha = 0.05, chỉ một kiểm
    định chính cho mỗi phép so sánh).

    Trả về DataFrame gồm kết quả của TẤT CẢ mức PCA được so sánh, phân biệt
    bằng cột 'comparison' (vd 'pca_90_vs_no_pca', 'pca_95_vs_no_pca').
    """
    df_pca = _dedupe_pipeline_runs(df_pca)
    if variance_pipelines is None:
        variance_pipelines = _detect_pca_variance_pipelines(df_pca)
    if not variance_pipelines:
        raise ValueError(
            "Không tìm thấy pipeline PCA nào (khác 'no_pca') trong df_pca."
        )

    all_results = []
    for pca_pipeline in variance_pipelines:
        result = analyze_pipeline_effect(
            df_pca,
            pipeline_baseline=baseline_pipeline,
            pipeline_treatment=pca_pipeline,
            models=models,
            metrics=metrics,
            bootstrap=bootstrap,
            effect_label=f"{pca_pipeline}_vs_{baseline_pipeline}",
        )
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True)


def summarize_pca_config(df_pca: pd.DataFrame, models=None,
                          variance_pipelines=None) -> pd.DataFrame:
    """
    Tổng hợp cấu hình PCA thực tế cho từng (pipeline PCA, model): số principal
    components thực tế, tỷ lệ explained variance và thời gian fit PCA riêng --
    CHỈ với các cột thực sự tồn tại trong df_pca (không suy diễn giá trị cho
    cột thiếu). Hiện tại PCA.csv chỉ có sẵn 'n_features_after' (dùng làm số
    principal components thực tế); nếu muốn có 'explained_variance_ratio' và
    'pca_fit_time_sec', cần ghi thêm các cột này ở bước log() của
    BuildPipeline/Preprocess.

    Trả về DataFrame dài với các cột thống kê (mean/min/max) cho mỗi cột cấu
    hình có sẵn, theo (pipeline, model).
    """
    df_pca = _dedupe_pipeline_runs(df_pca)
    if variance_pipelines is None:
        variance_pipelines = _detect_pca_variance_pipelines(df_pca)
    if models is None:
        models = sorted(df_pca["model"].unique())

    available_cols = [c for c in PCA_CONFIG_METRIC_CANDIDATES if c in df_pca.columns]
    if not available_cols:
        raise KeyError(
            f"Không có cột nào trong {PCA_CONFIG_METRIC_CANDIDATES} tồn tại "
            "trong df_pca; cần ghi thêm các cột này ở bước log() nếu muốn "
            "dùng summarize_pca_config()."
        )

    sub = df_pca[
        df_pca["pipeline"].isin(variance_pipelines) & df_pca["model"].isin(models)
    ]
    summary = sub.groupby(["pipeline", "model"])[available_cols].agg(["mean", "min", "max"])
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index()


# ---------------------------------------------------------------------------
# Yêu cầu (1): so sánh CHÉO "no_pca" (PCA.csv) vs "scaled" (Scaling.csv)
# -- KHÔNG phải kiểm định thống kê, chỉ tính chênh lệch giữa hai lần chạy của
#    CÙNG một pipeline tiền xử lý (chỉ StandardScaler, không PCA), trên cùng
#    input (cùng block) và cùng model, để kiểm tra tính nhất quán/tái lập.
# ---------------------------------------------------------------------------
def compare_no_pca_vs_scaled(df_pca: pd.DataFrame, df_scaling: pd.DataFrame,
                              models=None, metrics=None) -> pd.DataFrame:
    """
    So sánh pipeline "no_pca" (trong PCA.csv) với pipeline "scaled" (trong
    Scaling.csv) trên CÙNG block_id và CÙNG model, cho accuracy, macro_f1,
    fit_runtime_sec và inference_runtime_sec.

    Vì sao so sánh được: cả hai pipeline này dùng CÙNG input (cùng block, tức
    cùng dữ liệu train/test) và về mặt tiền xử lý là MỘT -- theo
    preprocess.py, điều kiện `if self.flag_scale or self.flag_pca:` khiến
    case (flag_scale=False, flag_pca=True, n_components=None) của Thí nghiệm
    PCA (baseline "no_pca") vẫn được StandardScaler áp dụng giống hệt pipeline
    "scaled" của Thí nghiệm 1. Do đó chênh lệch tính ra ở đây KHÔNG phải hiệu
    ứng của một biến thí nghiệm (không có biến nào thay đổi giữa hai cột),
    mà là chênh lệch giữa hai lần GridSearchCV độc lập trên cùng dữ liệu --
    hữu ích để kiểm tra mức độ ổn định/tái lập của toàn bộ pipeline huấn
    luyện (vd do GridSearchCV chọn ra bộ siêu tham số tốt nhất khác nhau giữa
    hai lần chạy).

    Trả về DataFrame dài (long format) với các cột:
    model, metric, block_id, no_pca_value, scaled_value, diff (scaled - no_pca).
    """
    if models is None:
        models = sorted(
            set(df_pca["model"].unique()) & set(df_scaling["model"].unique())
        )
    if metrics is None:
        metrics = [
            m for m in DIFF_METRIC_CANDIDATES
            if m in df_pca.columns and m in df_scaling.columns
        ]

    # Ghép hai lát cắt liên quan của hai DataFrame lại để tái dùng
    # _get_paired_values() (vốn chỉ thao tác trên MỘT DataFrame).
    combined = pd.concat(
        [
            df_pca[df_pca["pipeline"] == "no_pca"],
            df_scaling[df_scaling["pipeline"] == "scaled"],
        ],
        ignore_index=True,
    )
    combined = _dedupe_pipeline_runs(combined)

    rows = []
    for model in models:
        for metric in metrics:
            block_ids, no_pca_vals, scaled_vals = _get_paired_values(
                combined, model, metric, pipeline_a="no_pca", pipeline_b="scaled"
            )
            for block_id, v_no_pca, v_scaled in zip(block_ids, no_pca_vals, scaled_vals):
                rows.append({
                    "model": model,
                    "metric": metric,
                    "block_id": int(block_id),
                    "no_pca_value": v_no_pca,
                    "scaled_value": v_scaled,
                    "diff": v_scaled - v_no_pca,
                })

    return pd.DataFrame(rows)


def summarize_no_pca_vs_scaled(diff_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tổng hợp (mean, std, min, max, mean tuyệt đối) của cột 'diff' trả về từ
    compare_no_pca_vs_scaled(), theo từng (model, metric). Dùng để nhanh
    chóng nhận biết cặp (model, metric) nào có chênh lệch lớn bất thường
    giữa hai lần chạy "cùng pipeline" -- gợi ý bộ siêu tham số được
    GridSearchCV chọn ra không ổn định giữa hai lần chạy.
    """
    summary = (
        diff_df.groupby(["model", "metric"])["diff"]
        .agg(mean_diff="mean", std_diff="std", min_diff="min", max_diff="max",
             mean_abs_diff=lambda s: s.abs().mean())
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# In & lưu kết quả
# ---------------------------------------------------------------------------
def print_report(results_df: pd.DataFrame):
    """In kết quả Thí nghiệm 1 (Scaling) -- giữ nguyên định dạng cũ."""
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


def print_pipeline_effect_report(results_df: pd.DataFrame):
    """
    Bản in TỔNG QUÁT cho kết quả của analyze_pipeline_effect() / analyze_pca()
    (dùng cột 'comparison', 'baseline_*', 'treatment_*' thay vì raw_*/scaled_*
    cố định, nên in được cho mọi phép so sánh, kể cả nhiều mức PCA cùng lúc).
    """
    for _, r in results_df.iterrows():
        print("=" * 70)
        print(f"So sánh: {r['comparison']} | Model: {r['model']} | Metric: {r['metric']} "
              f"| n_blocks={r['n_blocks']}")
        print(f"  {r['baseline_pipeline']}: {r['baseline_mean']:.4f} +/- {r['baseline_std']:.4f}")
        print(f"  {r['treatment_pipeline']}: {r['treatment_mean']:.4f} +/- {r['treatment_std']:.4f}")
        print(f"  Mean paired difference (treatment - baseline):   {r['mean_diff']:.4f} "
              f"(std={r['std_diff']:.4f})")
        print(f"  Median paired difference (treatment - baseline): {r['median_diff']:.4f}")
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
    results_dir = resolve_path(cfg, "results")

    # --- Thí nghiệm 1: Scaling (raw vs scaled) ---
    df_scaling = pd.read_csv(results_dir / "Scaling.csv")
    scaling_results_df = analyze_scaling(df_scaling, bootstrap=bootstrap)
    print("\n### THÍ NGHIỆM 1 -- SCALING (raw vs scaled) ###")
    print_report(scaling_results_df)
    scaling_out_path = results_dir / "scaling_stats_report.csv"
    scaling_results_df.to_csv(scaling_out_path, index=False)
    print(f"\nSaved: {scaling_out_path}")

    # --- Thí nghiệm 2: PCA (no_pca vs từng mức phương sai) ---
    df_pca = pd.read_csv(results_dir / "PCA.csv")
    pca_results_df = analyze_pca(df_pca, bootstrap=bootstrap)
    print("\n### THÍ NGHIỆM 2 -- PCA (no_pca vs từng mức phương sai) ###")
    print_pipeline_effect_report(pca_results_df)
    pca_out_path = results_dir / "pca_stats_report.csv"
    pca_results_df.to_csv(pca_out_path, index=False)
    print(f"\nSaved: {pca_out_path}")

    # --- Yêu cầu (1): so sánh chéo no_pca (PCA.csv) vs scaled (Scaling.csv) ---
    diff_df = compare_no_pca_vs_scaled(df_pca, df_scaling)
    diff_summary_df = summarize_no_pca_vs_scaled(diff_df)
    print("\n### KIỂM TRA NHẤT QUÁN -- no_pca (PCA.csv) vs scaled (Scaling.csv) ###")
    print(diff_summary_df.to_string(index=False))
    diff_out_path = results_dir / "no_pca_vs_scaled_diff.csv"
    diff_summary_out_path = results_dir / "no_pca_vs_scaled_diff_summary.csv"
    diff_df.to_csv(diff_out_path, index=False)
    diff_summary_df.to_csv(diff_summary_out_path, index=False)
    print(f"\nSaved: {diff_out_path}")
    print(f"Saved: {diff_summary_out_path}")


if __name__ == "__main__":
    # Đổi thành main(bootstrap=True) nếu muốn có CI cho cả trường hợp Wilcoxon.
    main(bootstrap=False)
