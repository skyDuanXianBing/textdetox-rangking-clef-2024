#!/usr/bin/env python3
"""
对比人工评价、LLM评价和机器评价的结果
包含多种一致性评价指标和评价质量分析
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import kendalltau, spearmanr, pearsonr, rankdata
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def collect_new_model_results(evaluation_type):
    """收集新模型的评价结果"""
    import glob

    base_dir = "data/result/llm_evolution"
    if not os.path.exists(base_dir):
        return {}

    results = {}
    models = ["deepseek", "qwen"]

    for model in models:
        model_results = []
        team_dirs = glob.glob(os.path.join(base_dir, "*_en"))

        for team_dir in team_dirs:
            team_name = os.path.basename(team_dir).replace("_en", "")
            model_dir = os.path.join(team_dir, model)

            if os.path.exists(model_dir):
                if evaluation_type == "zero_shot":
                    result_file = os.path.join(model_dir, f"etd_zero_shot_results_en_ENPrompt_{model}.csv")
                else:
                    result_file = os.path.join(model_dir, f"etd_few_shot_results_en_ENPrompt_{model}.csv")

                if os.path.exists(result_file):
                    try:
                        df = pd.read_csv(result_file)
                        if len(df) > 0:
                            stats_data = {
                                "team_name": team_name,
                                "j_score_mean": df["j_score"].mean(),
                                "sta_mean": df["STA"].mean(),
                                "cs_mean": df["CS"].mean(),
                                "fs_mean": df["FS"].mean(),
                                "total_samples": len(df)
                            }
                            model_results.append(stats_data)
                    except Exception as e:
                        print(f"❌ 读取 {result_file} 失败: {e}")

        if model_results:
            results[model] = pd.DataFrame(model_results)

    return results


def create_ranking_from_new_model(model_name, evaluation_type):
    """从新模型结果创建排名DataFrame"""
    new_results = collect_new_model_results(evaluation_type)

    if model_name not in new_results:
        print(f"❌ 未找到 {model_name} 模型的结果")
        return pd.DataFrame()

    df = new_results[model_name].copy()
    df = df.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
    print(f"✅ 使用 {model_name} 模型结果: {len(df)} 个团队")

    return df


def compare_all_new_models(evaluation_type, human_df, machine_df):
    """对比所有新模型的结果"""
    new_results = collect_new_model_results(evaluation_type)

    if not new_results:
        print("❌ 未找到新模型结果")
        return

    print(f"\n🤖 对比所有新模型 ({evaluation_type.replace('_', '-').title()} 评估)")
    print("=" * 80)

    for model_name, model_df in new_results.items():
        print(f"\n📈 {model_name.upper()} 模型结果:")
        print("-" * 50)

        # 重新排序并添加排名
        model_df = model_df.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
        model_df["llm_rank"] = model_df.index + 1

        # 显示前10名
        print(f"{'排名':<4} {'团队名称':<25} {'J-Score':<8} {'STA':<6} {'CS':<6} {'FS':<6}")
        print("-" * 60)
        for _, row in model_df.head(10).iterrows():
            print(f"{row['llm_rank']:<4} {row['team_name']:<25} {row['j_score_mean']:<8.3f} "
                  f"{row['sta_mean']:<6.2f} {row['cs_mean']:<6.2f} {row['fs_mean']:<6.2f}")

        # 与人工评价对比
        analyze_model_correlation(model_df, human_df, model_name, evaluation_type)


def analyze_model_correlation(model_df, human_df, model_name, evaluation_type):
    """分析模型与人工评价的相关性"""
    from scipy.stats import spearmanr, pearsonr, kendalltau

    # 团队名称映射
    name_mapping = {
        "Team SINAI": "Team_SINAI",
        "delete": "delete_baseline",
        "mT5": "mt5_baseline",
        "backtranslation": "backtranslation_baseline",
        "Team MarSanAI": "Team MarSan_AI",
        "Team nlpjoyers": "Team nlp_enjoyers",
    }

    # 合并数据进行对比
    comparison_data = []
    for _, model_row in model_df.iterrows():
        team = model_row["team_name"]

        # 查找对应的人工评价
        human_row = human_df[human_df["team_normalized"] == team]

        if len(human_row) > 0:
            comparison_data.append({
                "team_name": team,
                "human_score": human_row.iloc[0]["human_score"],
                "human_rank": human_row.iloc[0]["human_rank"],
                "llm_score": model_row["j_score_mean"],
                "llm_rank": model_row["llm_rank"],
            })

    if len(comparison_data) > 2:
        comparison_df = pd.DataFrame(comparison_data)

        # 计算相关性
        human_scores = comparison_df["human_score"].values
        llm_scores = comparison_df["llm_score"].values

        spearman_corr, spearman_p = spearmanr(human_scores, llm_scores)
        pearson_corr, pearson_p = pearsonr(human_scores, llm_scores)
        kendall_corr, kendall_p = kendalltau(human_scores, llm_scores)

        print(f"\n🔗 {model_name.upper()} 与人工评价的相关性分析:")
        print(f"   • 基于 {len(comparison_df)} 个团队的对比:")
        print(f"   • Spearman相关系数: {spearman_corr:.3f} (p={spearman_p:.3f})")
        print(f"   • Pearson相关系数:  {pearson_corr:.3f} (p={pearson_p:.3f})")
        print(f"   • Kendall's tau:   {kendall_corr:.3f} (p={kendall_p:.3f})")

        # 排名差异分析
        comparison_df["rank_diff"] = comparison_df["llm_rank"] - comparison_df["human_rank"]
        mean_rank_diff = comparison_df["rank_diff"].mean()
        abs_mean_rank_diff = comparison_df["rank_diff"].abs().mean()

        print(f"   • 平均排名差异: {mean_rank_diff:+.1f} 位")
        print(f"   • 绝对排名差异: {abs_mean_rank_diff:.1f} 位")

        # 保存结果
        output_dir = os.path.join("data", "evaluation_results", "new_models_comparison")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{model_name}_{evaluation_type}_detailed_comparison.csv")
        comparison_df.to_csv(output_file, index=False)
        print(f"   📁 详细对比结果已保存到: {output_file}")
    else:
        print(f"   ❌ 数据不足，无法进行相关性分析 (只有 {len(comparison_data)} 个匹配团队)")


def compare_all_evaluations():
    """对比三种评价方法的结果"""

    # 人工评价数据（英文分数）
    human_data = [
        {"team": "SomethingAwful", "human_score": 0.86},
        {"team": "VitalyProtasov", "human_score": 0.69},
        {"team": "nikita.sushko", "human_score": 0.70},
        {"team": "erehulka", "human_score": 0.88},
        {"team": "Team NLPunks", "human_score": 0.84},
        {"team": "mkrisnai", "human_score": 0.89},
        {"team": "Team cake", "human_score": 0.91},
        {"team": "Team SINAI", "human_score": 0.85},
        {"team": "gleb.shnshn", "human_score": 0.74},
        {"team": "delete", "human_score": 0.47},
        {"team": "mT5", "human_score": 0.68},
        {"team": "Team nlp_enjoyers", "human_score": 0.67},
        {"team": "Team Iron Autobots", "human_score": 0.74},
        {"team": "ZhongyuLuo", "human_score": 0.73},
        {"team": "backtranslation", "human_score": 0.73},
        {"team": "Team MarSanAI", "human_score": 0.89},
        {"team": "dkenco", "human_score": 0.68},
    ]

    # 机器评价数据（从图片中提取，英文列）
    machine_data = [
        {"team": "Team SmurfCat", "machine_score": 0.602},
        {"team": "Imeribal", "machine_score": 0.593},
        {"team": "nikita.sushko", "machine_score": 0.553},
        {"team": "VitalyProtasov", "machine_score": 0.531},
        {"team": "erehulka", "machine_score": 0.543},
        {"team": "SomethingAwful", "machine_score": 0.522},
        {"team": "mareksuppa", "machine_score": 0.537},
        {"team": "kofeinix", "machine_score": 0.497},
        {"team": "Yekaterina29", "machine_score": 0.510},
        {"team": "AlekseevArtem", "machine_score": 0.427},
        {"team": "Team NLPunks", "machine_score": 0.489},
        {"team": "pavelshtykov", "machine_score": 0.489},
        {"team": "gleb.shnshn", "machine_score": 0.462},
        {"team": "Volodimirich", "machine_score": 0.472},
        {"team": "ansafronov", "machine_score": 0.506},
        {"team": "MOOsipenko", "machine_score": 0.411},
        {"team": "mkrisnai", "machine_score": 0.475},
        {"team": "Team MarSanAI", "machine_score": 0.504},
        {"team": "Team nlp_enjoyers", "machine_score": 0.418},
        {"team": "Team cake", "machine_score": 0.468},
        {"team": "mT5", "machine_score": 0.418},
        {"team": "gangopsa", "machine_score": 0.472},
        {"team": "Team SINAI", "machine_score": 0.413},
        {"team": "delete", "machine_score": 0.447},
        {"team": "Team Iron Autobots", "machine_score": 0.345},
        {"team": "LanaKlitotekhnis", "machine_score": 0.460},
        {"team": "Anastasia1706", "machine_score": 0.349},
        {"team": "ZhongyuLuo", "machine_score": 0.506},
        {"team": "cocount", "machine_score": 0.271},
        {"team": "backtranslation", "machine_score": 0.506},
        {"team": "etomoscow", "machine_score": 0.293},
        {"team": "cointegrated", "machine_score": 0.160},
        {"team": "dkenco", "machine_score": 0.183},
        {"team": "FD", "machine_score": 0.061},
        {"team": "duplicate", "machine_score": 0.061},
    ]

    # 创建DataFrame
    human_df = pd.DataFrame(human_data)
    machine_df = pd.DataFrame(machine_data)

    # 团队名称映射
    name_mapping = {
        "Team SINAI": "Team_SINAI",
        "delete": "delete_baseline",
        "mT5": "mt5_baseline",
        "backtranslation": "backtranslation_baseline",
        "Team MarSanAI": "Team MarSan_AI",
    }

    # LLM评价文件中的团队名称映射（处理名称不一致问题）
    llm_name_mapping = {"Team nlpjoyers": "Team nlp_enjoyers"}  # LLM文件中缺少下划线

    # 应用名称映射
    human_df["team_normalized"] = (
        human_df["team"].map(name_mapping).fillna(human_df["team"])
    )
    machine_df["team_normalized"] = (
        machine_df["team"].map(name_mapping).fillna(machine_df["team"])
    )

    # 计算排名
    human_df = human_df.sort_values("human_score", ascending=False).reset_index(
        drop=True
    )
    human_df["human_rank"] = human_df.index + 1

    machine_df = machine_df.sort_values("machine_score", ascending=False).reset_index(
        drop=True
    )
    machine_df["machine_rank"] = machine_df.index + 1

    # 检查可用的模型结果
    llm_candidates = [
        "llm_evaluation_ranking_zero_shot.csv",
        os.path.join(
            "data",
            "ranking_results",
            "en",
            "llm_evaluation_ranking_zero_shot.csv",
        ),
        os.path.join(
            "data",
            "result",
            "llm_evaluation",
            "EN",
            "llm_evaluation_ranking_zero_shot.csv",
        ),
    ]

    llm_file = next((p for p in llm_candidates if os.path.exists(p)), None)
    new_model_results = collect_new_model_results("zero_shot")

    # 让用户选择使用哪种模型结果
    print("\n🤖 选择要分析的模型结果:")
    options = []

    if llm_file:
        options.append(("OpenAI模型 (Zero-Shot)", "openai"))
        print(f"1. OpenAI模型 (Zero-Shot)")

    if new_model_results:
        if "deepseek" in new_model_results:
            options.append(("DeepSeek模型 (Zero-Shot)", "deepseek"))
            print(f"{len(options)+1}. DeepSeek模型 (Zero-Shot)")

        if "qwen" in new_model_results:
            options.append(("Qwen模型 (Zero-Shot)", "qwen"))
            print(f"{len(options)+1}. Qwen模型 (Zero-Shot)")

        options.append(("对比所有新模型", "compare_all"))
        print(f"{len(options)+1}. 对比所有新模型")

    if not options:
        print("❌ 未找到任何LLM评价结果文件。请确认是否已生成模型结果。")
        return

    while True:
        try:
            choice = int(input(f"\n请选择 (1-{len(options)}): ").strip())
            if 1 <= choice <= len(options):
                selected_option = options[choice-1][1]
                break
            else:
                print(f"请输入 1-{len(options)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")

    if selected_option == "openai":
        llm_df = pd.read_csv(llm_file)
        print(f"✅ 使用OpenAI模型结果: {len(llm_df)} 个团队")
    elif selected_option == "deepseek":
        llm_df = create_ranking_from_new_model("deepseek", "zero_shot")
    elif selected_option == "qwen":
        llm_df = create_ranking_from_new_model("qwen", "zero_shot")
    elif selected_option == "compare_all":
        compare_all_new_models("zero_shot", human_df, machine_df)
        return

    if llm_df.empty:
        print("❌ 无法加载选择的模型结果")
        return

    # 应用LLM团队名称映射
    llm_df["team_name"] = (
        llm_df["team_name"].map(llm_name_mapping).fillna(llm_df["team_name"])
    )

    llm_df = llm_df.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
    llm_df["llm_rank"] = llm_df.index + 1

    print("🔍 三种评价方法对比分析（以人工评价为准）")
    print("=" * 100)

    # 以人工评价为准，找出有人工评价的团队
    human_teams = set(human_df["team_normalized"])

    print(f"人工评价团队数: {len(human_teams)} 个")

    # 过滤机器评价和LLM评价，只保留有人工评价的团队
    machine_df_filtered = machine_df[
        machine_df["team_normalized"].isin(human_teams)
    ].copy()
    llm_df_filtered = llm_df[llm_df["team_name"].isin(human_teams)].copy()

    # 重新计算机器评价和LLM评价的排名（只针对有人工评价的团队）
    machine_df_filtered = machine_df_filtered.sort_values(
        "machine_score", ascending=False
    ).reset_index(drop=True)
    machine_df_filtered["machine_rank"] = machine_df_filtered.index + 1

    llm_df_filtered = llm_df_filtered.sort_values(
        "j_score_mean", ascending=False
    ).reset_index(drop=True)
    llm_df_filtered["llm_rank"] = llm_df_filtered.index + 1

    print(f"过滤后机器评价团队数: {len(machine_df_filtered)} 个")
    print(f"过滤后LLM评价团队数: {len(llm_df_filtered)} 个")

    # 合并数据
    comparison_data = []

    for _, human_row in human_df.iterrows():
        team = human_row["team_normalized"]

        # 查找对应的机器评价和LLM评价
        machine_row = machine_df_filtered[
            machine_df_filtered["team_normalized"] == team
        ]
        llm_row = llm_df_filtered[llm_df_filtered["team_name"] == team]

        comparison_data.append(
            {
                "team_name": team,
                "human_rank": human_row["human_rank"],
                "human_score": human_row["human_score"],
                "machine_rank": (
                    machine_row.iloc[0]["machine_rank"]
                    if len(machine_row) > 0
                    else None
                ),
                "machine_score": (
                    machine_row.iloc[0]["machine_score"]
                    if len(machine_row) > 0
                    else None
                ),
                "llm_rank": llm_row.iloc[0]["llm_rank"] if len(llm_row) > 0 else None,
                "llm_score": (
                    llm_row.iloc[0]["j_score_mean"] if len(llm_row) > 0 else None
                ),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    # 按人工排名排序显示
    comparison_df = comparison_df.sort_values("human_rank")

    print(f"\n📊 详细对比 (共 {len(comparison_df)} 个团队)")
    print("-" * 100)
    print(
        f"{'团队名称':<25} {'人工排名':<8} {'人工分数':<8} {'机器排名':<8} {'机器分数':<8} {'LLM排名':<8} {'LLM分数':<8}"
    )
    print("-" * 100)

    for _, row in comparison_df.iterrows():
        machine_rank_str = (
            str(int(row["machine_rank"])) if pd.notna(row["machine_rank"]) else "N/A"
        )
        machine_score_str = (
            f"{row['machine_score']:.3f}" if pd.notna(row["machine_score"]) else "N/A"
        )
        llm_rank_str = str(int(row["llm_rank"])) if pd.notna(row["llm_rank"]) else "N/A"
        llm_score_str = (
            f"{row['llm_score']:.3f}" if pd.notna(row["llm_score"]) else "N/A"
        )

        print(
            f"{row['team_name']:<25} {row['human_rank']:<8} {row['human_score']:<8.2f} "
            f"{machine_rank_str:<8} {machine_score_str:<8} "
            f"{llm_rank_str:<8} {llm_score_str:<8}"
        )

    # 详细相关性分析
    print(f"\n🔗 详细相关性分析:")
    print("=" * 80)

    # 过滤有效数据
    valid_machine_df = comparison_df.dropna(subset=["machine_score"])
    valid_llm_df = comparison_df.dropna(subset=["llm_score"])
    valid_all_df = comparison_df.dropna(subset=["machine_score", "llm_score"])

    def calculate_correlations(x, y, name1, name2, n_samples):
        """计算多种相关性指标"""
        if len(x) < 2:
            print(f"   {name1} vs {name2}: 数据不足 (只有 {len(x)} 个样本)")
            return

        # Pearson相关系数 (线性相关)
        pearson_r, pearson_p = pearsonr(x, y)

        # Spearman等级相关系数 (单调相关)
        spearman_r, spearman_p = spearmanr(x, y)

        # Kendall's tau (排名一致性)
        kendall_tau, kendall_p = kendalltau(x, y)

        print(f"\n   📊 {name1} vs {name2} (基于 {n_samples} 个团队):")
        print(f"      • Pearson相关系数:  {pearson_r:.3f} (p={pearson_p:.3f})")
        print(f"      • Spearman等级相关: {spearman_r:.3f} (p={spearman_p:.3f})")
        print(f"      • Kendall's tau:   {kendall_tau:.3f} (p={kendall_p:.3f})")

        # 解释相关性强度
        def interpret_correlation(r):
            abs_r = abs(r)
            if abs_r >= 0.9:
                return "非常强"
            elif abs_r >= 0.7:
                return "强"
            elif abs_r >= 0.5:
                return "中等"
            elif abs_r >= 0.3:
                return "弱"
            else:
                return "很弱"

        print(f"      • 相关性强度: {interpret_correlation(spearman_r)}")

        # 计算预测误差
        mae = mean_absolute_error(x, y)
        rmse = np.sqrt(mean_squared_error(x, y))
        print(f"      • 平均绝对误差(MAE): {mae:.3f}")
        print(f"      • 均方根误差(RMSE): {rmse:.3f}")
        
        # 添加方差和绝对值分析
        print(f"      • 方差分析:")
        x_var = np.var(x, ddof=1)
        y_var = np.var(y, ddof=1)
        x_std = np.std(x, ddof=1)
        y_std = np.std(y, ddof=1)
        print(f"        - {name1}方差: {x_var:.4f} (标准差: {x_std:.3f})")
        print(f"        - {name2}方差: {y_var:.4f} (标准差: {y_std:.3f})")
        
        # 方差比较
        var_ratio = x_var / y_var if y_var > 0 else float('inf')
        if var_ratio > 1.5:
            print(f"        - 方差比较: {name1}变异性更大 (比值: {var_ratio:.2f})")
        elif var_ratio < 0.67:
            print(f"        - 方差比较: {name2}变异性更大 (比值: {var_ratio:.2f})")
        else:
            print(f"        - 方差比较: 变异性相近 (比值: {var_ratio:.2f})")
        
        # 绝对值分析
        print(f"      • 绝对值分析:")
        diff = np.array(x) - np.array(y)
        abs_diff = np.abs(diff)
        mean_abs_diff = np.mean(abs_diff)
        median_abs_diff = np.median(abs_diff)
        max_abs_diff = np.max(abs_diff)
        min_abs_diff = np.min(abs_diff)
        
        # 按照公式(1)和(2)计算 abs 和 var 指标
        # 公式(1): abs = (1/N) * Σ|JS_MER - JS_M| （与平均绝对差值相同）
        print(f"        - abs (公式1): {mean_abs_diff:.3f}")
        
        # 公式(2): var = (1/N) * Σ|JS_MER - JS_M|²  
        var_formula = np.mean((diff) ** 2)
        print(f"        - var (公式2): {var_formula:.3f}")
        
        print(f"        - 平均绝对差值: {mean_abs_diff:.3f}")
        print(f"        - 中位数绝对差值: {median_abs_diff:.3f}")
        print(f"        - 最大绝对差值: {max_abs_diff:.3f}")
        print(f"        - 最小绝对差值: {min_abs_diff:.3f}")
        
        # 绝对差值分布
        small_diff_count = np.sum(abs_diff <= 0.1)
        medium_diff_count = np.sum((abs_diff > 0.1) & (abs_diff <= 0.2))
        large_diff_count = np.sum(abs_diff > 0.2)
        
        print(f"        - 绝对差值分布:")
        print(f"          * 小差异(≤0.1): {small_diff_count} 个 ({small_diff_count/len(abs_diff)*100:.1f}%)")
        print(f"          * 中等差异(0.1-0.2): {medium_diff_count} 个 ({medium_diff_count/len(abs_diff)*100:.1f}%)")
        print(f"          * 大差异(>0.2): {large_diff_count} 个 ({large_diff_count/len(abs_diff)*100:.1f}%)")
        
        # 一致性指标
        consistency_threshold = 0.1  # 可调整的一致性阈值
        consistency_rate = np.sum(abs_diff <= consistency_threshold) / len(abs_diff)
        print(f"        - 一致性率(差异≤{consistency_threshold}): {consistency_rate:.3f}")
        
        # 异常值检测
        q75, q25 = np.percentile(abs_diff, [75, 25])
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        outliers = abs_diff > outlier_threshold
        outlier_count = np.sum(outliers)
        
        if outlier_count > 0:
            print(f"        - 异常差值: {outlier_count} 个 (阈值: {outlier_threshold:.3f})")
            # 找出异常值对应的团队（如果有团队信息的话）
            if outlier_count <= 3:  # 只显示前3个异常值
                outlier_indices = np.where(outliers)[0]
                for idx in outlier_indices[:3]:
                    print(f"          * 索引{idx}: 差值 {abs_diff[idx]:.3f}")
        else:
            print(f"        - 异常差值: 无 (阈值: {outlier_threshold:.3f})")

    # 人工 vs 机器评价
    if len(valid_machine_df) > 1:
        calculate_correlations(
            valid_machine_df["human_score"],
            valid_machine_df["machine_score"],
            "人工评价",
            "机器评价",
            len(valid_machine_df),
        )

    # 人工 vs LLM评价
    if len(valid_llm_df) > 1:
        calculate_correlations(
            valid_llm_df["human_score"],
            valid_llm_df["llm_score"],
            "人工评价",
            "LLM评价",
            len(valid_llm_df),
        )

    # 机器 vs LLM评价
    if len(valid_all_df) > 1:
        calculate_correlations(
            valid_all_df["machine_score"],
            valid_all_df["llm_score"],
            "机器评价",
            "LLM评价",
            len(valid_all_df),
        )

    # 前5名对比
    print(f"\n🏆 前5名对比:")

    print("\n人工评价前5名:")
    top5_human = comparison_df.nsmallest(5, "human_rank")
    for i, (_, row) in enumerate(top5_human.iterrows(), 1):
        print(f"   {i}. {row['team_name']:<25} (分数: {row['human_score']:.2f})")

    print("\n机器评价前5名:")
    valid_machine_for_top5 = comparison_df.dropna(subset=["machine_rank"])
    if len(valid_machine_for_top5) >= 5:
        top5_machine = valid_machine_for_top5.nsmallest(5, "machine_rank")
        for i, (_, row) in enumerate(top5_machine.iterrows(), 1):
            print(f"   {i}. {row['team_name']:<25} (分数: {row['machine_score']:.3f})")
    else:
        print(f"   机器评价数据不足，只有 {len(valid_machine_for_top5)} 个团队")

    print("\nLLM评价前5名:")
    valid_llm_for_top5 = comparison_df.dropna(subset=["llm_rank"])
    if len(valid_llm_for_top5) >= 5:
        top5_llm = valid_llm_for_top5.nsmallest(5, "llm_rank")
        for i, (_, row) in enumerate(top5_llm.iterrows(), 1):
            print(f"   {i}. {row['team_name']:<25} (分数: {row['llm_score']:.3f})")
    else:
        print(f"   LLM评价数据不足，只有 {len(valid_llm_for_top5)} 个团队")

    # 排名差异分析（只对有效数据进行分析）
    comparison_df["human_machine_diff"] = (
        comparison_df["machine_rank"] - comparison_df["human_rank"]
    )
    comparison_df["human_llm_diff"] = (
        comparison_df["llm_rank"] - comparison_df["human_rank"]
    )
    comparison_df["machine_llm_diff"] = (
        comparison_df["llm_rank"] - comparison_df["machine_rank"]
    )

    print(f"\n📈 排名差异统计:")

    # 人工 vs 机器差异
    valid_hm_diff = comparison_df["human_machine_diff"].dropna()
    if len(valid_hm_diff) > 0:
        mean_diff_hm = valid_hm_diff.mean()
        abs_mean_diff_hm = valid_hm_diff.abs().mean()
        print(
            f"   - 人工 vs 机器平均差异: {mean_diff_hm:+.1f} 位 (绝对差异: {abs_mean_diff_hm:.1f} 位)"
        )
        print(
            f"     基于 {len(valid_hm_diff)} 个团队，标准差: {valid_hm_diff.std():.1f}"
        )
    else:
        print(f"   - 人工 vs 机器: 无有效数据")

    # 人工 vs LLM差异
    valid_hl_diff = comparison_df["human_llm_diff"].dropna()
    if len(valid_hl_diff) > 0:
        mean_diff_hl = valid_hl_diff.mean()
        abs_mean_diff_hl = valid_hl_diff.abs().mean()
        print(
            f"   - 人工 vs LLM平均差异: {mean_diff_hl:+.1f} 位 (绝对差异: {abs_mean_diff_hl:.1f} 位)"
        )
        print(
            f"     基于 {len(valid_hl_diff)} 个团队，标准差: {valid_hl_diff.std():.1f}"
        )
    else:
        print(f"   - 人工 vs LLM: 无有效数据")

    # 机器 vs LLM差异
    valid_ml_diff = comparison_df["machine_llm_diff"].dropna()
    if len(valid_ml_diff) > 0:
        mean_diff_ml = valid_ml_diff.mean()
        abs_mean_diff_ml = valid_ml_diff.abs().mean()
        print(
            f"   - 机器 vs LLM平均差异: {mean_diff_ml:+.1f} 位 (绝对差异: {abs_mean_diff_ml:.1f} 位)"
        )
        print(
            f"     基于 {len(valid_ml_diff)} 个团队，标准差: {valid_ml_diff.std():.1f}"
        )
    else:
        print(f"   - 机器 vs LLM: 无有效数据")

    # 找出最大差异
    print(f"\n⚠️  最大排名差异:")

    if len(valid_hm_diff) > 0:
        max_human_machine = comparison_df.loc[
            comparison_df["human_machine_diff"].abs().idxmax()
        ]
        print(
            f"   - 人工 vs 机器: {max_human_machine['team_name']} ({max_human_machine['human_machine_diff']:+.0f}位)"
        )
    else:
        print(f"   - 人工 vs 机器: 无有效数据")

    if len(valid_hl_diff) > 0:
        max_human_llm = comparison_df.loc[
            comparison_df["human_llm_diff"].abs().idxmax()
        ]
        print(
            f"   - 人工 vs LLM: {max_human_llm['team_name']} ({max_human_llm['human_llm_diff']:+.0f}位)"
        )
    else:
        print(f"   - 人工 vs LLM: 无有效数据")

    if len(valid_ml_diff) > 0:
        max_machine_llm = comparison_df.loc[
            comparison_df["machine_llm_diff"].abs().idxmax()
        ]
        print(
            f"   - 机器 vs LLM: {max_machine_llm['team_name']} ({max_machine_llm['machine_llm_diff']:+.0f}位)"
        )
    else:
        print(f"   - 机器 vs LLM: 无有效数据")

    # 排名差异分布分析
    print(f"\n📊 排名差异分布分析:")

    def analyze_ranking_differences(diff_series, name):
        """分析排名差异的分布"""
        if len(diff_series) == 0:
            return

        # 计算不同差异范围的团队数量
        exact_match = len(diff_series[diff_series == 0])
        small_diff = len(diff_series[diff_series.abs() <= 2])
        medium_diff = len(
            diff_series[(diff_series.abs() > 2) & (diff_series.abs() <= 5)]
        )
        large_diff = len(diff_series[diff_series.abs() > 5])

        total = len(diff_series)

        print(f"   {name}:")
        print(
            f"     • 完全一致 (差异=0): {exact_match} 个 ({exact_match/total*100:.1f}%)"
        )
        print(f"     • 小差异 (≤2位): {small_diff} 个 ({small_diff/total*100:.1f}%)")
        print(
            f"     • 中等差异 (3-5位): {medium_diff} 个 ({medium_diff/total*100:.1f}%)"
        )
        print(f"     • 大差异 (>5位): {large_diff} 个 ({large_diff/total*100:.1f}%)")

    if len(valid_hm_diff) > 0:
        analyze_ranking_differences(valid_hm_diff, "人工 vs 机器")

    if len(valid_hl_diff) > 0:
        analyze_ranking_differences(valid_hl_diff, "人工 vs LLM")

    if len(valid_ml_diff) > 0:
        analyze_ranking_differences(valid_ml_diff, "机器 vs LLM")

    # 排名质量评价指标
    print(f"\n📊 排名质量评价指标:")
    print("=" * 80)

    def calculate_ranking_quality_metrics(df):
        """计算排名质量指标 (NDCG, MAP等)"""
        valid_all_evaluations = df.dropna(subset=["machine_rank", "llm_rank"])

        if len(valid_all_evaluations) < 3:
            print("   数据不足，无法计算排名质量指标")
            return

        # 将分数转换为相关性等级 (用于NDCG计算)
        def score_to_relevance(scores, levels=5):
            """将连续分数转换为离散相关性等级"""
            percentiles = np.linspace(0, 100, levels + 1)
            relevance = np.zeros(len(scores))
            for i, score in enumerate(scores):
                for level in range(levels, 0, -1):
                    if score >= np.percentile(scores, percentiles[level - 1]):
                        relevance[i] = level
                        break
            return relevance.astype(int)

        def calculate_dcg(relevance_scores, k=None):
            """计算DCG (Discounted Cumulative Gain)"""
            if k is None:
                k = len(relevance_scores)
            relevance_scores = relevance_scores[:k]
            dcg = relevance_scores[0]
            for i in range(1, len(relevance_scores)):
                dcg += relevance_scores[i] / np.log2(i + 1)
            return dcg

        def calculate_ndcg(true_relevance, predicted_ranking, k=None):
            """计算NDCG (Normalized Discounted Cumulative Gain)"""
            if k is None:
                k = len(true_relevance)

            # 根据预测排名重新排序真实相关性
            predicted_relevance = true_relevance[predicted_ranking[:k]]

            # 计算DCG
            dcg = calculate_dcg(predicted_relevance, k)

            # 计算IDCG (理想DCG)
            ideal_relevance = np.sort(true_relevance)[::-1]
            idcg = calculate_dcg(ideal_relevance, k)

            return dcg / idcg if idcg > 0 else 0

        def calculate_map(true_relevance, predicted_ranking, k=None):
            """计算MAP (Mean Average Precision)"""
            if k is None:
                k = len(true_relevance)

            predicted_relevance = true_relevance[predicted_ranking[:k]]

            if np.sum(predicted_relevance > 0) == 0:
                return 0

            precision_sum = 0
            relevant_count = 0

            for i in range(len(predicted_relevance)):
                if predicted_relevance[i] > 0:
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    precision_sum += precision_at_i

            return (
                precision_sum / np.sum(true_relevance > 0)
                if np.sum(true_relevance > 0) > 0
                else 0
            )

        # 准备数据
        human_scores = valid_all_evaluations["human_score"].values
        machine_scores = valid_all_evaluations["machine_score"].values
        llm_scores = valid_all_evaluations["llm_score"].values

        # 转换为相关性等级
        human_relevance = score_to_relevance(human_scores)

        # 获取排名索引
        human_ranking = np.argsort(-human_scores)  # 降序排列
        machine_ranking = np.argsort(-machine_scores)
        llm_ranking = np.argsort(-llm_scores)

        print(f"   基于 {len(valid_all_evaluations)} 个团队的排名质量分析:")

        # 计算不同K值的NDCG和MAP
        for k in [3, 5, 10]:
            if k <= len(valid_all_evaluations):
                print(f"\n   📈 Top-{k} 排名质量指标:")

                # NDCG
                ndcg_machine = calculate_ndcg(human_relevance, machine_ranking, k)
                ndcg_llm = calculate_ndcg(human_relevance, llm_ranking, k)

                print(f"      • NDCG@{k}:")
                print(f"        - 机器评价: {ndcg_machine:.3f}")
                print(f"        - LLM评价:  {ndcg_llm:.3f}")

                # MAP
                map_machine = calculate_map(human_relevance, machine_ranking, k)
                map_llm = calculate_map(human_relevance, llm_ranking, k)

                print(f"      • MAP@{k}:")
                print(f"        - 机器评价: {map_machine:.3f}")
                print(f"        - LLM评价:  {map_llm:.3f}")

                # 比较哪个方法更好
                if ndcg_machine > ndcg_llm:
                    better_ndcg = "机器评价"
                    ndcg_diff = ndcg_machine - ndcg_llm
                elif ndcg_llm > ndcg_machine:
                    better_ndcg = "LLM评价"
                    ndcg_diff = ndcg_llm - ndcg_machine
                else:
                    better_ndcg = "相等"
                    ndcg_diff = 0

                if map_machine > map_llm:
                    better_map = "机器评价"
                    map_diff = map_machine - map_llm
                elif map_llm > map_machine:
                    better_map = "LLM评价"
                    map_diff = map_llm - map_machine
                else:
                    better_map = "相等"
                    map_diff = 0

                print(f"      • 排名质量比较:")
                print(f"        - NDCG@{k}: {better_ndcg} 更好 (差异: {ndcg_diff:.3f})")
                print(f"        - MAP@{k}:  {better_map} 更好 (差异: {map_diff:.3f})")

        # 整体排名质量评估
        print(f"\n   🏆 整体排名质量评估:")

        # 计算完整排名的NDCG和MAP
        full_ndcg_machine = calculate_ndcg(human_relevance, machine_ranking)
        full_ndcg_llm = calculate_ndcg(human_relevance, llm_ranking)
        full_map_machine = calculate_map(human_relevance, machine_ranking)
        full_map_llm = calculate_map(human_relevance, llm_ranking)

        print(f"      • 完整排名NDCG:")
        print(f"        - 机器评价: {full_ndcg_machine:.3f}")
        print(f"        - LLM评价:  {full_ndcg_llm:.3f}")

        print(f"      • 完整排名MAP:")
        print(f"        - 机器评价: {full_map_machine:.3f}")
        print(f"        - LLM评价:  {full_map_llm:.3f}")

        # 综合评价
        machine_avg = (full_ndcg_machine + full_map_machine) / 2
        llm_avg = (full_ndcg_llm + full_map_llm) / 2

        print(f"\n   🎯 综合排名质量评分:")
        print(f"      • 机器评价综合得分: {machine_avg:.3f}")
        print(f"      • LLM评价综合得分:  {llm_avg:.3f}")

        if machine_avg > llm_avg:
            print(
                f"      • 结论: 机器评价在排名质量上表现更好 (优势: {machine_avg - llm_avg:.3f})"
            )
        elif llm_avg > machine_avg:
            print(
                f"      • 结论: LLM评价在排名质量上表现更好 (优势: {llm_avg - machine_avg:.3f})"
            )
        else:
            print(f"      • 结论: 两种评价方法在排名质量上表现相当")

    calculate_ranking_quality_metrics(comparison_df)

    # 保存结果
    output_dir = os.path.join("data", "evaluation_results", "llm_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_evaluations_comparison.csv")
    comparison_df.to_csv(output_file, index=False)
    print(f"\n📁 对比结果已保存到: {output_file}")

    # 生成评价方法比较总结
    print(f"\n📋 评价方法比较总结:")
    print("=" * 80)

    def generate_evaluation_summary(df):
        """生成评价方法比较的总结报告"""
        valid_machine_df = df.dropna(subset=["machine_score"])
        valid_llm_df = df.dropna(subset=["llm_score"])
        valid_all_df = df.dropna(subset=["machine_score", "llm_score"])

        summary = {
            "data_coverage": {
                "total_teams": len(df),
                "machine_coverage": len(valid_machine_df),
                "llm_coverage": len(valid_llm_df),
                "complete_coverage": len(valid_all_df),
            },
            "correlations": {},
            "reliability": {},
            "ranking_quality": {},
        }

        # 计算相关性指标
        if len(valid_machine_df) > 2:
            spearman_hm, _ = spearmanr(
                valid_machine_df["human_score"], valid_machine_df["machine_score"]
            )
            summary["correlations"]["human_machine"] = spearman_hm

        if len(valid_llm_df) > 2:
            spearman_hl, _ = spearmanr(
                valid_llm_df["human_score"], valid_llm_df["llm_score"]
            )
            summary["correlations"]["human_llm"] = spearman_hl

        if len(valid_all_df) > 2:
            spearman_ml, _ = spearmanr(
                valid_all_df["machine_score"], valid_all_df["llm_score"]
            )
            summary["correlations"]["machine_llm"] = spearman_ml

        print(f"   🔍 数据覆盖情况:")
        print(f"      • 总团队数: {summary['data_coverage']['total_teams']}")
        print(
            f"      • 机器评价覆盖: {summary['data_coverage']['machine_coverage']} ({summary['data_coverage']['machine_coverage']/summary['data_coverage']['total_teams']*100:.1f}%)"
        )
        print(
            f"      • LLM评价覆盖: {summary['data_coverage']['llm_coverage']} ({summary['data_coverage']['llm_coverage']/summary['data_coverage']['total_teams']*100:.1f}%)"
        )
        print(
            f"      • 完整数据覆盖: {summary['data_coverage']['complete_coverage']} ({summary['data_coverage']['complete_coverage']/summary['data_coverage']['total_teams']*100:.1f}%)"
        )

        print(f"\n   📊 相关性表现:")
        if "human_machine" in summary["correlations"]:
            print(
                f"      • 人工 vs 机器: {summary['correlations']['human_machine']:.3f}"
            )
        if "human_llm" in summary["correlations"]:
            print(f"      • 人工 vs LLM:  {summary['correlations']['human_llm']:.3f}")
        if "machine_llm" in summary["correlations"]:
            print(f"      • 机器 vs LLM:  {summary['correlations']['machine_llm']:.3f}")

        # 评价方法推荐
        print(f"\n   💡 评价方法推荐:")

        machine_corr = summary["correlations"].get("human_machine", 0)
        llm_corr = summary["correlations"].get("human_llm", 0)

        if machine_corr > 0 and llm_corr > 0:
            if machine_corr > llm_corr + 0.1:
                print(f"      • 推荐使用: 机器评价 (与人工评价相关性更高)")
                print(f"      • 优势: 客观性强、可重复性好、成本低")
                print(f"      • 注意: 可能无法捕捉语言的细微差别")
            elif llm_corr > machine_corr + 0.1:
                print(f"      • 推荐使用: LLM评价 (与人工评价相关性更高)")
                print(f"      • 优势: 能理解语义、捕捉细微差别")
                print(f"      • 注意: 可能存在偏见、成本较高")
            else:
                print(f"      • 推荐使用: 混合评价方法")
                print(f"      • 建议: 结合机器评价的客观性和LLM评价的语义理解")

        print(f"\n   ⚠️  评价局限性:")
        print(f"      • 人工评价: 主观性、成本高、一致性可能不足")
        print(f"      • 机器评价: 可能忽略语义细节、对创新表达不敏感")
        print(f"      • LLM评价: 可能存在模型偏见、结果不够稳定")

        print(f"\n   🎯 改进建议:")
        print(f"      • 增加评价者数量以提高人工评价的可靠性")
        print(f"      • 使用多种自动评价指标的组合")
        print(f"      • 定期校准自动评价方法与人工评价的一致性")
        print(f"      • 考虑任务特定的评价指标")

        return summary

    generate_evaluation_summary(comparison_df)

    # 高级一致性分析
    print(f"\n🎯 高级一致性分析:")
    print("=" * 80)

    def calculate_ranking_agreement_metrics(df):
        """计算排名一致性指标"""
        valid_all_evaluations = df.dropna(subset=["machine_rank", "llm_rank"])

        if len(valid_all_evaluations) < 3:
            print("   数据不足，无法进行排名一致性分析")
            return

        # 1. 排名相关性 (Ranking Correlation)
        print(f"\n   📈 排名相关性分析 (基于 {len(valid_all_evaluations)} 个团队):")

        human_ranks = valid_all_evaluations["human_rank"].values
        machine_ranks = valid_all_evaluations["machine_rank"].values
        llm_ranks = valid_all_evaluations["llm_rank"].values

        # Spearman排名相关
        spearman_hm, _ = spearmanr(human_ranks, machine_ranks)
        spearman_hl, _ = spearmanr(human_ranks, llm_ranks)
        spearman_ml, _ = spearmanr(machine_ranks, llm_ranks)

        print(f"      • 人工 vs 机器排名相关: {spearman_hm:.3f}")
        print(f"      • 人工 vs LLM排名相关:  {spearman_hl:.3f}")
        print(f"      • 机器 vs LLM排名相关:  {spearman_ml:.3f}")

        # 2. 排名距离 (Ranking Distance)
        print(f"\n   📏 排名距离分析:")

        def kendall_distance(rank1, rank2):
            """计算Kendall距离 (不一致对的数量)"""
            n = len(rank1)
            distance = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if (rank1[i] - rank1[j]) * (rank2[i] - rank2[j]) < 0:
                        distance += 1
            return distance

        def normalized_kendall_distance(rank1, rank2):
            """归一化的Kendall距离"""
            n = len(rank1)
            max_distance = n * (n - 1) // 2
            return (
                kendall_distance(rank1, rank2) / max_distance if max_distance > 0 else 0
            )

        kendall_hm = normalized_kendall_distance(human_ranks, machine_ranks)
        kendall_hl = normalized_kendall_distance(human_ranks, llm_ranks)
        kendall_ml = normalized_kendall_distance(machine_ranks, llm_ranks)

        print(
            f"      • 人工 vs 机器Kendall距离: {kendall_hm:.3f} (0=完全一致, 1=完全不一致)"
        )
        print(f"      • 人工 vs LLMKendall距离:  {kendall_hl:.3f}")
        print(f"      • 机器 vs LLMKendall距离:  {kendall_ml:.3f}")

        # 3. Top-K一致性分析
        print(f"\n   🏆 Top-K一致性分析:")

        def top_k_overlap(rank1, rank2, k):
            """计算Top-K重叠率"""
            top_k_1 = set(np.argsort(rank1)[:k])
            top_k_2 = set(np.argsort(rank2)[:k])
            return len(top_k_1.intersection(top_k_2)) / k

        for k in [3, 5, 10]:
            if k <= len(valid_all_evaluations):
                overlap_hm = top_k_overlap(human_ranks, machine_ranks, k)
                overlap_hl = top_k_overlap(human_ranks, llm_ranks, k)
                overlap_ml = top_k_overlap(machine_ranks, llm_ranks, k)

                print(f"      • Top-{k}重叠率:")
                print(f"        - 人工 vs 机器: {overlap_hm:.3f}")
                print(f"        - 人工 vs LLM:  {overlap_hl:.3f}")
                print(f"        - 机器 vs LLM:  {overlap_ml:.3f}")

        # 4. 一致性团队识别
        print(f"\n   🎯 一致性团队识别:")

        n_teams = len(valid_all_evaluations)

        # 前1/3一致性
        top_third = n_teams // 3
        consistent_top_third = set()

        for _, row in valid_all_evaluations.iterrows():
            if (
                row["human_rank"] <= top_third
                and row["machine_rank"] <= top_third
                and row["llm_rank"] <= top_third
            ):
                consistent_top_third.add(row["team_name"])

        print(f"      • 三种评价都在前1/3的团队: {len(consistent_top_third)} 个")
        if consistent_top_third:
            for team in sorted(consistent_top_third):
                print(f"        * {team}")

        # 前50%一致性
        top_half = n_teams // 2
        consistent_top_half = set()

        for _, row in valid_all_evaluations.iterrows():
            if (
                row["human_rank"] <= top_half
                and row["machine_rank"] <= top_half
                and row["llm_rank"] <= top_half
            ):
                consistent_top_half.add(row["team_name"])

        print(f"      • 三种评价都在前50%的团队: {len(consistent_top_half)} 个")
        if consistent_top_half:
            for team in sorted(consistent_top_half):
                if team not in consistent_top_third:  # 只显示不在前1/3的
                    print(f"        * {team}")

        # 5. 评价方法可靠性分析
        print(f"\n   🔍 评价方法可靠性分析:")

        # 以人工评价为金标准，分析其他方法的可靠性
        def calculate_reliability_metrics(human_scores, auto_scores, method_name):
            """计算自动评价方法的可靠性指标"""
            if len(human_scores) < 3:
                return

            # 相关性
            corr, p_val = spearmanr(human_scores, auto_scores)

            # 平均绝对排名差异
            human_ranks_norm = rankdata(human_scores, method="ordinal")
            auto_ranks_norm = rankdata(auto_scores, method="ordinal")
            mean_rank_diff = np.mean(np.abs(human_ranks_norm - auto_ranks_norm))

            # 一致性比例 (排名差异在±2以内)
            consistency_ratio = np.mean(np.abs(human_ranks_norm - auto_ranks_norm) <= 2)

            print(f"      • {method_name}可靠性:")
            print(f"        - 与人工评价相关性: {corr:.3f} (p={p_val:.3f})")
            print(f"        - 平均排名差异: {mean_rank_diff:.1f} 位")
            print(f"        - 一致性比例(±2位): {consistency_ratio:.3f}")

            # 可靠性等级
            if corr >= 0.8:
                reliability = "高"
            elif corr >= 0.6:
                reliability = "中等"
            elif corr >= 0.4:
                reliability = "较低"
            else:
                reliability = "低"
            print(f"        - 可靠性等级: {reliability}")

        # 分析机器评价可靠性
        machine_data = valid_all_evaluations.dropna(subset=["machine_score"])
        if len(machine_data) > 2:
            calculate_reliability_metrics(
                machine_data["human_score"].values,
                machine_data["machine_score"].values,
                "机器评价",
            )

        # 分析LLM评价可靠性
        llm_data = valid_all_evaluations.dropna(subset=["llm_score"])
        if len(llm_data) > 2:
            calculate_reliability_metrics(
                llm_data["human_score"].values, llm_data["llm_score"].values, "LLM评价"
            )

    calculate_ranking_agreement_metrics(comparison_df)

    # 显示数据完整性统计
    print(f"\n📊 数据完整性统计:")
    print("=" * 50)
    print(f"   - 总团队数 (人工评价): {len(comparison_df)}")
    print(
        f"   - 有机器评价的团队: {len(comparison_df.dropna(subset=['machine_rank']))}"
    )
    print(f"   - 有LLM评价的团队: {len(comparison_df.dropna(subset=['llm_rank']))}")
    print(
        f"   - 三种评价都有的团队: {len(comparison_df.dropna(subset=['machine_rank', 'llm_rank']))}"
    )


if __name__ == "__main__":
    compare_all_evaluations()
