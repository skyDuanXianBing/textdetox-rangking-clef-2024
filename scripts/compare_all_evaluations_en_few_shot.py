#!/usr/bin/env python3
"""
对比人工评价、LLM评价(Few-Shot)和机器评价的结果
包含多种一致性评价指标和评价质量分析
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import kendalltau, spearmanr, pearsonr
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


def compare_all_evaluations_few_shot():
    """对比三种评价方法的结果 (使用Few-Shot LLM评价)"""

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
    llm_file = os.path.join("data", "ranking_results", "en", "llm_evaluation_ranking_few_shot.csv")
    new_model_results = collect_new_model_results("few_shot")

    # 让用户选择使用哪种模型结果
    print("\n🤖 选择要分析的模型结果 (Few-Shot):")
    options = []

    if os.path.exists(llm_file):
        options.append(("OpenAI模型 (Few-Shot)", "openai"))
        print(f"1. OpenAI模型 (Few-Shot)")

    if new_model_results:
        if "deepseek" in new_model_results:
            options.append(("DeepSeek模型 (Few-Shot)", "deepseek"))
            print(f"{len(options)+1}. DeepSeek模型 (Few-Shot)")

        if "qwen" in new_model_results:
            options.append(("Qwen模型 (Few-Shot)", "qwen"))
            print(f"{len(options)+1}. Qwen模型 (Few-Shot)")

        options.append(("对比所有新模型", "compare_all"))
        print(f"{len(options)+1}. 对比所有新模型")

    if not options:
        print("❌ 未找到任何LLM Few-Shot评价结果文件。请确认是否已生成模型结果。")
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
        print(f"✅ 使用OpenAI Few-Shot模型结果: {len(llm_df)} 个团队")
    elif selected_option == "deepseek":
        llm_df = create_ranking_from_new_model("deepseek", "few_shot")
    elif selected_option == "qwen":
        llm_df = create_ranking_from_new_model("qwen", "few_shot")
    elif selected_option == "compare_all":
        compare_all_new_models("few_shot", human_df, machine_df)
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

    print("🔍 三种评价方法对比分析（Few-Shot LLM评价，以人工评价为准）")
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

    print(f"\n📊 详细对比 (共 {len(comparison_df)} 个团队) - Few-Shot LLM评价")
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
    print(f"\n🔗 详细相关性分析 (Few-Shot):")
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
            "LLM评价(Few-Shot)",
            len(valid_llm_df),
        )

    # 机器 vs LLM评价
    if len(valid_all_df) > 1:
        calculate_correlations(
            valid_all_df["machine_score"],
            valid_all_df["llm_score"],
            "机器评价",
            "LLM评价(Few-Shot)",
            len(valid_all_df),
        )

    # 前5名对比
    print(f"\n🏆 前5名对比 (Few-Shot):")

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

    print("\nLLM评价(Few-Shot)前5名:")
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

    print(f"\n📈 排名差异统计 (Few-Shot):")

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
    print(f"\n⚠️  最大排名差异 (Few-Shot):")

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
    print(f"\n📊 排名差异分布分析 (Few-Shot):")

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

    # 保存结果
    output_dir = os.path.join("data", "evaluation_results", "llm_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_evaluations_comparison_few_shot.csv")
    comparison_df.to_csv(output_file, index=False)
    print(f"\n📁 Few-Shot对比结果已保存到: {output_file}")

    print(f"\n📋 Few-Shot评价方法比较总结:")
    print("=" * 80)

    # 计算相关性指标
    if len(valid_llm_df) > 2:
        spearman_hl, _ = spearmanr(
            valid_llm_df["human_score"], valid_llm_df["llm_score"]
        )
        print(f"   📊 人工 vs LLM(Few-Shot)相关性: {spearman_hl:.3f}")
        
        if spearman_hl > 0.7:
            print(f"   💡 Few-Shot LLM评价与人工评价高度一致")
        elif spearman_hl > 0.5:
            print(f"   💡 Few-Shot LLM评价与人工评价中等一致")
        else:
            print(f"   ⚠️  Few-Shot LLM评价与人工评价一致性较低")

    # 数据完整性统计
    print(f"\n📊 数据完整性统计 (Few-Shot):")
    print("=" * 50)
    print(f"   - 总团队数 (人工评价): {len(comparison_df)}")
    print(f"   - 有机器评价的团队: {len(comparison_df.dropna(subset=['machine_score']))}")
    print(f"   - 有LLM评价的团队: {len(comparison_df.dropna(subset=['llm_score']))}")
    print(f"   - 三种评价都有的团队: {len(comparison_df.dropna(subset=['machine_score', 'llm_score']))}")

    # Zero-Shot vs Few-Shot 对比提示
    print(f"\n💡 Few-Shot vs Zero-Shot 对比提示:")
    print("=" * 50)
    print(f"   - 要对比 Zero-Shot 结果，请运行: python scripts/compare_all_evaluations_en.py")
    print(f"   - Few-Shot 结果文件: data/evaluation_results/llm_evaluation/all_evaluations_comparison_few_shot.csv")
    print(f"   - Zero-Shot 结果文件: data/evaluation_results/llm_evaluation/all_evaluations_comparison.csv")

if __name__ == "__main__":
    compare_all_evaluations_few_shot()