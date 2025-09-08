#!/usr/bin/env python3
"""
对比中文few-shot人工评价、机器评价和LLM评价的结果
基于few-shot评价结果进行对比分析
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def compare_all_evaluations_zh_few_shot():
    """对比中文few-shot四种评价结果：人工、机器、Qwen LLM、DeepSeek LLM、OpenAI LLM（只分析中文的15个队伍）"""

    print("🔍 中文Few-Shot评价方法对比分析（包含多个LLM模型）")
    print("=" * 100)

    # 使用与zero-shot脚本相同的中文人工评价和机器评价数据
    # 人工评价数据（中文分数，来自"Test Phase: Manual Evaluation Final Results"表的 zh 列）
    human_data = [
        {"team": "SomethingAwful", "human_score": 0.53},
        {"team": "VitalyProtasov", "human_score": 0.49},
        {"team": "nikita.sushko", "human_score": 0.47},
        {"team": "erehulka", "human_score": 0.68},
        {"team": "Team NLPunks", "human_score": 0.60},
        {"team": "mkrisnai", "human_score": 0.34},
        {"team": "Team cake", "human_score": 0.84},
        {"team": "Team SINAI", "human_score": 0.33},
        {"team": "gleb.shnshn", "human_score": 0.41},
        {"team": "delete", "human_score": 0.43},
        {"team": "mT5", "human_score": 0.43},
        {"team": "Team nlp_enjoyers", "human_score": 0.23},
        {"team": "Team Iron Autobots", "human_score": 0.53},
        {"team": "ZhongyuLuo", "human_score": 0.56},
        {"team": "backtranslation", "human_score": 0.34},
    ]

    # 机器评价数据（中文分数，来自"Test Phase: Automatic Evaluation Results"表的 zh 列）
    machine_data = [
        {"team": "nikita.sushko", "machine_score": 0.176},
        {"team": "VitalyProtasov", "machine_score": 0.175},
        {"team": "erehulka", "machine_score": 0.160},
        {"team": "SomethingAwful", "machine_score": 0.147},
        {"team": "mkrisnai", "machine_score": 0.109},
        {"team": "Team NLPunks", "machine_score": 0.150},
        {"team": "gleb.shnshn", "machine_score": 0.155},
        {"team": "Team cake", "machine_score": 0.086},
        {"team": "mT5", "machine_score": 0.096},
        {"team": "Team nlp_enjoyers", "machine_score": 0.104},
        {"team": "Team SINAI", "machine_score": 0.126},
        {"team": "delete", "machine_score": 0.175},
        {"team": "Team Iron Autobots", "machine_score": 0.124},
        {"team": "ZhongyuLuo", "machine_score": 0.052},
        {"team": "backtranslation", "machine_score": 0.027},
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
    }

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

    # 读取中文LLM Few-Shot评价结果
    llm_qwen_file = os.path.join("data", "ranking_results", "zh", "llm_evaluation_ranking_few_shot_qwen.csv")
    llm_deepseek_file = os.path.join("data", "ranking_results", "zh", "llm_evaluation_ranking_few_shot_deepseek.csv")
    llm_openai_file = os.path.join("data", "ranking_results", "zh", "llm_evaluation_ranking_few_shot_openai.csv")

    if not os.path.exists(llm_qwen_file):
        print("❌ 未找到中文Qwen LLM Few-Shot评价结果文件。请先运行: python scripts/generate_zh_llm_ranking_few_shot.py")
        return

    if not os.path.exists(llm_deepseek_file):
        print("❌ 未找到中文DeepSeek LLM Few-Shot评价结果文件。请先运行: python scripts/generate_zh_llm_ranking_deepseek_few_shot.py")
        return

    if not os.path.exists(llm_openai_file):
        print("❌ 未找到中文OpenAI LLM Few-Shot评价结果文件。请先运行: python scripts/generate_zh_llm_ranking_openai.py")
        return

    llm_qwen_df = pd.read_csv(llm_qwen_file)
    llm_deepseek_df = pd.read_csv(llm_deepseek_file)
    llm_openai_df = pd.read_csv(llm_openai_file)

    # LLM名称映射（处理名称不一致问题）
    llm_name_mapping = {"Team nlpjoyers": "Team nlp_enjoyers"}

    # 应用LLM名称映射
    llm_qwen_df["team_name"] = (
        llm_qwen_df["team_name"].map(llm_name_mapping).fillna(llm_qwen_df["team_name"])
    )
    llm_deepseek_df["team_name"] = (
        llm_deepseek_df["team_name"].map(llm_name_mapping).fillna(llm_deepseek_df["team_name"])
    )
    llm_openai_df["team_name"] = (
        llm_openai_df["team_name"].map(llm_name_mapping).fillna(llm_openai_df["team_name"])
    )

    # 获取有人工评价的团队
    human_teams = set(human_df["team_normalized"])

    print(f"人工评价团队数: {len(human_teams)} 个")

    # 过滤各种评价，只保留有人工评价的团队
    machine_df_filtered = machine_df[machine_df["team_normalized"].isin(human_teams)].copy()
    llm_qwen_df_filtered = llm_qwen_df[llm_qwen_df["team_name"].isin(human_teams)].copy()
    llm_deepseek_df_filtered = llm_deepseek_df[llm_deepseek_df["team_name"].isin(human_teams)].copy()
    llm_openai_df_filtered = llm_openai_df[llm_openai_df["team_name"].isin(human_teams)].copy()

    # 重新计算排名（只针对有人工评价的团队）
    machine_df_filtered = machine_df_filtered.sort_values("machine_score", ascending=False).reset_index(drop=True)
    machine_df_filtered["machine_rank"] = machine_df_filtered.index + 1

    llm_qwen_df_filtered = llm_qwen_df_filtered.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
    llm_qwen_df_filtered["llm_qwen_rank"] = llm_qwen_df_filtered.index + 1

    llm_deepseek_df_filtered = llm_deepseek_df_filtered.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
    llm_deepseek_df_filtered["llm_deepseek_rank"] = llm_deepseek_df_filtered.index + 1

    llm_openai_df_filtered = llm_openai_df_filtered.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
    llm_openai_df_filtered["llm_openai_rank"] = llm_openai_df_filtered.index + 1

    print(f"过滤后机器评价团队数: {len(machine_df_filtered)} 个")
    print(f"过滤后Qwen LLM评价团队数: {len(llm_qwen_df_filtered)} 个")
    print(f"过滤后DeepSeek LLM评价团队数: {len(llm_deepseek_df_filtered)} 个")
    print(f"过滤后OpenAI LLM评价团队数: {len(llm_openai_df_filtered)} 个")

    print(f"🔍 五种评价方法对比分析（中文Few-Shot数据，以人工评价为准，共{len(human_df)}个团队）")
    print("=" * 160)

    # 合并数据
    comparison_data = []

    for _, human_row in human_df.iterrows():
        team = human_row["team_normalized"]

        # 查找对应的各种评价
        machine_row = machine_df_filtered[machine_df_filtered["team_normalized"] == team]
        llm_qwen_row = llm_qwen_df_filtered[llm_qwen_df_filtered["team_name"] == team]
        llm_deepseek_row = llm_deepseek_df_filtered[llm_deepseek_df_filtered["team_name"] == team]
        llm_openai_row = llm_openai_df_filtered[llm_openai_df_filtered["team_name"] == team]

        comparison_data.append({
            "team_name": team,
            "human_rank": human_row["human_rank"],
            "human_score": human_row["human_score"],
            "machine_rank": machine_row.iloc[0]["machine_rank"] if len(machine_row) > 0 else None,
            "machine_score": machine_row.iloc[0]["machine_score"] if len(machine_row) > 0 else None,
            "llm_qwen_rank": llm_qwen_row.iloc[0]["llm_qwen_rank"] if len(llm_qwen_row) > 0 else None,
            "llm_qwen_score": llm_qwen_row.iloc[0]["j_score_mean"] if len(llm_qwen_row) > 0 else None,
            "llm_deepseek_rank": llm_deepseek_row.iloc[0]["llm_deepseek_rank"] if len(llm_deepseek_row) > 0 else None,
            "llm_deepseek_score": llm_deepseek_row.iloc[0]["j_score_mean"] if len(llm_deepseek_row) > 0 else None,
            "llm_openai_rank": llm_openai_row.iloc[0]["llm_openai_rank"] if len(llm_openai_row) > 0 else None,
            "llm_openai_score": llm_openai_row.iloc[0]["j_score_mean"] if len(llm_openai_row) > 0 else None,
        })

    comparison_df = pd.DataFrame(comparison_data)
    
    # 按人工排名排序显示
    comparison_df = comparison_df.sort_values("human_rank")

    # 显示对比结果
    print(f"{'团队名称':<25} {'人工排名':<8} {'人工分数':<8} {'机器排名':<8} {'机器分数':<8} {'Qwen排名':<8} {'Qwen分数':<8} {'DeepSeek排名':<12} {'DeepSeek分数':<12} {'OpenAI排名':<10} {'OpenAI分数':<10}")
    print("-" * 160)

    for _, row in comparison_df.iterrows():
        qwen_rank_str = str(int(row["llm_qwen_rank"])) if pd.notna(row["llm_qwen_rank"]) else "N/A"
        qwen_score_str = f"{row['llm_qwen_score']:.4f}" if pd.notna(row["llm_qwen_score"]) else "N/A"

        deepseek_rank_str = str(int(row["llm_deepseek_rank"])) if pd.notna(row["llm_deepseek_rank"]) else "N/A"
        deepseek_score_str = f"{row['llm_deepseek_score']:.4f}" if pd.notna(row["llm_deepseek_score"]) else "N/A"

        openai_rank_str = str(int(row["llm_openai_rank"])) if pd.notna(row["llm_openai_rank"]) else "N/A"
        openai_score_str = f"{row['llm_openai_score']:.4f}" if pd.notna(row["llm_openai_score"]) else "N/A"

        print(f"{row['team_name']:<25} {row['human_rank']:<8} {row['human_score']:<8.2f} "
              f"{row['machine_rank']:<8} {row['machine_score']:<8.3f} "
              f"{qwen_rank_str:<8} {qwen_score_str:<8} "
              f"{deepseek_rank_str:<12} {deepseek_score_str:<12} "
              f"{openai_rank_str:<10} {openai_score_str:<10}")

    # 保存结果
    output_dir = os.path.join("data", "evaluation_results", "llm_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_evaluations_comparison_zh_few_shot_all_models.csv")
    comparison_df.to_csv(output_file, index=False)
    print(f"\n📁 中文Few-Shot对比结果（包含所有模型）已保存到: {output_file}")

    # 详细相关性分析
    print(f"\n🔗 详细相关性分析:")
    print("=" * 80)

    # 过滤有效数据
    valid_machine_df = comparison_df.dropna(subset=["machine_score"])
    valid_llm_qwen_df = comparison_df.dropna(subset=["llm_qwen_score"])
    valid_llm_deepseek_df = comparison_df.dropna(subset=["llm_deepseek_score"])
    valid_llm_openai_df = comparison_df.dropna(subset=["llm_openai_score"])
    valid_all_df = comparison_df.dropna(subset=["machine_score", "llm_qwen_score", "llm_deepseek_score", "llm_openai_score"])

    def interpret_correlation(r):
        """解释相关系数强度"""
        abs_r = abs(r)
        if abs_r >= 0.8:
            return "非常强"
        elif abs_r >= 0.6:
            return "强"
        elif abs_r >= 0.4:
            return "中等"
        elif abs_r >= 0.2:
            return "弱"
        else:
            return "很弱"

    def calculate_detailed_stats(x, y, name1, name2, sample_size):
        """计算并显示详细统计指标"""
        print(f"\n📊 {name1} vs {name2} (样本数: {sample_size})")
        print("-" * 60)

        # 基本统计
        x_mean, y_mean = np.mean(x), np.mean(y)
        x_std, y_std = np.std(x), np.std(y)
        x_var, y_var = np.var(x), np.var(y)

        print(f"      • 基本统计:")
        print(f"        - {name1}: 均值={x_mean:.3f}, 标准差={x_std:.3f}")
        print(f"        - {name2}: 均值={y_mean:.3f}, 标准差={y_std:.3f}")

        # 相关性分析
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        kendall_tau, kendall_p = kendalltau(x, y)

        print(f"      • 相关性分析:")
        print(f"        - Pearson相关系数: {pearson_r:.3f} (p={pearson_p:.3f})")
        print(f"        - Spearman相关系数: {spearman_r:.3f} (p={spearman_p:.3f})")
        print(f"        - Kendall's tau: {kendall_tau:.3f} (p={kendall_p:.3f})")
        print(f"        - 相关性强度: {interpret_correlation(spearman_r)}")

        # 预测误差
        mae = mean_absolute_error(x, y)
        rmse = np.sqrt(mean_squared_error(x, y))
        print(f"        - 平均绝对误差(MAE): {mae:.3f}")
        print(f"        - 均方根误差(RMSE): {rmse:.3f}")

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
            if outlier_count <= 3:  # 只显示前3个异常值
                outlier_indices = np.where(outliers)[0]
                for idx in outlier_indices[:3]:
                    print(f"          * 索引{idx}: 差值 {abs_diff[idx]:.3f}")
        else:
            print(f"        - 异常差值: 无 (阈值: {outlier_threshold:.3f})")

    # 人工 vs 机器评价
    if len(valid_machine_df) > 1:
        calculate_detailed_stats(
            valid_machine_df["human_score"],
            valid_machine_df["machine_score"],
            "人工评价",
            "机器评价",
            len(valid_machine_df),
        )

    # 人工 vs 各个LLM评价
    if len(valid_llm_qwen_df) > 1:
        calculate_detailed_stats(
            valid_llm_qwen_df["human_score"],
            valid_llm_qwen_df["llm_qwen_score"],
            "人工评价",
            "Qwen LLM评价",
            len(valid_llm_qwen_df),
        )

    if len(valid_llm_deepseek_df) > 1:
        calculate_detailed_stats(
            valid_llm_deepseek_df["human_score"],
            valid_llm_deepseek_df["llm_deepseek_score"],
            "人工评价",
            "DeepSeek LLM评价",
            len(valid_llm_deepseek_df),
        )

    if len(valid_llm_openai_df) > 1:
        calculate_detailed_stats(
            valid_llm_openai_df["human_score"],
            valid_llm_openai_df["llm_openai_score"],
            "人工评价",
            "OpenAI LLM评价",
            len(valid_llm_openai_df),
        )

    # 机器 vs 各个LLM评价
    if len(valid_all_df) > 1:
        calculate_detailed_stats(
            valid_all_df["machine_score"],
            valid_all_df["llm_qwen_score"],
            "机器评价",
            "Qwen LLM评价",
            len(valid_all_df),
        )

        calculate_detailed_stats(
            valid_all_df["machine_score"],
            valid_all_df["llm_deepseek_score"],
            "机器评价",
            "DeepSeek LLM评价",
            len(valid_all_df),
        )

        calculate_detailed_stats(
            valid_all_df["machine_score"],
            valid_all_df["llm_openai_score"],
            "机器评价",
            "OpenAI LLM评价",
            len(valid_all_df),
        )

        # LLM之间的对比
        calculate_detailed_stats(
            valid_all_df["llm_qwen_score"],
            valid_all_df["llm_deepseek_score"],
            "Qwen LLM评价",
            "DeepSeek LLM评价",
            len(valid_all_df),
        )

        calculate_detailed_stats(
            valid_all_df["llm_qwen_score"],
            valid_all_df["llm_openai_score"],
            "Qwen LLM评价",
            "OpenAI LLM评价",
            len(valid_all_df),
        )

        calculate_detailed_stats(
            valid_all_df["llm_deepseek_score"],
            valid_all_df["llm_openai_score"],
            "DeepSeek LLM评价",
            "OpenAI LLM评价",
            len(valid_all_df),
        )

    # 显示前5名对比
    print(f"\n🏆 各评价方法前5名对比:")
    print("=" * 80)

    print("人工评价前5名:")
    for i, (_, row) in enumerate(comparison_df.head(5).iterrows(), 1):
        print(f"   {i}. {row['team_name']:<25} (分数: {row['human_score']:.3f})")

    print("\n机器评价前5名:")
    valid_machine_for_top5 = comparison_df.dropna(subset=["machine_rank"])
    if len(valid_machine_for_top5) >= 5:
        top5_machine = valid_machine_for_top5.nsmallest(5, "machine_rank")
        for i, (_, row) in enumerate(top5_machine.iterrows(), 1):
            print(f"   {i}. {row['team_name']:<25} (分数: {row['machine_score']:.3f})")
    else:
        print(f"   机器评价数据不足，只有 {len(valid_machine_for_top5)} 个团队")

    print("\nQwen LLM评价前5名:")
    valid_qwen_for_top5 = comparison_df.dropna(subset=["llm_qwen_rank"])
    if len(valid_qwen_for_top5) >= 5:
        top5_qwen = valid_qwen_for_top5.nsmallest(5, "llm_qwen_rank")
        for i, (_, row) in enumerate(top5_qwen.iterrows(), 1):
            print(f"   {i}. {row['team_name']:<25} (分数: {row['llm_qwen_score']:.4f})")
    else:
        print(f"   Qwen LLM评价数据不足，只有 {len(valid_qwen_for_top5)} 个团队")

    print("\nDeepSeek LLM评价前5名:")
    valid_deepseek_for_top5 = comparison_df.dropna(subset=["llm_deepseek_rank"])
    if len(valid_deepseek_for_top5) >= 5:
        top5_deepseek = valid_deepseek_for_top5.nsmallest(5, "llm_deepseek_rank")
        for i, (_, row) in enumerate(top5_deepseek.iterrows(), 1):
            print(f"   {i}. {row['team_name']:<25} (分数: {row['llm_deepseek_score']:.4f})")
    else:
        print(f"   DeepSeek LLM评价数据不足，只有 {len(valid_deepseek_for_top5)} 个团队")

    print("\nOpenAI LLM评价前5名:")
    valid_openai_for_top5 = comparison_df.dropna(subset=["llm_openai_rank"])
    if len(valid_openai_for_top5) >= 5:
        top5_openai = valid_openai_for_top5.nsmallest(5, "llm_openai_rank")
        for i, (_, row) in enumerate(top5_openai.iterrows(), 1):
            print(f"   {i}. {row['team_name']:<25} (分数: {row['llm_openai_score']:.4f})")
    else:
        print(f"   OpenAI LLM评价数据不足，只有 {len(valid_openai_for_top5)} 个团队")


    # 显示数据完整性统计
    print(f"\n📊 数据完整性统计:")
    print("=" * 50)
    print(f"   - 总团队数 (人工评价): {len(comparison_df)}")
    print(f"   - 有机器评价的团队: {len(comparison_df.dropna(subset=['machine_rank']))}")
    print(f"   - 有Qwen LLM评价的团队: {len(comparison_df.dropna(subset=['llm_qwen_rank']))}")
    print(f"   - 有DeepSeek LLM评价的团队: {len(comparison_df.dropna(subset=['llm_deepseek_rank']))}")
    print(f"   - 有OpenAI LLM评价的团队: {len(comparison_df.dropna(subset=['llm_openai_rank']))}")
    print(f"   - 五种评价都有的团队: {len(comparison_df.dropna(subset=['machine_rank', 'llm_qwen_rank', 'llm_deepseek_rank', 'llm_openai_rank']))}")

    return comparison_df

if __name__ == "__main__":
    compare_all_evaluations_zh_few_shot()