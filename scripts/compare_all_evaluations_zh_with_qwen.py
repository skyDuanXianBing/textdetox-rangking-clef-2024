#!/usr/bin/env python3
"""
对比中文人工评价、机器评价和两个LLM评价的结果
包含原始LLM模型和Qwen模型的对比分析
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def compare_all_evaluations_zh_with_qwen():
    """对比五种评价方法的中文结果：人工、机器、原始LLM、Qwen LLM、DeepSeek LLM"""

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

    # LLM评价文件中的团队名称映射（处理名称不一致问题）
    llm_name_mapping = {"Team nlpjoyers": "Team nlp_enjoyers"}

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

    # 读取原始LLM评价结果
    llm_original_file = os.path.join("data", "ranking_results", "zh", "llm_evaluation_ranking_zero_shot.csv")
    if not os.path.exists(llm_original_file):
        print("❌ 未找到原始LLM评价结果文件。请先运行: python scripts/generate_zh_llm_ranking.py")
        return

    llm_original_df = pd.read_csv(llm_original_file)
    llm_original_df["team_name"] = (
        llm_original_df["team_name"].map(llm_name_mapping).fillna(llm_original_df["team_name"])
    )
    llm_original_df = llm_original_df.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
    llm_original_df["llm_original_rank"] = llm_original_df.index + 1

    # 读取Qwen LLM评价结果
    llm_qwen_file = os.path.join("data", "ranking_results", "zh", "llm_evaluation_ranking_zero_shot_qwen.csv")
    if not os.path.exists(llm_qwen_file):
        print("❌ 未找到Qwen LLM评价结果文件。请先运行: python scripts/generate_zh_llm_ranking_qwen.py")
        return

    llm_qwen_df = pd.read_csv(llm_qwen_file)
    llm_qwen_df["team_name"] = (
        llm_qwen_df["team_name"].map(llm_name_mapping).fillna(llm_qwen_df["team_name"])
    )
    llm_qwen_df = llm_qwen_df.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
    llm_qwen_df["llm_qwen_rank"] = llm_qwen_df.index + 1

    # 读取DeepSeek LLM评价结果
    llm_deepseek_file = os.path.join("data", "ranking_results", "zh", "llm_evaluation_ranking_zero_shot_deepseek.csv")
    if not os.path.exists(llm_deepseek_file):
        print("❌ 未找到DeepSeek LLM评价结果文件。请先运行: python scripts/generate_zh_llm_ranking_deepseek.py")
        return

    llm_deepseek_df = pd.read_csv(llm_deepseek_file)
    llm_deepseek_df["team_name"] = (
        llm_deepseek_df["team_name"].map(llm_name_mapping).fillna(llm_deepseek_df["team_name"])
    )
    llm_deepseek_df = llm_deepseek_df.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
    llm_deepseek_df["llm_deepseek_rank"] = llm_deepseek_df.index + 1

    print("🔍 五种评价方法对比分析（中文数据，以人工评价为准）")
    print("=" * 140)

    # 以人工评价为准，找出有人工评价的团队
    human_teams = set(human_df["team_normalized"])

    print(f"人工评价团队数: {len(human_teams)} 个")

    # 过滤各种评价，只保留有人工评价的团队
    machine_df_filtered = machine_df[machine_df["team_normalized"].isin(human_teams)].copy()
    llm_original_df_filtered = llm_original_df[llm_original_df["team_name"].isin(human_teams)].copy()
    llm_qwen_df_filtered = llm_qwen_df[llm_qwen_df["team_name"].isin(human_teams)].copy()
    llm_deepseek_df_filtered = llm_deepseek_df[llm_deepseek_df["team_name"].isin(human_teams)].copy()

    # 重新计算排名（只针对有人工评价的团队）
    machine_df_filtered = machine_df_filtered.sort_values("machine_score", ascending=False).reset_index(drop=True)
    machine_df_filtered["machine_rank"] = machine_df_filtered.index + 1

    llm_original_df_filtered = llm_original_df_filtered.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
    llm_original_df_filtered["llm_original_rank"] = llm_original_df_filtered.index + 1

    llm_qwen_df_filtered = llm_qwen_df_filtered.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
    llm_qwen_df_filtered["llm_qwen_rank"] = llm_qwen_df_filtered.index + 1

    llm_deepseek_df_filtered = llm_deepseek_df_filtered.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
    llm_deepseek_df_filtered["llm_deepseek_rank"] = llm_deepseek_df_filtered.index + 1

    print(f"过滤后机器评价团队数: {len(machine_df_filtered)} 个")
    print(f"过滤后原始LLM评价团队数: {len(llm_original_df_filtered)} 个")
    print(f"过滤后Qwen LLM评价团队数: {len(llm_qwen_df_filtered)} 个")
    print(f"过滤后DeepSeek LLM评价团队数: {len(llm_deepseek_df_filtered)} 个")

    # 合并数据
    comparison_data = []

    for _, human_row in human_df.iterrows():
        team = human_row["team_normalized"]

        # 查找对应的各种评价
        machine_row = machine_df_filtered[machine_df_filtered["team_normalized"] == team]
        llm_original_row = llm_original_df_filtered[llm_original_df_filtered["team_name"] == team]
        llm_qwen_row = llm_qwen_df_filtered[llm_qwen_df_filtered["team_name"] == team]
        llm_deepseek_row = llm_deepseek_df_filtered[llm_deepseek_df_filtered["team_name"] == team]

        comparison_data.append({
            "team_name": team,
            "human_rank": human_row["human_rank"],
            "human_score": human_row["human_score"],
            "machine_rank": machine_row.iloc[0]["machine_rank"] if len(machine_row) > 0 else None,
            "machine_score": machine_row.iloc[0]["machine_score"] if len(machine_row) > 0 else None,
            "llm_original_rank": llm_original_row.iloc[0]["llm_original_rank"] if len(llm_original_row) > 0 else None,
            "llm_original_score": llm_original_row.iloc[0]["j_score_mean"] if len(llm_original_row) > 0 else None,
            "llm_qwen_rank": llm_qwen_row.iloc[0]["llm_qwen_rank"] if len(llm_qwen_row) > 0 else None,
            "llm_qwen_score": llm_qwen_row.iloc[0]["j_score_mean"] if len(llm_qwen_row) > 0 else None,
            "llm_deepseek_rank": llm_deepseek_row.iloc[0]["llm_deepseek_rank"] if len(llm_deepseek_row) > 0 else None,
            "llm_deepseek_score": llm_deepseek_row.iloc[0]["j_score_mean"] if len(llm_deepseek_row) > 0 else None,
        })

    comparison_df = pd.DataFrame(comparison_data)

    # 按人工排名排序显示
    comparison_df = comparison_df.sort_values("human_rank")

    print(f"\n📊 详细对比 (共 {len(comparison_df)} 个团队)")
    print("-" * 160)
    print(f"{'团队名称':<25} {'人工排名':<8} {'人工分数':<8} {'机器排名':<8} {'机器分数':<8} {'原始LLM排名':<12} {'原始LLM分数':<12} {'Qwen排名':<8} {'Qwen分数':<8} {'DeepSeek排名':<12} {'DeepSeek分数':<12}")
    print("-" * 160)

    for _, row in comparison_df.iterrows():
        machine_rank_str = str(int(row["machine_rank"])) if pd.notna(row["machine_rank"]) else "N/A"
        machine_score_str = f"{row['machine_score']:.3f}" if pd.notna(row["machine_score"]) else "N/A"
        
        llm_original_rank_str = str(int(row["llm_original_rank"])) if pd.notna(row["llm_original_rank"]) else "N/A"
        llm_original_score_str = f"{row['llm_original_score']:.3f}" if pd.notna(row["llm_original_score"]) else "N/A"
        
        llm_qwen_rank_str = str(int(row["llm_qwen_rank"])) if pd.notna(row["llm_qwen_rank"]) else "N/A"
        llm_qwen_score_str = f"{row['llm_qwen_score']:.3f}" if pd.notna(row["llm_qwen_score"]) else "N/A"

        llm_deepseek_rank_str = str(int(row["llm_deepseek_rank"])) if pd.notna(row["llm_deepseek_rank"]) else "N/A"
        llm_deepseek_score_str = f"{row['llm_deepseek_score']:.3f}" if pd.notna(row["llm_deepseek_score"]) else "N/A"

        print(f"{row['team_name']:<25} {row['human_rank']:<8} {row['human_score']:<8.2f} "
              f"{machine_rank_str:<8} {machine_score_str:<8} "
              f"{llm_original_rank_str:<12} {llm_original_score_str:<12} "
              f"{llm_qwen_rank_str:<8} {llm_qwen_score_str:<8} "
              f"{llm_deepseek_rank_str:<12} {llm_deepseek_score_str:<12}")

    # 保存结果
    output_dir = os.path.join("data", "evaluation_results", "llm_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_evaluations_comparison_zh_with_all_models.csv")
    comparison_df.to_csv(output_file, index=False)
    print(f"\n📁 中文对比结果（包含所有模型）已保存到: {output_file}")

    # 详细相关性分析
    print(f"\n🔗 详细相关性分析:")
    print("=" * 80)

    # 过滤有效数据
    valid_machine_df = comparison_df.dropna(subset=["machine_score"])
    valid_llm_original_df = comparison_df.dropna(subset=["llm_original_score"])
    valid_llm_qwen_df = comparison_df.dropna(subset=["llm_qwen_score"])
    valid_llm_deepseek_df = comparison_df.dropna(subset=["llm_deepseek_score"])
    valid_all_df = comparison_df.dropna(subset=["machine_score", "llm_original_score", "llm_qwen_score", "llm_deepseek_score"])

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
        
        # 按照公式(1)和(2)计算 abs 和 var 指标
        # 公式(1): abs = (1/N) * Σ|JS_MER - JS_M| （与平均绝对差值相同）
        print(f"        - abs (公式1): {mean_abs_diff:.3f}")
        
        # 公式(2): var = (1/N) * Σ|JS_MER - JS_M|²  
        var_formula = np.mean((diff) ** 2)
        print(f"        - var (公式2): {var_formula:.3f}")
        median_abs_diff = np.median(abs_diff)
        max_abs_diff = np.max(abs_diff)
        min_abs_diff = np.min(abs_diff)
        
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

    # 各种评价方法之间的相关性分析
    if len(valid_machine_df) > 1:
        calculate_correlations(
            valid_machine_df["human_score"], valid_machine_df["machine_score"],
            "人工评价", "机器评价", len(valid_machine_df)
        )

    if len(valid_llm_original_df) > 1:
        calculate_correlations(
            valid_llm_original_df["human_score"], valid_llm_original_df["llm_original_score"],
            "人工评价", "原始LLM评价", len(valid_llm_original_df)
        )

    if len(valid_llm_qwen_df) > 1:
        calculate_correlations(
            valid_llm_qwen_df["human_score"], valid_llm_qwen_df["llm_qwen_score"],
            "人工评价", "Qwen LLM评价", len(valid_llm_qwen_df)
        )

    if len(valid_llm_deepseek_df) > 1:
        calculate_correlations(
            valid_llm_deepseek_df["human_score"], valid_llm_deepseek_df["llm_deepseek_score"],
            "人工评价", "DeepSeek LLM评价", len(valid_llm_deepseek_df)
        )

    if len(valid_all_df) > 1:
        calculate_correlations(
            valid_all_df["machine_score"], valid_all_df["llm_original_score"],
            "机器评价", "原始LLM评价", len(valid_all_df)
        )

        calculate_correlations(
            valid_all_df["machine_score"], valid_all_df["llm_qwen_score"],
            "机器评价", "Qwen LLM评价", len(valid_all_df)
        )

        calculate_correlations(
            valid_all_df["llm_original_score"], valid_all_df["llm_qwen_score"],
            "原始LLM评价", "Qwen LLM评价", len(valid_all_df)
        )

        calculate_correlations(
            valid_all_df["machine_score"], valid_all_df["llm_deepseek_score"],
            "机器评价", "DeepSeek LLM评价", len(valid_all_df)
        )

        calculate_correlations(
            valid_all_df["llm_original_score"], valid_all_df["llm_deepseek_score"],
            "原始LLM评价", "DeepSeek LLM评价", len(valid_all_df)
        )

        calculate_correlations(
            valid_all_df["llm_qwen_score"], valid_all_df["llm_deepseek_score"],
            "Qwen LLM评价", "DeepSeek LLM评价", len(valid_all_df)
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

    print("\n原始LLM评价前5名:")
    valid_llm_original_for_top5 = comparison_df.dropna(subset=["llm_original_rank"])
    if len(valid_llm_original_for_top5) >= 5:
        top5_llm_original = valid_llm_original_for_top5.nsmallest(5, "llm_original_rank")
        for i, (_, row) in enumerate(top5_llm_original.iterrows(), 1):
            print(f"   {i}. {row['team_name']:<25} (分数: {row['llm_original_score']:.3f})")

    print("\nQwen LLM评价前5名:")
    valid_llm_qwen_for_top5 = comparison_df.dropna(subset=["llm_qwen_rank"])
    if len(valid_llm_qwen_for_top5) >= 5:
        top5_llm_qwen = valid_llm_qwen_for_top5.nsmallest(5, "llm_qwen_rank")
        for i, (_, row) in enumerate(top5_llm_qwen.iterrows(), 1):
            print(f"   {i}. {row['team_name']:<25} (分数: {row['llm_qwen_score']:.3f})")

    print("\nDeepSeek LLM评价前5名:")
    valid_llm_deepseek_for_top5 = comparison_df.dropna(subset=["llm_deepseek_rank"])
    if len(valid_llm_deepseek_for_top5) >= 5:
        top5_llm_deepseek = valid_llm_deepseek_for_top5.nsmallest(5, "llm_deepseek_rank")
        for i, (_, row) in enumerate(top5_llm_deepseek.iterrows(), 1):
            print(f"   {i}. {row['team_name']:<25} (分数: {row['llm_deepseek_score']:.3f})")

    # 显示数据完整性统计
    print(f"\n📊 数据完整性统计:")
    print("=" * 50)
    print(f"   - 总团队数 (人工评价): {len(comparison_df)}")
    print(f"   - 有机器评价的团队: {len(comparison_df.dropna(subset=['machine_rank']))}")
    print(f"   - 有原始LLM评价的团队: {len(comparison_df.dropna(subset=['llm_original_rank']))}")
    print(f"   - 有Qwen LLM评价的团队: {len(comparison_df.dropna(subset=['llm_qwen_rank']))}")
    print(f"   - 有DeepSeek LLM评价的团队: {len(comparison_df.dropna(subset=['llm_deepseek_rank']))}")
    print(f"   - 五种评价都有的团队: {len(comparison_df.dropna(subset=['machine_rank', 'llm_original_rank', 'llm_qwen_rank', 'llm_deepseek_rank']))}")

    return comparison_df


if __name__ == "__main__":
    compare_all_evaluations_zh_with_qwen()
