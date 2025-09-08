#!/usr/bin/env python3
"""
对比中文人工评价、LLM评价和机器评价的结果
包含多种一致性评价指标和评价质量分析
"""
import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def compare_all_evaluations_zh():
    """对比三种评价方法的中文结果"""

    # 人工评价数据（中文分数，来自“Test Phase: Manual Evaluation Final Results”表的 zh 列）
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

    # 机器评价数据（中文分数，来自“Test Phase: Automatic Evaluation Results”表的 zh 列）
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

    # 读取中文LLM评价结果
    llm_candidates = [
        "llm_evaluation_ranking_zero_shot.csv",
        os.path.join(
            "data",
            "ranking_results",
            "zh",
            "llm_evaluation_ranking_zero_shot.csv",
        ),
    ]
    llm_file = next((p for p in llm_candidates if os.path.exists(p)), None)
    if not llm_file:
        print(
            "❌ 未找到中文LLM评价结果文件。请确认是否已生成：\n  - data/ranking_results/zh/llm_evaluation_ranking_zero_shot.csv"
        )
        return

    llm_df = pd.read_csv(llm_file)

    # 应用LLM团队名称映射
    llm_df["team_name"] = (
        llm_df["team_name"].map(llm_name_mapping).fillna(llm_df["team_name"])
    )

    llm_df = llm_df.sort_values("j_score_mean", ascending=False).reset_index(drop=True)
    llm_df["llm_rank"] = llm_df.index + 1

    print("🔍 三种评价方法对比分析（中文数据，以人工评价为准）")
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

    # 保存结果
    output_dir = os.path.join("data", "evaluation_results", "llm_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_evaluations_comparison_zh.csv")
    comparison_df.to_csv(output_file, index=False)
    print(f"\n📁 中文对比结果已保存到: {output_file}")

    # 检查是否填入了实际分数
    has_human_scores = any(row["human_score"] > 0 for _, row in human_df.iterrows())
    has_machine_scores = any(row["machine_score"] > 0 for _, row in machine_df.iterrows())

    if not has_human_scores or not has_machine_scores:
        print(f"\n⚠️  注意：请在脚本中填入实际的人工评价和机器评价分数后重新运行")
        print(f"   - 人工评价分数：第21-36行的 human_score 字段")
        print(f"   - 机器评价分数：第39-54行的 machine_score 字段")
        return

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
    compare_all_evaluations_zh()
