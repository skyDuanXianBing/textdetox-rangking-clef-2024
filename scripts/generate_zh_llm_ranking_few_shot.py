#!/usr/bin/env python3
"""
生成中文LLM评价排名文件 (Few-Shot)
基于各队伍的中文LLM Few-Shot评价结果生成汇总排名
"""
import pandas as pd
import numpy as np
import os
import glob

def generate_zh_llm_ranking_few_shot():
    """生成中文LLM Few-Shot评价排名文件"""
    
    print("🚀 开始生成中文LLM Few-Shot评价排名")
    print("=" * 60)
    
    # 中文LLM结果目录
    llm_evolution_dir = "data/result/llm_evolution"
    
    if not os.path.exists(llm_evolution_dir):
        print(f"❌ 未找到LLM评价结果目录: {llm_evolution_dir}")
        return
    
    # 查找所有中文few-shot结果文件
    zh_pattern = os.path.join(llm_evolution_dir, "*_zh", "etd_few_shot_results_zh_ZHPrompt_qwen-plus-latest.csv")
    zh_files = glob.glob(zh_pattern)
    
    print(f"✅ 找到 {len(zh_files)} 个中文LLM Few-Shot评价结果文件")
    
    ranking_data = []
    
    for file_path in zh_files:
        try:
            # 从文件路径提取团队名称
            team_dir = os.path.basename(os.path.dirname(file_path))
            team_name = team_dir.replace("_zh", "")
            
            print(f"📊 处理团队: {team_name}")
            
            # 读取评价结果
            df = pd.read_csv(file_path)
            
            # 检查必要的列是否存在
            required_columns = ['STA', 'CS', 'FS', 'j_score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"⚠️  团队 {team_name} 缺少列: {missing_columns}")
                continue
            
            # 计算统计指标
            total_samples = len(df)
            
            # 计算各指标的均值和标准差
            sta_mean = df['STA'].mean()
            cs_mean = df['CS'].mean()
            fs_mean = df['FS'].mean()
            j_score_mean = df['j_score'].mean()
            
            sta_std = df['STA'].std()
            cs_std = df['CS'].std()
            fs_std = df['FS'].std()
            j_score_std = df['j_score'].std()
            
            # 计算完美分数和零分数的数量
            perfect_scores = len(df[df['j_score'] == 1.0])
            zero_scores = len(df[df['j_score'] == 0.0])
            
            # 构建相对文件路径
            relative_path = os.path.relpath(file_path, "data/result")
            
            ranking_data.append({
                'team_name': team_name,
                'evaluation_type': 'few_shot_qwen',
                'total_samples': total_samples,
                'sta_mean': sta_mean,
                'cs_mean': cs_mean,
                'fs_mean': fs_mean,
                'j_score_mean': j_score_mean,
                'sta_std': sta_std,
                'cs_std': cs_std,
                'fs_std': fs_std,
                'j_score_std': j_score_std,
                'perfect_scores': perfect_scores,
                'zero_scores': zero_scores,
                'file_path': relative_path
            })
            
        except Exception as e:
            print(f"❌ 处理文件 {file_path} 时出错: {str(e)}")
            continue
    
    if not ranking_data:
        print("❌ 没有找到有效的评价结果文件")
        return
    
    # 创建DataFrame并按j_score_mean降序排序
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values('j_score_mean', ascending=False).reset_index(drop=True)
    
    # 确保输出目录存在
    output_dir = os.path.join("data", "ranking_results", "zh")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存排名结果
    output_file = os.path.join(output_dir, "llm_evaluation_ranking_few_shot_qwen.csv")
    ranking_df.to_csv(output_file, index=False)
    
    print(f"\n📁 中文LLM Few-Shot评价排名已保存到: {output_file}")
    print("\n🏆 中文LLM Few-Shot评价排名 (前10名):")
    print("=" * 100)
    print(f"{'排名':<4} {'团队名称':<25} {'联合分数':<10} {'STA':<8} {'CS':<8} {'FS':<8} {'样本数':<8}")
    print("-" * 100)
    
    for idx, row in ranking_df.head(10).iterrows():
        rank = idx + 1
        print(f"{rank:<4} {row['team_name']:<25} {row['j_score_mean']:<10.4f} "
              f"{row['sta_mean']:<8.3f} {row['cs_mean']:<8.3f} {row['fs_mean']:<8.3f} "
              f"{row['total_samples']:<8}")
    
    print(f"\n📊 统计摘要:")
    print(f"   - 参与团队数: {len(ranking_df)}")
    print(f"   - 平均联合分数: {ranking_df['j_score_mean'].mean():.4f}")
    print(f"   - 最高联合分数: {ranking_df['j_score_mean'].max():.4f}")
    print(f"   - 最低联合分数: {ranking_df['j_score_mean'].min():.4f}")
    
    return ranking_df

if __name__ == "__main__":
    generate_zh_llm_ranking_few_shot()
