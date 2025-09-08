#!/usr/bin/env python3
"""
生成英文LLM评价排名文件 (Few-Shot)
基于各队伍的英文LLM Few-Shot评价结果生成汇总排名
"""
import pandas as pd
import numpy as np
import os
import glob

def generate_en_llm_ranking_few_shot():
    """生成英文LLM Few-Shot评价排名文件"""
    
    print("🚀 开始生成英文LLM Few-Shot评价排名")
    print("=" * 60)
    
    # 英文LLM结果目录
    llm_evolution_dir = "data/result/llm_evolution"
    
    if not os.path.exists(llm_evolution_dir):
        print(f"❌ 未找到LLM评价结果目录: {llm_evolution_dir}")
        return
    
    # 查找所有英文few-shot结果文件
    en_pattern = os.path.join(llm_evolution_dir, "*_en", "etd_few_shot_results_en_ENPrompt.csv")
    en_files = glob.glob(en_pattern)
    
    print(f"✅ 找到 {len(en_files)} 个英文LLM Few-Shot评价结果文件")
    
    ranking_data = []
    
    for file_path in en_files:
        try:
            # 从路径中提取队伍名称
            team_dir = os.path.dirname(file_path)
            team_name = os.path.basename(team_dir).replace("_en", "")
            
            # 读取评价结果
            df = pd.read_csv(file_path)
            
            if df.empty:
                print(f"⚠️  {team_name}: 结果文件为空")
                continue
            
            # 计算统计指标
            total_samples = len(df)
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
            
            ranking_data.append({
                'team_name': team_name,
                'evaluation_type': 'few_shot',
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
                'file_path': f"LLMEvolution/{team_name}_en/etd_few_shot_results_en_ENPrompt.csv"
            })
            
            print(f"✅ {team_name}: {total_samples} 条数据, J-Score均值: {j_score_mean:.3f}")
            
        except Exception as e:
            print(f"❌ 处理 {file_path} 失败: {e}")
            continue
    
    if not ranking_data:
        print("❌ 未找到有效的评价数据")
        return
    
    # 创建DataFrame并按J-Score排序
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values('j_score_mean', ascending=False).reset_index(drop=True)
    
    # 创建输出目录
    output_dir = os.path.join("data", "ranking_results", "en")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存排名文件
    output_file = os.path.join(output_dir, "llm_evaluation_ranking_few_shot.csv")
    ranking_df.to_csv(output_file, index=False)
    
    print(f"\n📊 Few-Shot排名结果 (按J-Score降序):")
    print("-" * 80)
    print(f"{'排名':<4} {'队伍名称':<25} {'J-Score':<8} {'样本数':<6} {'完美分':<6} {'零分':<6}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(ranking_df.iterrows(), 1):
        print(f"{i:<4} {row['team_name']:<25} {row['j_score_mean']:<8.3f} {row['total_samples']:<6} {row['perfect_scores']:<6} {row['zero_scores']:<6}")
    
    print(f"\n📁 Few-Shot排名文件已保存到: {output_file}")
    print(f"✅ 共处理 {len(ranking_df)} 个队伍的英文LLM Few-Shot评价结果")

if __name__ == "__main__":
    generate_en_llm_ranking_few_shot()