#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算chrF流畅度分数的脚本
使用chrF++.py来计算人类解毒句子和模型解毒句子之间的流畅度分数
"""

import pandas as pd
import os
import tempfile

# 导入chrF++模块
import importlib.util
spec = importlib.util.spec_from_file_location("chrF_plus_plus", "chrF-master/chrF++.py")
chrF_plus_plus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chrF_plus_plus)
computeChrF = chrF_plus_plus.computeChrF


def calculate_chrf_score(reference_sentence, hypothesis_sentence, nworder=2, ncorder=6, beta=2.0):
    """
    计算两个句子之间的chrF分数
    
    Args:
        reference_sentence (str): 参考句子（人类解毒的句子）
        hypothesis_sentence (str): 假设句子（模型解毒的句子）
        nworder (int): 词n-gram的阶数，默认为2
        ncorder (int): 字符n-gram的阶数，默认为6
        beta (float): beta参数，默认为2.0
    
    Returns:
        float: chrF分数（0-1之间）
    """
    try:
        # 创建临时文件来存储句子
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as ref_file:
            ref_file.write(reference_sentence.strip() + '\n')
            ref_filename = ref_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as hyp_file:
            hyp_file.write(hypothesis_sentence.strip() + '\n')
            hyp_filename = hyp_file.name
        
        # 使用computeChrF函数计算分数
        with open(ref_filename, 'r') as ref_fp, open(hyp_filename, 'r') as hyp_fp:
            totalF, _, _, _ = computeChrF(
                ref_fp, hyp_fp, nworder, ncorder, beta
            )
        
        # 清理临时文件
        os.unlink(ref_filename)
        os.unlink(hyp_filename)
        
        return totalF
        
    except Exception as e:
        print(f"计算chrF分数时出错: {e}")
        return None


def load_and_match_data():
    """
    加载并匹配两个数据文件中的句子
    
    Returns:
        tuple: (matched_data, en_tsv_df) 匹配的数据和EN.tsv数据框
    """
    # 读取数据文件
    print("正在读取数据文件...")
    en_tsv = pd.read_csv('chrF-master/EN.tsv', sep='\t')
    english_test = pd.read_csv('2024_with_answers/english_test.csv')
    
    print(f"EN.tsv: {len(en_tsv)} 行")
    print(f"english_test.csv: {len(english_test)} 行")
    
    # 创建匹配字典
    test_dict = {}
    for _, row in english_test.iterrows():
        toxic_key = row['toxic_sentence'].strip().lower()
        test_dict[toxic_key] = row['neutral_sentence']
    
    # 匹配句子
    matched_data = []
    for idx, row in en_tsv.iterrows():
        toxic_key = row['toxic_sentence'].strip().lower()
        if toxic_key in test_dict:
            matched_data.append({
                'index': idx,
                'toxic_sentence': row['toxic_sentence'],
                'model_neutral': row['neutral_sentence'],
                'human_neutral': test_dict[toxic_key]
            })
    
    print(f"匹配到 {len(matched_data)} 个句子对")
    return matched_data, en_tsv


def calculate_all_fluency_scores():
    """
    计算所有匹配句子的流畅度分数
    """
    print("开始计算流畅度分数...")
    
    # 加载和匹配数据
    matched_data, en_tsv = load_and_match_data()
    
    if not matched_data:
        print("没有找到匹配的句子，无法计算流畅度分数")
        return
    
    # 计算流畅度分数
    fluency_scores = {}
    
    for i, data in enumerate(matched_data):
        print(f"\n处理第 {i+1}/{len(matched_data)} 个句子对...")
        print(f"有毒句子: {data['toxic_sentence'][:50]}...")
        print(f"模型解毒: {data['model_neutral'][:50]}...")
        print(f"人类解毒: {data['human_neutral'][:50]}...")
        
        # 计算chrF分数（人类解毒句子作为参考，模型解毒句子作为假设）
        fluency_score = calculate_chrf_score(
            data['human_neutral'],  # 参考句子（人类解毒）
            data['model_neutral']   # 假设句子（模型解毒）
        )
        
        if fluency_score is not None:
            fluency_scores[data['index']] = fluency_score
            print(f"流畅度分数: {fluency_score:.4f}")
        else:
            print("计算失败")
    
    # 更新EN.tsv文件
    print(f"\n成功计算了 {len(fluency_scores)} 个流畅度分数")
    
    # 将流畅度分数添加到数据框中
    en_tsv['Fluency_score'] = None
    for idx, score in fluency_scores.items():
        en_tsv.loc[idx, 'Fluency_score'] = score
    
    # 保存更新后的文件
    output_file = 'chrF-master/EN_with_fluency.tsv'
    en_tsv.to_csv(output_file, sep='\t', index=False)
    print(f"\n结果已保存到: {output_file}")

    # 同时更新原始的EN.tsv文件
    original_file = 'chrF-master/EN.tsv'
    en_tsv.to_csv(original_file, sep='\t', index=False)
    print(f"原始文件已更新: {original_file}")
    
    # 显示统计信息
    calculated_scores = en_tsv['Fluency_score'].dropna()
    if len(calculated_scores) > 0:
        print(f"\n流畅度分数统计:")
        print(f"  平均值: {calculated_scores.mean():.4f}")
        print(f"  最小值: {calculated_scores.min():.4f}")
        print(f"  最大值: {calculated_scores.max():.4f}")
        print(f"  标准差: {calculated_scores.std():.4f}")


if __name__ == "__main__":
    calculate_all_fluency_scores()
