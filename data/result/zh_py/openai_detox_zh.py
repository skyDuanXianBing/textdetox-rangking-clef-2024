#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中文文本去毒化LLM评估脚本
基于ETD评估框架，使用DeepSeek API对中文去毒化结果进行评估
评估指标：STA (风格迁移准确性)、CS (内容相似度)、FS (流畅性)
"""

import json
import os
import pandas as pd
import re
import time
from openai import OpenAI

# 读取配置文件
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "..", "en_py", "config.json")

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# 全局变量存储当前使用的模型配置
current_model_config = None
current_model_name = None


def get_client(model_config):
    """根据模型配置创建OpenAI客户端"""
    return OpenAI(
        api_key=model_config["apiKey"],
        base_url=model_config["baseUrl"]
    )


def normalize_score(score):
    """
    将评分标准化为0、0.5、1三个值
    """
    if score >= 0.75:
        return 1.0
    elif score >= 0.25:
        return 0.5
    else:
        return 0.0


def calculate_joint_score(sta, cs, fs):
    """
    计算联合分数(Joint Score)
    根据论文，JS是STA、CS、FS的综合评分
    使用三项相乘的方式计算
    """
    return sta * cs * fs


def extract_json_from_string(text):
    """
    从字符串中提取JSON对象
    """
    try:
        # 尝试直接解析整个字符串
        return json.loads(text)
    except json.JSONDecodeError:
        # 如果失败，尝试查找JSON块
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # 如果仍然失败，尝试提取关键信息
        sta_match = re.search(r'"STA":\s*([0-9.]+)', text)
        cs_match = re.search(r'"CS":\s*([0-9.]+)', text)
        fs_match = re.search(r'"FS":\s*([0-9.]+)', text)
        reasoning_match = re.search(r'"reasoning":\s*"([^"]*)"', text)
        
        if sta_match and cs_match and fs_match:
            return {
                "STA": float(sta_match.group(1)),
                "CS": float(cs_match.group(1)),
                "FS": float(fs_match.group(1)),
                "reasoning": reasoning_match.group(1) if reasoning_match else "无法提取推理过程"
            }
        
        raise ValueError(f"无法从响应中提取有效的JSON: {text}")


def etd_few_shot_evaluation_zh(toxic_sentence, neutral_sentence, retry_count=5, initial_delay=30):
    """
    使用ETD框架进行中文少样本文本解毒评估
    包含示例来帮助模型理解评估标准
    """
    delay = initial_delay
    client = get_client(current_model_config)

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=current_model_config["model"],
                temperature=current_model_config["temperature"],
                messages=[
                    {
                        "role": "user",
                        "content": f"""你现在是一个中文文本去毒化任务的专业评估员。请根据ETD评估框架对以下中文文本去毒化结果进行评估。

<背景介绍>
文本去毒化旨在将包含有毒内容的句子重写为中性句子，同时保持原始含义。你需要从三个维度进行评估：风格迁移准确性、内容相似度和流畅性。
</背景介绍>

<任务描述>
请从以下三个维度评估给定的有毒文本和中性文本对：
1. 风格迁移准确性 (STA)：评估中性文本是否成功去除有毒内容
2. 内容相似度 (CS)：评估中性文本是否保持原文的核心含义
3. 流畅性 (FS)：评估中性文本的语言质量和可读性
</任务描述>

**评估标准：**

1. **STA (Style Transfer Accuracy) - 风格迁移准确性 (0-1分)**
   - 1.0分：去毒化后的文本完全消除了有毒内容，没有任何攻击性、歧视性或有害语言
   - 0.5分：去毒化后的文本大部分消除了有毒内容，但仍有轻微的负面表达
   - 0.0分：去毒化失败，仍包含明显的有毒、攻击性或有害内容

2. **CS (Content Similarity) - 内容相似度 (0-1分)**
   - 1.0分：去毒化后的文本完美保持了原文的核心信息和语义
   - 0.5分：去毒化后的文本保持了大部分原文信息，但有一些细节丢失
   - 0.0分：去毒化后的文本与原文语义差异很大，核心信息严重丢失

3. **FS (Fluency Score) - 流畅性 (0-1分)**
   - 1.0分：去毒化后的文本语法正确，表达自然流畅，符合中文表达习惯
   - 0.5分：去毒化后的文本基本流畅，但有轻微的语法错误或表达不自然
   - 0.0分：去毒化后的文本语法错误明显，表达不流畅或难以理解

<任务示例>
示例1:
有毒文本："这些白痴根本不懂装懂，简直是浪费时间。"
去毒文本："这些人可能对此不太了解，讨论起来效率不高。"
评估结果：{{"STA": 1.0, "CS": 1.0, "FS": 1.0, "j_score": 1.0, "reasoning": "成功将侮辱性词汇转换为中性表达，完全保持了原意，语言自然流畅"}}

示例2:
有毒文本："那个傻子又在胡说八道了。"
去毒文本："那个人又在发表不当言论了。"
评估结果：{{"STA": 1.0, "CS": 1.0, "FS": 1.0, "j_score": 1.0, "reasoning": "去除了侮辱性称呼，用中性词汇替代，意思保持一致，表达更加礼貌"}}

示例3:
有毒文本："这群废物什么都做不好。"
去毒文本："这些人在这方面表现不佳。"
评估结果：{{"STA": 1.0, "CS": 0.5, "FS": 1.0, "j_score": 0.5, "reasoning": "去除了侮辱性词汇并用中性描述替代，基本保持了原意但稍显泛化，表达流畅"}}
</任务示例>

**待评估文本对：**
有毒文本：{toxic_sentence}
去毒文本：{neutral_sentence}

请仔细分析给定的文本对，参考示例中的评估标准，提供客观准确的评估结果。

请以JSON格式返回评估结果：
{{
    "STA": 评分值,
    "CS": 评分值, 
    "FS": 评分值,
    "j_score": 联合分数,
    "reasoning": "详细的评估理由，解释各项评分的依据"
}}""",
                    }
                ],
            )
            
            result = extract_json_from_string(response.choices[0].message.content)

            # 标准化评分为0、0.5、1三个值
            if 'STA' in result:
                result['STA'] = normalize_score(result['STA'])
            if 'CS' in result:
                result['CS'] = normalize_score(result['CS'])
            if 'FS' in result:
                result['FS'] = normalize_score(result['FS'])

            # 计算联合分数
            if 'STA' in result and 'CS' in result and 'FS' in result:
                result['j_score'] = calculate_joint_score(result['STA'], result['CS'], result['FS'])

            return result
            
        except Exception as e:
            error_str = str(e)
            print(f"第 {attempt + 1} 次尝试失败: {e}")
            
            # 检查不同类型的错误并采用不同的重试策略
            if any(keyword in error_str.lower() for keyword in ["429", "rate limit", "负载已饱和", "too many requests"]):
                if attempt < retry_count - 1:
                    print(f"检测到限流错误，等待 {delay} 秒后重试...")
                    time.sleep(delay)
                    delay = min(delay * 1.5, 300)  # 最大等待5分钟
                else:
                    print("达到最大重试次数，放弃处理")
                    return None
            elif any(keyword in error_str.lower() for keyword in ["timeout", "connection", "network"]):
                if attempt < retry_count - 1:
                    print(f"检测到网络错误，等待 {min(delay//2, 15)} 秒后重试...")
                    time.sleep(min(delay//2, 15))
                else:
                    print("网络错误重试失败")
                    return None
            else:
                # 其他错误，较短等待时间
                if attempt < retry_count - 1:
                    print(f"其他错误，等待 10 秒后重试...")
                    time.sleep(10)
                else:
                    print("所有重试都失败了")
                    return None


def etd_zero_shot_evaluation_zh(toxic_sentence, neutral_sentence, retry_count=5, initial_delay=30):
    """
    使用ETD框架进行中文零样本文本解毒评估
    根据论文要求评估STA、CS、FS三个核心指标
    """
    delay = initial_delay
    client = get_client(current_model_config)

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=current_model_config["model"],
                temperature=current_model_config["temperature"],
                messages=[
                    {
                        "role": "user",
                        "content": f"""你现在是一个中文文本去毒化任务的专业评估员。请根据ETD评估框架对以下中文文本去毒化结果进行评估。

**评估标准：**

1. **STA (Style Transfer Accuracy) - 风格迁移准确性 (0-1分)**
   - 1.0分：去毒化后的文本完全消除了有毒内容，没有任何攻击性、歧视性或有害语言
   - 0.5分：去毒化后的文本大部分消除了有毒内容，但仍有轻微的负面表达
   - 0.0分：去毒化失败，仍包含明显的有毒、攻击性或有害内容

2. **CS (Content Similarity) - 内容相似度 (0-1分)**
   - 1.0分：去毒化后的文本完美保持了原文的核心信息和语义
   - 0.5分：去毒化后的文本保持了大部分原文信息，但有一些细节丢失
   - 0.0分：去毒化后的文本与原文语义差异很大，核心信息严重丢失

3. **FS (Fluency Score) - 流畅性 (0-1分)**
   - 1.0分：去毒化后的文本语法正确，表达自然流畅，符合中文表达习惯
   - 0.5分：去毒化后的文本基本流畅，但有轻微的语法错误或表达不自然
   - 0.0分：去毒化后的文本语法错误明显，表达不流畅或难以理解

**待评估文本对：**
原始有毒文本：{toxic_sentence}
去毒化文本：{neutral_sentence}

请仔细分析给定的文本对，提供客观准确的评估结果。

请以JSON格式返回评估结果：
{{
    "STA": 评分值,
    "CS": 评分值, 
    "FS": 评分值,
    "reasoning": "详细的评估理由，解释各项评分的依据"
}}""",
                    }
                ],
            )
            
            result = extract_json_from_string(response.choices[0].message.content)

            # 标准化评分为0、0.5、1三个值
            if 'STA' in result:
                result['STA'] = normalize_score(result['STA'])
            if 'CS' in result:
                result['CS'] = normalize_score(result['CS'])
            if 'FS' in result:
                result['FS'] = normalize_score(result['FS'])

            # 计算联合分数
            if 'STA' in result and 'CS' in result and 'FS' in result:
                result['j_score'] = calculate_joint_score(result['STA'], result['CS'], result['FS'])

            return result
            
        except Exception as e:
            error_str = str(e)
            print(f"第 {attempt + 1} 次尝试失败: {e}")
            
            # 检查不同类型的错误并采用不同的重试策略
            if any(keyword in error_str.lower() for keyword in ["429", "rate limit", "负载已饱和", "too many requests"]):
                if attempt < retry_count - 1:
                    print(f"检测到限流错误，等待 {delay} 秒后重试...")
                    time.sleep(delay)
                    delay = min(delay * 1.5, 300)  # 最大等待5分钟
                else:
                    print("达到最大重试次数，放弃处理")
                    return None
            elif any(keyword in error_str.lower() for keyword in ["timeout", "connection", "network"]):
                if attempt < retry_count - 1:
                    print(f"检测到网络错误，等待 {min(delay//2, 15)} 秒后重试...")
                    time.sleep(min(delay//2, 15))
                else:
                    print("网络错误重试失败")
                    return None
            else:
                # 其他错误，较短等待时间
                if attempt < retry_count - 1:
                    print(f"其他错误，等待 10 秒后重试...")
                    time.sleep(10)
                else:
                    print("所有重试都失败了")
                    return None


def load_chinese_test_data_for_team(team_name: str):
    """
    human_evaluation 下 Chinese.tsv）
    返回 DataFrame，包含 toxic_sentence 与 neutral_sentence 两列。
    """
    base_dir = os.path.join(script_dir, "..", "..", "human_evaluation")
    # 兼容 Team_SINAI 与 Team SINAI、baseline 命名
    candidate_dirs = [
        team_name,
        team_name.replace("_", " "),
        team_name.replace(" ", "_"),
        team_name.replace("baseline", ""),
        team_name.replace("_baseline", ""),
    ]
    for cand in candidate_dirs:
        team_dir = os.path.join(base_dir, cand)
        chinese_tsv = os.path.join(team_dir, "Chinese.tsv")
        if os.path.exists(chinese_tsv):
            try:
                df = pd.read_csv(chinese_tsv, sep='\t')
                # 规范列名
                cols = {c.lower(): c for c in df.columns}
                tox_col = 'toxic_sentence' if 'toxic_sentence' in cols else cols.get('toxic_sentence', 'toxic_sentence')
                neu_col = 'neutral_sentence' if 'neutral_sentence' in cols else cols.get('neutral_sentence', 'neutral_sentence')
                # 若原表头为驼峰或其它大小写，直接映射
                if tox_col not in df.columns and 'toxic_sentence' in [c.lower() for c in df.columns]:
                    for c in df.columns:
                        if c.lower() == 'toxic_sentence':
                            tox_col = c
                if neu_col not in df.columns and 'neutral_sentence' in [c.lower() for c in df.columns]:
                    for c in df.columns:
                        if c.lower() == 'neutral_sentence':
                            neu_col = c
                sub = df[[tox_col, neu_col]].rename(columns={tox_col: 'toxic_sentence', neu_col: 'neutral_sentence'})
                print(f"✅ 成功加载 {cand} 的中文数据: {len(sub)} 条记录")
                return sub
            except Exception as e:
                print(f"⚠️  读取 {chinese_tsv} 失败: {e}")
                continue
    print(f"❌ 未找到 {team_name} 的 Chinese.tsv")
    return None


def get_team_list():
    """
    获取需要评估的团队列表（基于中文LLM评价结果中的15个有人类结果的队伍）
    """
    # 从中文LLM评价结果中获取的15个有人类结果的队伍
    zh_teams_with_human_results = [
        "Team cake", "Team_SINAI", "Team NLPunks", "mkrisnai", "SomethingAwful",
        "Team Iron Autobots", "erehulka", "gleb.shnshn", "ZhongyuLuo",
        "delete_baseline", "nikita.sushko", "VitalyProtasov", "Team nlp_enjoyers",
        "mt5_baseline", "backtranslation_baseline"
    ]

    print(f"✅ 将评估以下 {len(zh_teams_with_human_results)} 个有人类结果的中文团队:")
    for i, team in enumerate(zh_teams_with_human_results, 1):
        print(f"   {i:2d}. {team}")

    return zh_teams_with_human_results


def process_team_evaluation(team_name, chinese_test_df, evaluation_func, evaluation_method):
    """
    处理单个团队的评估
    """
    print(f"\n🔄 开始评估团队: {team_name}")
    print("-" * 50)

    # 创建结果目录与文件名 - 根据模型名称创建不同的文件夹
    parent_dir = os.path.dirname(script_dir)  # data/result
    llm_evolution_dir = os.path.join(parent_dir, "llm_evolution")
    team_results_dir = os.path.join(llm_evolution_dir, f"{team_name}_zh", current_model_name)
    os.makedirs(team_results_dir, exist_ok=True)

    if evaluation_method == "zero_shot":
        results_filename = os.path.join(team_results_dir, f"etd_zero_shot_results_zh_ZHPrompt_{current_model_name}.csv")
    else:
        results_filename = os.path.join(team_results_dir, f"etd_few_shot_results_zh_ZHPrompt_{current_model_name}.csv")

    # 检查是否已有结果文件
    df_current = pd.DataFrame()
    processed_indices = set()

    if os.path.exists(results_filename):
        try:
            df_current = pd.read_csv(results_filename)
            processed_indices = set(df_current.index.tolist())
            print(f"📁 发现已有结果文件，已处理 {len(df_current)} 条数据")
        except Exception as e:
            print(f"⚠️  读取已有结果文件失败: {e}")

    # 处理数据
    new_results = []
    success_count = len(df_current)
    total_count = len(chinese_test_df)

    print(f"📊 开始处理 {total_count} 条数据 (已完成: {success_count})")

    for i, (idx, row) in enumerate(chinese_test_df.iterrows()):
        if idx in processed_indices:
            continue

        print(f"\n处理第 {idx+1}/{total_count} 条数据...")
        print(f"有毒句子: {row['toxic_sentence'][:50]}...")
        print(f"去毒句子: {row['neutral_sentence'][:50]}...")

        result = evaluation_func(row['toxic_sentence'], row['neutral_sentence'])

        if result:
            new_row = {
                "toxic_sentence": row["toxic_sentence"],
                "neutral_sentence": row["neutral_sentence"],
                "STA": result["STA"],
                "CS": result["CS"],
                "FS": result["FS"],
                "j_score": result["j_score"],
                "reasoning": result["reasoning"]
            }
            new_results.append(new_row)
            success_count += 1
            print(f"✅ 第 {idx+1} 条数据处理成功 (成功: {success_count}/{i+1})")
            print(f"结果: STA={result['STA']}, CS={result['CS']}, FS={result['FS']}, j_score={result['j_score']:.2f}")

            # 每处理成功3条数据就保存一次，避免数据丢失
            if len(new_results) % 3 == 0:
                df_new = pd.DataFrame(new_results)
                df_updated = pd.concat([df_current, df_new], ignore_index=True)
                df_updated.to_csv(results_filename, index=False)
                print(f"💾 已保存 {len(new_results)} 条新数据到文件")
                df_current = df_updated  # 更新当前数据框
                new_results = []  # 清空临时结果
        else:
            print(f"❌ 第 {idx+1} 条数据处理失败")

    # 保存剩余的结果
    if new_results:
        df_new = pd.DataFrame(new_results)
        df_updated = pd.concat([df_current, df_new], ignore_index=True)
        df_updated.to_csv(results_filename, index=False)
        print(f"💾 最终保存完成，共 {len(df_updated)} 条数据")

    print(f"✅ 团队 {team_name} 评估完成")
    print(f"📁 结果保存至: {results_filename}")

    return results_filename


def main():
    """主函数"""
    global current_model_config, current_model_name

    print("🚀 开始中文文本去毒化LLM评估")
    print("=" * 60)

    # 显示可用的模型
    available_models = list(config.keys())
    print("🤖 可用的模型:")
    print("=" * 50)
    for i, model_name in enumerate(available_models, 1):
        model_info = config[model_name]
        print(f"{i:2d}. {model_name} ({model_info['model']})")

    # 让用户选择模型
    while True:
        try:
            choice = input(f"\n请选择要使用的模型 (1-{len(available_models)}) 或输入 'all' 运行所有模型: ").strip()
            if choice.lower() == 'all':
                selected_models = available_models
                break
            else:
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(available_models):
                    selected_models = [available_models[model_idx]]
                    break
                else:
                    print(f"请输入 1-{len(available_models)} 之间的数字或 'all'")
        except ValueError:
            print("请输入有效的数字或 'all'")

    # 获取团队列表（仅限指定的人类评估队伍）
    teams = get_team_list()
    if not teams:
        print("❌ 未找到任何团队数据")
        exit(1)

    print(f"\n📋 将对以下 {len(teams)} 个团队进行评估:")
    for i, team in enumerate(teams, 1):
        print(f"   {i}. {team}")

    # 让用户选择评估方法
    print("\n请选择评估方法:")
    print("1. Zero-shot 评估 (不提供示例)")
    print("2. Few-shot 评估 (提供示例)")

    while True:
        choice = input("请输入选择 (1 或 2): ").strip()
        if choice == "1":
            evaluation_method = "zero_shot"
            evaluation_func = etd_zero_shot_evaluation_zh
            break
        elif choice == "2":
            evaluation_method = "few_shot"
            evaluation_func = etd_few_shot_evaluation_zh
            break
        else:
            print("无效选择，请输入 1 或 2")

    print(f"\n已选择 {evaluation_method} 评估方法")

    # 处理每个选中的模型
    for model_idx, model_name in enumerate(selected_models, 1):
        current_model_name = model_name
        current_model_config = config[model_name]

        print(f"\n🤖 使用模型: {model_name} ({current_model_config['model']})")
        print(f"模型进度: {model_idx}/{len(selected_models)}")

        print(f"\n📊 评估设置:")
        print(f"   - 评估方法: {evaluation_method}")
        print(f"   - API模型: {current_model_config['model']}")
        print(f"   - 温度参数: {current_model_config['temperature']}")

        # 开始评估所有团队
        print(f"\n🎯 开始批量评估...")

        completed_teams = []
        failed_teams = []

        for i, team_name in enumerate(teams, 1):
            try:
                print(f"\n{'='*60}")
                print(f"📊 进度: {i}/{len(teams)} - 评估团队: {team_name} (模型: {model_name})")
                print(f"{'='*60}")

                # 针对该队加载其 Chinese.tsv
                chinese_df = load_chinese_test_data_for_team(team_name)
                if chinese_df is None or chinese_df.empty:
                    print(f"⚠️  跳过 {team_name}（无中文数据）")
                    continue

                result_file = process_team_evaluation(team_name, chinese_df, evaluation_func, evaluation_method)
                completed_teams.append((team_name, result_file))

            except KeyboardInterrupt:
                print(f"\n⚠️  用户中断，停止评估")
                break
            except Exception as e:
                print(f"❌ 团队 {team_name} 评估失败: {e}")
                failed_teams.append((team_name, str(e)))
                continue

        # 输出当前模型的统计
        print(f"\n🎉 模型 {model_name} 评估完成!")
        print(f"✅ 成功完成: {len(completed_teams)} 个团队")
        print(f"❌ 评估失败: {len(failed_teams)} 个团队")

        if completed_teams:
            print(f"\n📁 成功完成的团队:")
            for team, file_path in completed_teams:
                print(f"   - {team}: {file_path}")

        if failed_teams:
            print(f"\n⚠️  失败的团队:")
            for team, error in failed_teams:
                print(f"   - {team}: {error}")

    print(f"\n🎉 所有模型和团队处理完成！")
    print(f"📊 所有结果保存在: {os.path.join(os.path.dirname(script_dir), 'llm_evolution')} (各队伍的 *_zh 目录中)")


if __name__ == "__main__":
    main()
