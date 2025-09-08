from openai import OpenAI
import re
import json
import pandas as pd
import os
import time

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取配置文件
config_path = os.path.join(script_dir, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# 全局变量存储当前使用的模型配置
current_model_config = None
current_model_name = None


def extract_json_from_string(input_string):
    """从字符串中提取JSON内容"""
    # 使用正则表达式匹配 JSON 部分
    json_match = re.search(r"\{.*\}", input_string, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        # 将 JSON 字符串中的双引号替换为单引号
        json_str = json_str.replace('""', '"')
        # 将 JSON 字符串转换为字典
        json_dict = json.loads(json_str)
        return json_dict
    else:
        print(f"无效的json字段{input_string}")
        raise ValueError("未找到有效的 JSON 内容")


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


def etd_few_shot_evaluation(toxic_sentence, neutral_sentence, retry_count=5, initial_delay=30):
    """
    使用ETD框架进行少样本文本解毒评估
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
                        "content": f"""You are now an expert evaluator for text detoxification tasks. Please evaluate the following text detoxification results according to the ETD evaluation framework.

<Background>
Text detoxification aims to rewrite sentences containing toxic content into neutral sentences while preserving the original meaning. You need to evaluate from three dimensions: Style Transfer Accuracy, Content Similarity, and Fluency.
</Background>

<Task Description>
Please evaluate the given toxic text and neutral text pair from the following three dimensions:
1. Style Transfer Accuracy (STA): Evaluate whether the neutral text successfully removes toxic content
2. Content Similarity (CS): Evaluate whether the neutral text maintains the core meaning of the original text
3. Fluency (FS): Evaluate the language quality and readability of the neutral text
</Task Description>

<Evaluation Criteria>
1. Style Transfer Accuracy (STA): Only 0, 0.5, 1 scores allowed
   - 1.0: Completely removes toxic content, language is completely neutral
   - 0.5: Partially removes toxic content, still has some toxic expressions
   - 0.0: Almost no toxic content removed

2. Content Similarity (CS): Only 0, 0.5, 1 scores allowed
   - 1.0: Completely preserves original meaning, no loss of core information
   - 0.5: Partially preserves original meaning, some information loss
   - 0.0: Almost no preservation of original meaning

3. Fluency (FS): Only 0, 0.5, 1 scores allowed
   - 1.0: Language is completely fluent, no grammatical errors
   - 0.5: Understandable but has some grammatical or expression issues
   - 0.0: Difficult to understand, serious grammatical errors
</Evaluation Criteria>

<Task Examples>
Example 1:
Toxic text: "any of us who did half this shit would be locked the fuck up already."
Neutral text: "any of us who did half of these actions would likely be incarcerated already."
Evaluation result: {{"STA": 1.0, "CS": 1.0, "FS": 1.0, "j_score": 1.0, "reasoning": "Successfully converted vulgar language to formal expression, completely preserved original meaning, language is fluent and natural"}}

Example 2:
Toxic text: "are they liars or just clueless?"
Neutral text: "are they dishonest or simply unaware?"
Evaluation result: {{"STA": 1.0, "CS": 1.0, "FS": 1.0, "j_score": 1.0, "reasoning": "Replaced aggressive vocabulary with neutral words, meaning remains consistent, expression is more polite"}}

Example 3:
Toxic text: "cretins like this rarely care about facts."
Neutral text: "people with such behavior rarely care about facts."
Evaluation result: {{"STA": 1.0, "CS": 0.5, "FS": 1.0, "j_score": 0.83, "reasoning": "Removed insulting vocabulary and replaced with neutral description, basically preserved original meaning but slightly generalized"}}
</Task Examples>

<Input Data Format>
Toxic text: {toxic_sentence}
Neutral text: {neutral_sentence}
</Input Data Format>

<Output Data Format>
Please strictly output the evaluation results in the following JSON format:

```json
{{
"STA": number, // Style Transfer Accuracy score (can only be 0, 0.5, or 1)
"CS": number,  // Content Similarity score (can only be 0, 0.5, or 1)
"FS": number,  // Fluency score (can only be 0, 0.5, or 1)
"j_score": number,  // Joint score (automatically calculated average)
"reasoning": "string" // Detailed evaluation reasoning, explaining the basis for each score
}}
```
</Output Data Format>

Please carefully analyze the given text pair, refer to the evaluation standards in the examples, and provide objective and accurate evaluation results.""",
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


def etd_zero_shot_evaluation(toxic_sentence, neutral_sentence, retry_count=5, initial_delay=30):
    """
    使用ETD框架进行零样本文本解毒评估
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
                        "content": f"""You are now an expert evaluator for text detoxification tasks. Please evaluate the following text detoxification results according to the ETD evaluation framework.

<Background>
Text detoxification aims to rewrite sentences containing toxic content into neutral sentences while preserving the original meaning. You need to evaluate from three dimensions: Style Transfer Accuracy, Content Similarity, and Fluency.
</Background>

<Task Description>
Please evaluate the given toxic text and neutral text pair from the following three dimensions:
1. Style Transfer Accuracy (STA): Evaluate whether the neutral text successfully removes toxic content
2. Content Similarity (CS): Evaluate whether the neutral text maintains the core meaning of the original text
3. Fluency (FS): Evaluate the language quality and readability of the neutral text
</Task Description>

<Evaluation Criteria>
1. Style Transfer Accuracy (STA): Only 0, 0.5, 1 scores allowed
   - 1.0: Completely removes toxic content, language is completely neutral
   - 0.5: Partially removes toxic content, still has some toxic expressions
   - 0.0: Almost no toxic content removed

2. Content Similarity (CS): Only 0, 0.5, 1 scores allowed
   - 1.0: Completely preserves original meaning, no loss of core information
   - 0.5: Partially preserves original meaning, some information loss
   - 0.0: Almost no preservation of original meaning

3. Fluency (FS): Only 0, 0.5, 1 scores allowed
   - 1.0: Language is completely fluent, no grammatical errors
   - 0.5: Understandable but has some grammatical or expression issues
   - 0.0: Difficult to understand, serious grammatical errors
</Evaluation Criteria>

<Input Data Format>
Toxic text: {toxic_sentence}
Neutral text: {neutral_sentence}
</Input Data Format>

<Output Data Format>
Please strictly output the evaluation results in the following JSON format:

```json
{{
"STA": number, // Style Transfer Accuracy score (can only be 0, 0.5, or 1)
"CS": number,  // Content Similarity score (can only be 0, 0.5, or 1)
"FS": number,  // Fluency score (can only be 0, 0.5, or 1)
"j_score": number,  // Joint score (automatically calculated average)
"reasoning": "string" // Detailed evaluation reasoning, explaining the basis for each score
}}
```
</Output Data Format>

Please carefully analyze the given text pair and provide objective and accurate evaluation results.""",
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


def find_missing_data(df_original, df_current):
    """找出未处理的数据索引"""
    # 获取已处理的句子对
    processed_pairs = set()
    for _, row in df_current.iterrows():
        processed_pairs.add((row['toxic_sentence'], row['neutral_sentence']))
    
    # 找出未处理的数据索引
    missing_indices = []
    for idx, row in df_original.iterrows():
        pair = (row['toxic_sentence'], row['neutral_sentence'])
        if pair not in processed_pairs:
            missing_indices.append(idx)
    
    return missing_indices


def get_matched_teams():
    """获取与榜单匹配的团队列表"""
    # 榜单中的团队名称
    leaderboard_teams = [
        "Team MarSanAI", "Team nlp_enjoyers", "Team cake", "mT5", "gangopsa",
        "Team SINAI", "delete", "Team Iron Autobots", "LanaKlitotekhnis", 
        "Anastasia1706", "ZhongyuLuo", "cocount", "backtranslation", "etomoscow",
        "cointegrated", "dkenco", "FD", "duplicate", "Team SmurfCat", "Imeribal",
        "nikita.sushko", "VitalyProtasov", "erehulka", "SomethingAwful", 
        "mareksuppa", "kofeinix", "Yekaterina29", "AlekseevArtem", "Team NLPunks",
        "pavelshtykov", "gleb.shnshn", "Volodimirich", "ansafronov", "MOOsipenko", "mkrisnai"
    ]
    
    # 去除重复
    leaderboard_teams = list(set(leaderboard_teams))
    
    # 获取所有可用的TSV文件
    all_tsv_files = [f for f in os.listdir(script_dir) if f.endswith('_english.tsv')]
    available_teams = [f.replace('_english.tsv', '') for f in all_tsv_files]
    
    # 创建匹配映射
    name_mappings = {
        "Team MarSanAI": "Team MarSan_AI",
        "Team SINAI": "Team_SINAI", 
        "mT5": "mt5_baseline",
        "delete": "delete_baseline",
        "backtranslation": "backtranslation_baseline",
    }
    
    # 找出匹配的团队
    matched_teams = []
    for lb_team in leaderboard_teams:
        # 直接匹配
        if lb_team in available_teams:
            matched_teams.append(lb_team)
        # 通过映射匹配
        elif lb_team in name_mappings and name_mappings[lb_team] in available_teams:
            matched_teams.append(name_mappings[lb_team])
        # 模糊匹配
        else:
            lb_normalized = lb_team.lower().replace(" ", "").replace("_", "")
            for av_team in available_teams:
                av_normalized = av_team.lower().replace(" ", "").replace("_", "")
                if lb_normalized == av_normalized:
                    matched_teams.append(av_team)
                    break
    
    return matched_teams


def main():
    """主函数"""
    global current_model_config, current_model_name
    
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
    
    # 获取与榜单匹配的团队
    matched_teams = get_matched_teams()
    
    if not matched_teams:
        print("未找到任何与榜单匹配的团队文件！")
        return
    
    # 生成对应的TSV文件列表
    matched_tsv_files = [f"{team}_english.tsv" for team in matched_teams]
    
    print(f"\n🎯 找到 {len(matched_tsv_files)} 个匹配的团队文件:")
    for i, file in enumerate(matched_tsv_files, 1):
        team_name = file.replace('_english.tsv', '')
        print(f"{i:2d}. {team_name}")
    
    # 让用户选择要处理的文件
    while True:
        try:
            choice = input(f"\n请选择要处理的文件 (1-{len(matched_tsv_files)}) 或输入 'all' 处理所有匹配团队: ").strip()
            if choice.lower() == 'all':
                selected_files = matched_tsv_files
                break
            else:
                file_idx = int(choice) - 1
                if 0 <= file_idx < len(matched_tsv_files):
                    selected_files = [matched_tsv_files[file_idx]]
                    break
                else:
                    print(f"请输入 1-{len(matched_tsv_files)} 之间的数字或 'all'")
        except ValueError:
            print("请输入有效的数字或 'all'")
    
    # 让用户选择评估方法
    print("\n请选择评估方法:")
    print("1. Zero-shot 评估 (不提供示例)")
    print("2. Few-shot 评估 (提供示例)")

    while True:
        choice = input("请输入选择 (1 或 2): ").strip()
        if choice == "1":
            evaluation_method = "zero_shot"
            evaluation_func = etd_zero_shot_evaluation
            break
        elif choice == "2":
            evaluation_method = "few_shot"
            evaluation_func = etd_few_shot_evaluation
            break
        else:
            print("无效选择，请输入 1 或 2")

    print(f"已选择 {evaluation_method} 评估方法")
    
    # 处理每个选中的模型
    for model_idx, model_name in enumerate(selected_models, 1):
        current_model_name = model_name
        current_model_config = config[model_name]
        
        print(f"\n🤖 使用模型: {model_name} ({current_model_config['model']})")
        print(f"模型进度: {model_idx}/{len(selected_models)}")
        
        # 处理每个选中的文件
        print(f"\n🚀 开始处理 {len(selected_files)} 个匹配团队的文件")
        for i, tsv_file in enumerate(selected_files, 1):
            print(f"\n{'='*60}")
            print(f"处理进度: {i}/{len(selected_files)} - {tsv_file} (模型: {model_name})")
            print(f"{'='*60}")
            
            # 检查文件是否存在
            data_path = os.path.join(script_dir, tsv_file)
            if not os.path.exists(data_path):
                print(f"⚠️  文件不存在，跳过: {tsv_file}")
                continue
                
            # 读取数据文件
            try:
                df_c = pd.read_csv(data_path, sep="\t")
            except Exception as e:
                print(f"❌ 读取文件 {tsv_file} 失败: {e}")
                continue
            
            # 获取队伍名称（去掉_english.tsv后缀）
            team_name = tsv_file.replace('_english.tsv', '')
            
            # 创建结果文件夹和文件名 - 根据模型名称创建不同的文件夹
            parent_dir = os.path.dirname(script_dir)  # 获取上级目录 (data/result)
            llm_evolution_dir = os.path.join(parent_dir, "llm_evolution")
            results_dir = os.path.join(llm_evolution_dir, f"{team_name}_en", model_name)
            os.makedirs(results_dir, exist_ok=True)
            
            if evaluation_method == "zero_shot":
                results_filename = os.path.join(results_dir, f"etd_zero_shot_results_en_ENPrompt_{model_name}.csv")
            else:
                results_filename = os.path.join(results_dir, f"etd_few_shot_results_en_ENPrompt_{model_name}.csv")
            
            process_file(df_c, results_filename, evaluation_func, team_name, model_name)
    
    print(f"\n🎉 所有模型和团队处理完成！")
    print(f"📁 结果保存在: {os.path.join(os.path.dirname(script_dir), 'llm_evolution')}")


def process_file(df_c, results_filename, evaluation_func, team_name, model_name=None):
    """处理单个文件的函数"""
    print(f"\n处理队伍: {team_name}")
    if model_name:
        print(f"使用模型: {model_name}")
    print(f"数据量: {len(df_c)} 条")

    # 创建结果文件路径
    
    # 检查是否存在已有结果文件
    if os.path.exists(results_filename):
        df_current = pd.read_csv(results_filename)
        print(f"发现已有结果文件，已处理 {len(df_current)} 条数据")
        
        # 找出未处理的数据
        missing_indices = find_missing_data(df_c, df_current)
        if not missing_indices:
            print(f"队伍 {team_name} 的所有数据都已处理完成！")
            return
        
        print(f"还有 {len(missing_indices)} 条数据需要处理")
        start_indices = missing_indices
    else:
        # 创建新的结果文件
        df_current = pd.DataFrame(
            columns=[
                "toxic_sentence",
                "neutral_sentence", 
                "STA",  # 风格迁移准确性
                "CS",   # 内容保留度
                "FS",   # 流畅性
                "j_score",   # 联合分数
                "reasoning",  # 评估理由
            ]
        )
        df_current.to_csv(results_filename, index=False)
        start_indices = list(range(len(df_c)))
        print(f"创建新的结果文件，开始处理 {len(df_c)} 条数据...")
    
    # 处理数据
    success_count = 0
    new_results = []
    
    for i, idx in enumerate(start_indices):
        if idx < len(df_c):
            row = df_c.iloc[idx]
            print(f"\n正在处理第 {idx+1} 条数据 ({i+1}/{len(start_indices)})...")
            print(f"有毒句子: {row['toxic_sentence'][:50]}{'...' if len(row['toxic_sentence']) > 50 else ''}")
            print(f"中性句子: {row['neutral_sentence'][:50]}{'...' if len(row['neutral_sentence']) > 50 else ''}")
            
            result = evaluation_func(row["toxic_sentence"], row["neutral_sentence"])
            
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
                
            # 每处理完一条数据后短暂休息，避免请求过于频繁
            if i < len(start_indices) - 1:  # 不是最后一条数据
                time.sleep(2)
    
    # 保存剩余的结果
    if new_results:
        df_new = pd.DataFrame(new_results)
        df_updated = pd.concat([df_current, df_new], ignore_index=True)
        df_updated.to_csv(results_filename, index=False)
        print(f"\n💾 最终保存完成")
    
    # 重新读取最终结果并计算统计信息
    df_final = pd.read_csv(results_filename)
    
    print(f"\n📊 队伍 {team_name} 处理总结:")
    if model_name:
        print(f"   - 使用模型: {model_name}")
    print(f"   - 尝试处理: {len(start_indices)} 条数据")
    print(f"   - 成功处理: {success_count} 条数据")
    print(f"   - 失败数量: {len(start_indices) - success_count} 条数据")
    print(f"   - 成功率: {success_count/len(start_indices)*100:.1f}%")
    print(f"   - 文件中总数据量: {len(df_final)} 条")
    
    # 计算各指标的平均分
    if len(df_final) > 0:
        print(f"\n📈 评估结果统计:")
        print(f"   - STA平均分: {df_final['STA'].mean():.2f}")
        print(f"   - CS平均分: {df_final['CS'].mean():.2f}")
        print(f"   - FS平均分: {df_final['FS'].mean():.2f}")
        print(f"   - JS平均分: {df_final['j_score'].mean():.2f}")
    
    if len(df_final) < len(df_c):
        remaining = len(df_c) - len(df_final)
        print(f"   - 仍有 {remaining} 条数据未处理，可以再次运行此脚本继续处理")
    else:
        model_info = f" (模型: {model_name})" if model_name else ""
        print(f"🎉 队伍 {team_name}{model_info} 的所有数据处理完成！")


if __name__ == "__main__":
    main()
