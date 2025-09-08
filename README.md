# TextDetox CLEF-2024 评估与排名分析

本项目是基于 CLEF-2024 文本去毒化共享任务的综合评估和排名分析系统。项目包含人工评估、LLM评估以及多种评价指标的对比分析，覆盖9种语言的文本去毒化效果评估。

## 🌟 项目特色

- **多语言支持**: 覆盖英语、西班牙语、德语、中文、阿拉伯语、印地语、乌克兰语、俄语和阿姆哈拉语
- **多维度评估**: 包含人工评估、LLM评估（GPT-4、DeepSeek、Qwen等）和自动评估指标
- **综合排名系统**: 基于多种一致性指标的排名分析
- **详细比较报告**: 提供深入的评估质量分析和模型对比

## 📁 项目结构

```
textdetox_clef_2024/
├── data/                          # 数据目录
│   ├── human_evaluation/          # 人工评估数据
│   │   ├── SomethingAwful/        # 各参赛团队的评估结果
│   │   ├── Team Iron Autobots/
│   │   ├── Team NLPunks/
│   │   └── ...                    # 其他团队和基线方法
│   ├── result/                    # 评估结果
│   │   ├── llm_evolution/         # LLM评估演进结果
│   │   ├── en_py/                 # 英语评估脚本和结果
│   │   └── zh_py/                 # 中文评估脚本和结果
│   ├── evaluation_results/        # 综合评估结果
│   │   ├── llm_evaluation/        # LLM评估对比
│   │   ├── comprehensive/         # 综合模型对比
│   │   └── new_models/            # 新模型评估结果
│   └── ranking_results/           # 排名结果
│       ├── en/                    # 英语排名
│       └── zh/                    # 中文排名
├── scripts/                       # 分析脚本
│   ├── compare_all_evaluations_*.py  # 评估对比脚本
│   ├── generate_*_ranking*.py         # 排名生成脚本
│   └── calculate_fluency_score.py     # 流畅度评分计算
├── tools/                         # 评估工具
│   └── chrF-master/              # chrF++评估工具
├── docs/                         # 文档目录
│   ├── evaluation_comparison_report.md  # 评估对比报告
│   ├── LLM_Evaluation_Ranking_Report.md # LLM评估排名报告
│   └── instructions/             # 评估指导说明
└── README.md                     # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pandas, numpy, scipy, scikit-learn
- matplotlib, seaborn (用于可视化)

### 安装依赖

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

### 运行评估脚本

1. **英语评估对比**:
```bash
cd scripts
python compare_all_evaluations_en.py
```

2. **中文评估对比**:
```bash
python compare_all_evaluations_zh.py
```

3. **生成排名结果**:
```bash
python generate_en_llm_ranking_few_shot.py
python generate_zh_llm_ranking.py
```

## 📊 评估维度

### 三维评估框架

1. **风格转换准确性 (STA)**: 衡量毒性去除的有效性
2. **内容相似性 (CS)**: 评估语义保持质量  
3. **流畅度评分 (FS)**: 评估语言质量和可读性

### 评分标准

- **0分**: 不满足要求
- **0.5分**: 部分满足要求
- **1分**: 完全满足要求

## 🎯 主要功能

### 1. 人工评估分析
- 多语言人工评估数据收集
- 评估者一致性分析
- 质量控制和统计分析

### 2. LLM评估系统
- 支持多种大语言模型（GPT-4、DeepSeek、Qwen等）
- Zero-shot 和 Few-shot 评估模式
- 中英文双语提示工程

### 3. 综合排名系统
- 基于 Kendall Tau、Spearman、Pearson 相关性的排名
- 平均绝对误差和均方误差分析
- 多维度综合评分

### 4. 对比分析报告
- 人工评估 vs LLM评估一致性分析
- 不同模型间的评估质量对比
- 详细的统计分析和可视化

## 📈 评估指标

### 一致性指标
- **Kendall Tau**: 等级相关性
- **Spearman相关系数**: 单调关系度量
- **Pearson相关系数**: 线性关系度量

### 误差指标
- **平均绝对误差 (MAE)**: 评估精确度
- **均方误差 (MSE)**: 评估稳定性

## 🏆 参赛团队

项目评估了多个参赛团队的文本去毒化系统：

- **SomethingAwful**: 顶级表现团队
- **Team Iron Autobots**: 综合性能优秀
- **Team NLPunks**: 创新方法团队
- **Team cake**: 稳定表现团队
- 以及其他多个团队和基线方法

## 📊 结果概览

### 英语评估结果
- 最佳团队在所有维度上均表现出色
- LLM评估与人工评估显示出较高的一致性
- Few-shot模式比Zero-shot模式表现更好

### 中文评估结果  
- 中文文本去毒化面临更大挑战
- 不同模型在中文评估上存在差异
- 文化和语言特性影响评估质量

## 🔬 技术细节

### LLM评估提示工程
- 专业评估者身份设定
- 详细的评估标准说明
- 示例引导的few-shot评估

### 数据处理流程
1. 原始数据收集和清洗
2. 多维度评估执行
3. 结果标准化和对比
4. 统计分析和排名生成

## 📖 相关论文

如需引用本工作，请参考以下文献：

```bibtex
@inproceedings{dementieva2024overview,
  title={Overview of the Multilingual Text Detoxification Task at PAN 2024},
  author={Dementieva, Daryna and Moskovskiy, Daniil and Babakov, Nikolay and others},
  booktitle={Working Notes of CLEF 2024},
  year={2024}
}
```

## 📞 联系方式

- **项目负责人**: Daryna Dementieva (dardem96@gmail.com)
- **技术支持**: Nikolay Babakov (bbkhse@gmail.com)  
- **学术指导**: Alexander Panchenko (a.panchenko@skol.tech)

## 📄 许可证

本项目基于学术研究目的开放，具体使用条款请参考 CLEF-2024 共享任务规定。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。请确保：

1. 遵循现有的代码风格
2. 添加必要的测试和文档
3. 详细描述修改内容和原因

## 🔗 相关资源

- [CLEF-2024 TextDetox官方页面](https://pan.webis.de/clef24/pan24-web/text-detoxification.html)
- [ParaDetox数据集](https://huggingface.co/textdetox)
- [项目文档](./docs/)

---

*该项目是CLEF-2024文本去毒化共享任务的评估和排名分析系统，致力于推动多语言文本去毒化技术的发展。*