# TextDetox CLEF-2024 Evaluation and Ranking Analysis

This project is a comprehensive evaluation and ranking analysis system based on the CLEF-2024 Text Detoxification shared task. It includes human evaluation, LLM evaluation, and comparative analysis of various evaluation metrics, covering text detoxification effectiveness assessment across 9 languages.

## 🌟 Key Features

- **Multi-language Support**: Covers English, Spanish, German, Chinese, Arabic, Hindi, Ukrainian, Russian, and Amharic
- **Multi-dimensional Evaluation**: Includes human evaluation, LLM evaluation (GPT-4, DeepSeek, Qwen, etc.), and automatic evaluation metrics
- **Comprehensive Ranking System**: Ranking analysis based on various consistency indicators
- **Detailed Comparison Reports**: Provides in-depth evaluation quality analysis and model comparisons

## 📁 Project Structure

```
textdetox_clef_2024/
├── data/                          # Data directory
│   ├── human_evaluation/          # Human evaluation data
│   │   ├── SomethingAwful/        # Evaluation results from participating teams
│   │   ├── Team Iron Autobots/
│   │   ├── Team NLPunks/
│   │   └── ...                    # Other teams and baseline methods
│   ├── result/                    # Evaluation results
│   │   ├── llm_evolution/         # LLM evaluation evolution results
│   │   ├── en_py/                 # English evaluation scripts and results
│   │   └── zh_py/                 # Chinese evaluation scripts and results
│   ├── evaluation_results/        # Comprehensive evaluation results
│   │   ├── llm_evaluation/        # LLM evaluation comparisons
│   │   ├── comprehensive/         # Comprehensive model comparisons
│   │   └── new_models/            # New model evaluation results
│   └── ranking_results/           # Ranking results
│       ├── en/                    # English rankings
│       └── zh/                    # Chinese rankings
├── scripts/                       # Analysis scripts
│   ├── compare_all_evaluations_*.py  # Evaluation comparison scripts
│   ├── generate_*_ranking*.py         # Ranking generation scripts
│   └── calculate_fluency_score.py     # Fluency score calculation
├── tools/                         # Evaluation tools
│   └── chrF-master/              # chrF++ evaluation tool
├── docs/                         # Documentation directory
│   ├── evaluation_comparison_report.md  # Evaluation comparison report
│   ├── LLM_Evaluation_Ranking_Report.md # LLM evaluation ranking report
│   └── instructions/             # Evaluation guidelines
└── README.md                     # Project documentation
```

## 🚀 Quick Start

### Requirements

- Python 3.8+
- pandas, numpy, scipy, scikit-learn
- matplotlib, seaborn (for visualization)

### Install Dependencies

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

### Run Evaluation Scripts

1. **English Evaluation Comparison**:
```bash
cd scripts
python compare_all_evaluations_en.py
```

2. **Chinese Evaluation Comparison**:
```bash
python compare_all_evaluations_zh.py
```

3. **Generate Ranking Results**:
```bash
python generate_en_llm_ranking_few_shot.py
python generate_zh_llm_ranking.py
```

## 📊 Evaluation Dimensions

### Three-Dimensional Evaluation Framework

1. **Style Transfer Accuracy (STA)**: Measures the effectiveness of toxicity removal
2. **Content Similarity (CS)**: Evaluates semantic preservation quality
3. **Fluency Score (FS)**: Assesses language quality and readability

### Scoring Criteria

- **0 points**: Does not meet requirements
- **0.5 points**: Partially meets requirements
- **1 point**: Fully meets requirements

## 🎯 Main Features

### 1. Human Evaluation Analysis
- Multi-language human evaluation data collection
- Evaluator consistency analysis
- Quality control and statistical analysis

### 2. LLM Evaluation System
- Support for multiple large language models (GPT-4, DeepSeek, Qwen, etc.)
- Zero-shot and Few-shot evaluation modes
- Bilingual prompt engineering for Chinese and English

### 3. Comprehensive Ranking System
- Rankings based on Kendall Tau, Spearman, and Pearson correlations
- Mean Absolute Error and Mean Squared Error analysis
- Multi-dimensional comprehensive scoring

### 4. Comparative Analysis Reports
- Human evaluation vs LLM evaluation consistency analysis
- Evaluation quality comparison between different models
- Detailed statistical analysis and visualization

## 📈 Evaluation Metrics

### Consistency Metrics
- **Kendall Tau**: Rank correlation
- **Spearman Correlation**: Monotonic relationship measure
- **Pearson Correlation**: Linear relationship measure

### Error Metrics
- **Mean Absolute Error (MAE)**: Evaluation accuracy
- **Mean Squared Error (MSE)**: Evaluation stability

## 🏆 Participating Teams

The project evaluated text detoxification systems from multiple participating teams:

- **SomethingAwful**: Top-performing team
- **Team Iron Autobots**: Excellent comprehensive performance
- **Team NLPunks**: Innovative approach team
- **Team cake**: Stable performance team
- And several other teams and baseline methods

## 📊 Results Overview

### English Evaluation Results
- Best teams showed excellent performance across all dimensions
- LLM evaluation showed high consistency with human evaluation
- Few-shot mode performed better than Zero-shot mode

### Chinese Evaluation Results  
- Chinese text detoxification faces greater challenges
- Different models show variations in Chinese evaluation
- Cultural and linguistic characteristics affect evaluation quality

## 🔬 Technical Details

### LLM Evaluation Prompt Engineering
- Professional evaluator identity setup
- Detailed evaluation criteria explanation
- Example-guided few-shot evaluation

### Data Processing Pipeline
1. Raw data collection and cleaning
2. Multi-dimensional evaluation execution
3. Result standardization and comparison
4. Statistical analysis and ranking generation

## 📖 Related Papers

If you wish to cite this work, please refer to the following publications:

```bibtex
@inproceedings{dementieva2024overview,
  title={Overview of the Multilingual Text Detoxification Task at PAN 2024},
  author={Dementieva, Daryna and Moskovskiy, Daniil and Babakov, Nikolay and others},
  booktitle={Working Notes of CLEF 2024},
  year={2024}
}
```

## 📞 Contact Information

- **Project Lead**: Daryna Dementieva (dardem96@gmail.com)
- **Technical Support**: Nikolay Babakov (bbkhse@gmail.com)  
- **Academic Supervisor**: Alexander Panchenko (a.panchenko@skol.tech)

## 📄 License

This project is open for academic research purposes. Please refer to the CLEF-2024 shared task regulations for specific usage terms.

## 🤝 Contributing

We welcome Issues and Pull Requests to improve the project. Please ensure:

1. Follow existing code style
2. Add necessary tests and documentation
3. Provide detailed descriptions of modifications and reasons

## 🔗 Related Resources

- [CLEF-2024 TextDetox Official Page](https://pan.webis.de/clef24/pan24-web/text-detoxification.html)
- [ParaDetox Dataset](https://huggingface.co/textdetox)
- [Project Documentation](./docs/)

---

*This project is an evaluation and ranking analysis system for the CLEF-2024 Text Detoxification shared task, dedicated to advancing multilingual text detoxification technology.*