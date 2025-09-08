# 🏆 LLM文本解毒评估排名报告
**生成时间**: 2025-08-08 20:50:37
**评估方法**: Zero-shot ETD评估
**参与团队**: 17 个
**评估样本**: 每队100条数据

## 📊 评估指标说明
- **STA (Style Transfer Accuracy)**: 风格迁移准确性 (0-1)
- **CS (Content Similarity)**: 内容相似度 (0-1)
- **FS (Fluency)**: 流畅性 (0-1)
- **j_score**: 联合分数 (STA × CS × FS)
- **完美分**: j_score = 1.0 的样本比例

## 📈 总体统计
- **平均j_score**: 0.655
- **最高j_score**: 0.828 (Team cake)
- **最低j_score**: 0.259 (delete_baseline)
- **标准差**: 0.151

## 🏅 详细排名

| 排名 | 团队名称 | j_score | STA | CS | FS | 完美分率 | 等级 |
|------|----------|---------|-----|----|----|----------|------|
| 1 | Team cake | 0.828 | 0.960 | 0.920 | 0.945 | 68.0% | 🥇 优秀 |
| 2 | Team_SINAI | 0.825 | 0.965 | 0.905 | 0.945 | 69.0% | 🥇 优秀 |
| 3 | mkrisnai | 0.802 | 0.900 | 0.955 | 0.935 | 66.0% | 🥇 优秀 |
| 4 | SomethingAwful | 0.790 | 0.970 | 0.930 | 0.880 | 61.0% | 🥈 良好 |
| 5 | Team Iron Autobots | 0.772 | 0.995 | 0.810 | 0.950 | 63.0% | 🥈 良好 |
| 6 | dkenco | 0.752 | 0.975 | 0.810 | 0.960 | 59.0% | 🥈 良好 |
| 7 | erehulka | 0.713 | 0.960 | 0.910 | 0.825 | 48.0% | 🥈 良好 |
| 8 | gleb.shnshn | 0.698 | 0.970 | 0.830 | 0.845 | 53.0% | 🥉 中等 |
| 9 | Team MarSan_AI | 0.680 | 0.895 | 0.945 | 0.805 | 47.0% | 🥉 中等 |
| 10 | Team NLPunks | 0.680 | 0.935 | 0.835 | 0.870 | 48.0% | 🥉 中等 |
| 11 | VitalyProtasov | 0.623 | 0.925 | 0.815 | 0.750 | 45.0% | 🥉 中等 |
| 12 | backtranslation_baseline | 0.605 | 0.865 | 0.875 | 0.775 | 41.0% | 🥉 中等 |
| 13 | ZhongyuLuo | 0.595 | 0.870 | 0.870 | 0.760 | 39.0% | 📊 一般 |
| 14 | nikita.sushko | 0.525 | 0.905 | 0.770 | 0.720 | 30.0% | 📊 一般 |
| 15 | mt5_baseline | 0.497 | 0.790 | 0.815 | 0.750 | 29.0% | 📉 待改进 |
| 16 | Team nlpjoyers | 0.484 | 0.780 | 0.815 | 0.745 | 29.0% | 📉 待改进 |
| 17 | delete_baseline | 0.259 | 0.690 | 0.755 | 0.585 | 5.0% | 📉 待改进 |

## 🎯 分层分析

### 🥇 优秀团队 (j_score ≥ 0.8) - 3 个
- **Team cake**: 0.828
- **Team_SINAI**: 0.825
- **mkrisnai**: 0.802

### 🥈 良好团队 (0.7 ≤ j_score < 0.8) - 4 个
- **SomethingAwful**: 0.790
- **Team Iron Autobots**: 0.772
- **dkenco**: 0.752
- **erehulka**: 0.713

### 🥉 中等团队 (0.6 ≤ j_score < 0.7) - 5 个
- **gleb.shnshn**: 0.698
- **Team MarSan_AI**: 0.680
- **Team NLPunks**: 0.680
- **VitalyProtasov**: 0.623
- **backtranslation_baseline**: 0.605

### 📊 待改进团队 (j_score < 0.6) - 5 个
- **ZhongyuLuo**: 0.595
- **nikita.sushko**: 0.525
- **mt5_baseline**: 0.497
- **Team nlpjoyers**: 0.484
- **delete_baseline**: 0.259

## 🔍 各维度表现分析

### 风格迁移准确性 (STA) 最佳
**Team Iron Autobots**: 0.995

### 内容相似度 (CS) 最佳
**mkrisnai**: 0.955

### 流畅性 (FS) 最佳
**dkenco**: 0.960

### 完美分率最高
**Team_SINAI**: 69.0%
