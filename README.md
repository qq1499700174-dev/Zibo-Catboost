### Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.13.12, catboost 1.2.10, optuna 4.9.0, shap 0.52.0, scikit-learn 1.8.0, xgboost 3.2.0 and lightgbm 4.6.0.

### Usage

```bash
# Reproduce the paper's model and SHAP results (a few seconds)
python train_model.py

# Re-run the Optuna hyperparameter search (~15 minutes, optional)
python optuna_search.py
```

`train_model.py` writes `catboost_model_pan_noscale.model`, `shap_values.npy`, `shap_feature_importance.csv` and `explainer.pkl`.
`optuna_search.py` writes `optuna_best_params.json` and `model_comparison.csv`.

### Results

Final CatBoost model on the test set: **R² = 0.92, RMSE = 0.45 ppbv, MAE = 0.28 ppbv, MAPE = 16%** (training set: R² = 0.999). SHAP analysis attributes 69.9% of the total contribution to chemical factors and 30.1% to meteorological factors.

### Note on reproducibility

`train_model.py` uses the hyperparameters reported in the paper, which are fixed in the script. Running it reproduces the final model and its SHAP values exactly, on any machine.

`optuna_search.py` re-runs the hyperparameter search. Because Optuna is a sequential search — each trial is proposed based on the cross-validation scores of all previous trials, which come from actually training multithreaded gradient boosting models — the search trajectory depends on the machine and the library versions, so results may differ slightly between environments. The script records the results obtained on our machine in its header comments.

---

## 中文说明

### 概述

本仓库包含论文所用的数据与代码：训练用于预测 PAN（过氧乙酰硝酸酯）浓度的 CatBoost 模型，并用 SHAP 方法对模型进行可解释性分析。

模型输入为 12 个特征——5 个气象要素（温度、绝对湿度、纬向风、经向风、太阳辐射）和 7 个化学要素（NOx、NO/NO₂、HONO、PM₂.₅、O₃、PM₁、TVOC），共 1,704 条小时样本。数据集经随机打乱后按 80%（训练集）/ 20%（测试集）划分。超参数使用 Optuna 优化（每个模型 30 次 trial、5 折交叉验证、以负 MAPE 为优化目标），并对 CatBoost、XGBoost、LightGBM、Random Forest 四种集成算法做了对比。CatBoost 表现最优，被选为最终模型。

### 文件说明

| 文件 | 说明 |
| --- | --- |
| `X_noscale.npy` | 特征矩阵，形状 (1704, 12)，未归一化 |
| `y_noscale.npy` | 目标变量 PAN，形状 (1704, 1)，未归一化 |
| `train_model.py` | 用论文超参数训练最终 CatBoost 模型、评估并做 SHAP 分析 |
| `optuna_search.py` | 重跑四种模型的 Optuna 超参数搜索，输出性能对比 |
| `catboost_model_pan_noscale.model` | 论文使用的 CatBoost 模型 |
| `shap_values.npy` | 最终模型的 SHAP 值，形状 (1704, 12) |
| `shap_feature_importance.csv` | 基于平均绝对 SHAP 值的特征重要性排名 |
| `explainer.pkl` | 序列化的 `shap.TreeExplainer`（joblib） |
| `requirements.txt` | 依赖包清单 |

**特征顺序**——`X_noscale.npy` 的 12 列严格按以下顺序排列：

```
温度、湿度、U（纬向风）、V（经向风）、辐射、
TVOC、NOx、HONO、PM₂.₅、O₃、PM₁、NO/NO₂
```

### 运行环境

```bash
pip install -r requirements.txt
```

已在 Python 3.13.12 + catboost 1.2.10 + optuna 4.9.0 + shap 0.52.0 + scikit-learn 1.8.0 + xgboost 3.2.0 + lightgbm 4.6.0 下测试通过。

### 使用方法

```bash
# 复现论文模型与 SHAP 结果（几秒完成）
python train_model.py

# 重跑 Optuna 超参数搜索（约 15 分钟，可选）
python optuna_search.py
```

`train_model.py` 输出 `catboost_model_pan_noscale.model`、`shap_values.npy`、`shap_feature_importance.csv`、`explainer.pkl`。
`optuna_search.py` 输出 `optuna_best_params.json`、`model_comparison.csv`。

### 结果

最终 CatBoost 模型在测试集上：**R² = 0.92，RMSE = 0.45 ppbv，MAE = 0.28 ppbv，MAPE = 16%**（训练集 R² = 0.999）。SHAP 分析显示化学要素贡献占总贡献的 69.9%，气象要素占 30.1%。

### 关于复现性

`train_model.py` 使用论文报告的超参数（已固定写在脚本中），在任何机器上运行都能**完全一致地**复现最终模型及其 SHAP 值。

`optuna_search.py` 用于重跑超参数搜索。由于 Optuna 是串行搜索——每一轮 trial 的参数由之前各轮的实际交叉验证得分决定，而得分来自真实训练的多线程梯度提升模型——搜索轨迹会受机器与库版本影响，不同环境下结果可能有细微差异。脚本开头的注释中记录了我们在本机搜索得到的结果。
