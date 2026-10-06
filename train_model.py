# -*- coding: utf-8 -*-
"""
PAN 浓度可解释机器学习模型 —— 训练脚本
=======================================
输入数据（与本脚本在同一目录下）:
    X_noscale.npy  : 特征矩阵，形状 (n_samples, 12)，未归一化
    y_noscale.npy  : 目标变量 PAN，形状 (n_samples, 1)，未归一化

流程:
    1. 读取 npy 数据
    2. 按 8:2 划分训练集 / 测试集
    3. 用论文中的最优超参数训练 CatBoost 模型
    4. 在训练集 / 测试集上评估（RMSE / R² / MAE / MAPE）
    5. 保存模型
    6. SHAP 特征重要性分析（保存 SHAP 值、特征重要性表与解释器）

超参数说明:
    论文中的最优超参数由 Optuna 搜索得到（搜索过程见同目录 optuna_search.py）。
    由于 Optuna 为串行搜索，搜索结果依赖机器与库版本、无法跨环境复现，因此这里
    把论文使用的超参数固定下来，保证任何环境下运行都得到与论文一致的模型。
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import shap
import warnings

warnings.filterwarnings('ignore')

# 兼容 Windows 控制台默认 GBK 编码，避免特殊字符（如 ₂、₅）打印时报 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# ==================== 论文最优模型超参数 ====================
# 对应测试集表现: RMSE = 0.4510, R² = 0.9177, MAE = 0.2776, MAPE = 16.20%
PAPER_BEST_PARAMS = {
    "iterations": 746,
    "depth": 8,
    "learning_rate": 0.05603206903,
    "l2_leaf_reg": 1.97842741,
    "subsample": 0.9456014633,
    "colsample_bylevel": 0.9592036009,
    "min_child_samples": 32,
}

# ==================== 文件路径 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X_PATH = os.path.join(BASE_DIR, "X_noscale.npy")
Y_PATH = os.path.join(BASE_DIR, "y_noscale.npy")

# 特征名称（与 X_noscale.npy 的 12 列一一对应）
FEATURE_NAMES = [
    "温度", "湿度", "U", "V", "辐射",
    "TVOC", "NOx", "HONO", "PM₂.₅", "O₃", "PM₁", "NO/NO₂"
]
TARGET_NAME = "PAN"


def evaluate_regression(y_true, y_pred, dataset_name="数据集"):
    """计算并打印 MSE / RMSE / MAE / R² / MAPE"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    print(f"\n=== {dataset_name} 评估 ===")
    print(f"MAPE: {mape:.4f}% | MSE: {mse:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.6f}")
    return mse, rmse, mae, r2, mape


# ==================== 数据读取 ====================
X = np.load(X_PATH)
y = np.load(Y_PATH).reshape(-1)

if X.shape[1] != len(FEATURE_NAMES):
    raise ValueError(f"特征数与特征名称不匹配: X 有 {X.shape[1]} 列，"
                     f"FEATURE_NAMES 有 {len(FEATURE_NAMES)} 个")

print("=" * 60)
print("数据加载完成（未归一化）")
print("=" * 60)
print(f"X 形状: {X.shape} | y 形状: {y.shape}")
print(f"特征: {FEATURE_NAMES}")
print(f"{TARGET_NAME}: 均值 = {y.mean():.4f}, 标准差 = {y.std():.4f}, "
      f"最小值 = {y.min():.4f}, 最大值 = {y.max():.4f}")

# ==================== 划分训练 / 测试集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n训练集: {X_train.shape[0]} 样本 | 测试集: {X_test.shape[0]} 样本")

# ==================== 训练模型 ====================
print("=" * 60)
print("训练最终模型: CatBoost")
print("=" * 60)
print(f"超参数: {PAPER_BEST_PARAMS}")

model = CatBoostRegressor(**PAPER_BEST_PARAMS, random_seed=42, verbose=0,
                          allow_writing_files=False)
model.fit(X_train, y_train)

# ==================== 保存模型 ====================
save_path = os.path.join(BASE_DIR, "catboost_model_pan_noscale.model")
model.save_model(save_path)
print(f"\n模型已保存为: {os.path.basename(save_path)}")

# ==================== 训练集 / 测试集评估 ====================
evaluate_regression(y_train, model.predict(X_train), "CatBoost 训练集")
evaluate_regression(y_test, model.predict(X_test), "CatBoost 测试集")

# ==================== SHAP 分析（原始特征） ====================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
if hasattr(shap_values, 'values'):
    shap_values_np = shap_values.values
else:
    shap_values_np = np.asarray(shap_values)

feature_importance = pd.DataFrame({
    "特征名称": FEATURE_NAMES,
    "重要性得分": np.abs(shap_values_np).mean(axis=0)
}).sort_values(by="重要性得分", ascending=False).reset_index(drop=True)

print("\n" + "=" * 60)
print("特征 SHAP 重要性排名（CatBoost，原始数据）")
print("=" * 60)
print(feature_importance.round(4).to_string(index=False))

feature_importance.to_csv(os.path.join(BASE_DIR, "shap_feature_importance.csv"),
                          index=False, encoding="utf-8-sig")
np.save(os.path.join(BASE_DIR, "shap_values.npy"), shap_values_np)
# 压缩保存（未压缩的 explainer 约 110 MB，压缩后几 MB，便于上传 GitHub）
joblib.dump(explainer, os.path.join(BASE_DIR, "explainer.pkl"), compress=3)
print("\n已保存: shap_feature_importance.csv, shap_values.npy, explainer.pkl")

# ==================== 污染物 / 气象要素贡献占比 ====================
pollutants = ["TVOC", "NOx", "HONO", "PM₂.₅", "O₃", "PM₁", "NO/NO₂"]
meteorological_factors = ["温度", "湿度", "U", "V", "辐射"]

feature_shap_sums = np.abs(shap_values_np).sum(axis=0)
pollutant_shap_sum = sum(feature_shap_sums[FEATURE_NAMES.index(p)] for p in pollutants)
meteorological_shap_sum = sum(feature_shap_sums[FEATURE_NAMES.index(m)] for m in meteorological_factors)
total_shap_sum = pollutant_shap_sum + meteorological_shap_sum

print("\n" + "=" * 60)
print("SHAP 贡献占比")
print("=" * 60)
print(f"大气污染物: {pollutant_shap_sum / total_shap_sum * 100:.2f}%")
print(f"气象要素  : {meteorological_shap_sum / total_shap_sum * 100:.2f}%")
