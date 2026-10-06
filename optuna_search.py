# -*- coding: utf-8 -*-
"""
PAN 模型超参数搜索脚本（Optuna）
================================
对 CatBoost / XGBoost / LightGBM / RandomForest 四种模型分别执行超参数搜索
（每个模型 30 次 trial，5 折交叉验证，以 MAPE 为优化目标），并用各自的最优参数
在测试集上训练评估，输出四种模型的性能对比。

输入数据（与本脚本在同一目录下）:
    X_noscale.npy  : 特征矩阵，形状 (n_samples, 12)，未归一化
    y_noscale.npy  : 目标变量 PAN，形状 (n_samples, 1)，未归一化

输出:
    optuna_best_params.json   各模型搜索得到的最优超参数
    model_comparison.csv      四种模型在测试集上的性能对比

注意:
    最终模型由 train_model.py 用固定的论文超参数训练，本脚本只负责调参过程。
    Optuna 为串行搜索——每轮 trial 的参数由前面各轮的实际交叉验证得分决定，而
    得分来自真实训练模型，多线程 GBDT 在不同机器 / 库版本下的浮点求和顺序存在
    极小差异，早期任一轮得分变化都会使整条搜索轨迹偏移。因此换环境重跑，结果
    可能与下面记录的不同，论文中的模型以 train_model.py 的固定参数为准。

---------------------------------------------------------------------------
本机历史运行记录（2026-10-06，TPESampler(seed=42)，30 trials）:

  CatBoost      CV MAPE 0.1560%   RMSE 0.4363  R² 0.9230  MAE 0.2770  MAPE 15.98%
      iterations=865, depth=7, learning_rate=0.06077933753233,
      l2_leaf_reg=1.2685337387584834, subsample=0.9343842874949141,
      colsample_bylevel=0.8567048436929133, min_child_samples=19

  XGBoost       CV MAPE 0.1710%   RMSE 0.4860  R² 0.9044  MAE 0.2989  MAPE 17.37%
      n_estimators=603, max_depth=9, learning_rate=0.03248256232455074,
      subsample=0.6274889158559916, colsample_bylevel=0.8710962973731542,
      reg_alpha=1.032517825687112, reg_lambda=1.3055792681672975, min_child_weight=5

  LightGBM      CV MAPE 0.1765%   RMSE 0.5433  R² 0.8805  MAE 0.3344  MAPE 19.42%
      n_estimators=783, max_depth=8, learning_rate=0.054898641012869565,
      subsample=0.7661040109300066, colsample_bylevel=0.8212662822014305,
      reg_alpha=1.4247012756647537, reg_lambda=3.892023452622589, min_child_samples=8

  RandomForest  CV MAPE 0.2624%   RMSE 0.6634  R² 0.8219  MAE 0.4218  MAPE 25.40%
      n_estimators=641, max_depth=10, min_samples_split=18,
      min_samples_leaf=5, max_features=0.995284576827681

对比可见：重新搜索得到的 CatBoost 在该测试集上略优（R² 0.9230 vs 论文 0.9177），
但 SHAP 重要性排序随之变化（O₃ 与 PM₂.₅ 互换、TVOC 与 NO/NO₂ 互换），与论文
图件不符，故正文与图件仍以 train_model.py 中固定参数的模型为准。
---------------------------------------------------------------------------
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# 兼容 Windows 控制台默认 GBK 编码，避免特殊字符（如 ₂、₅）打印时报 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# ==================== 搜索配置 ====================
N_TRIALS = 30          # 每个模型的 trial 次数
SEED = 42              # 采样器与模型随机种子

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X_PATH = os.path.join(BASE_DIR, "X_noscale.npy")
Y_PATH = os.path.join(BASE_DIR, "y_noscale.npy")


def build_model(name, params):
    """按模型名称和参数构建模型实例"""
    if name == "CatBoost":
        return CatBoostRegressor(**params, random_seed=SEED, verbose=0,
                                 allow_writing_files=False)
    elif name == "XGBoost":
        return XGBRegressor(**params, objective='reg:squarederror',
                            random_state=SEED, n_jobs=-1)
    elif name == "LightGBM":
        return lgb.LGBMRegressor(**params, random_state=SEED,
                                 verbosity=-1, n_jobs=-1)
    elif name == "RandomForest":
        return RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)
    else:
        raise ValueError(f"未知模型: {name}")


# ==================== 数据读取 ====================
X = np.load(X_PATH)
y = np.load(Y_PATH).reshape(-1)

print("=" * 60)
print("数据加载完成（未归一化）")
print("=" * 60)
print(f"X 形状: {X.shape} | y 形状: {y.shape}")

# 与 train_model.py 保持一致：同样的划分方式与随机种子
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(f"训练集: {X_train.shape[0]} 样本 | 测试集: {X_test.shape[0]} 样本")


# ==================== 各模型的目标函数 ====================
def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_seed': SEED,
        'verbose': 0,
        'allow_writing_files': False
    }
    model = CatBoostRegressor(**params)
    return cross_val_score(model, X_train, y_train,
                           scoring='neg_mean_absolute_percentage_error',
                           cv=5, n_jobs=-1).mean()


def objective_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': SEED
    }
    model = XGBRegressor(**params, objective='reg:squarederror', n_jobs=-1)
    return cross_val_score(model, X_train, y_train,
                           scoring='neg_mean_absolute_percentage_error',
                           cv=5, n_jobs=-1).mean()


def objective_lightgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': SEED,
        'verbosity': -1
    }
    model = lgb.LGBMRegressor(**params, n_jobs=-1)
    return cross_val_score(model, X_train, y_train,
                           scoring='neg_mean_absolute_percentage_error',
                           cv=5, n_jobs=-1).mean()


def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
        'max_features': trial.suggest_float('max_features', 0.2, 1.0),
        'random_state': SEED,
        'n_jobs': -1
    }
    model = RandomForestRegressor(**params)
    return cross_val_score(model, X_train, y_train,
                           scoring='neg_mean_absolute_percentage_error',
                           cv=5, n_jobs=-1).mean()


# ==================== 执行搜索 ====================
print("=" * 60)
print(f"Optuna 超参数搜索（每个模型 {N_TRIALS} 次 trial）")
print("=" * 60)

studies = {}
for name, func in [("CatBoost", objective_catboost),
                   ("XGBoost", objective_xgboost),
                   ("LightGBM", objective_lightgbm),
                   ("RandomForest", objective_rf)]:
    print(f"\n优化 {name}...")
    study = optuna.create_study(direction='maximize',
                                sampler=TPESampler(seed=SEED))
    study.optimize(func, n_trials=N_TRIALS, show_progress_bar=True)
    studies[name] = study
    print(f"最佳参数: {study.best_params}")
    print(f"最佳交叉验证 MAPE: {-study.best_value:.4f}%")

# 保存最优超参数
best_params = {name: study.best_params for name, study in studies.items()}
params_path = os.path.join(BASE_DIR, "optuna_best_params.json")
with open(params_path, "w", encoding="utf-8") as f:
    json.dump(best_params, f, ensure_ascii=False, indent=2)
print(f"\n最优超参数已保存为: {os.path.basename(params_path)}")

# ==================== 用最优参数训练并对比 ====================
comparison_results = []
for name, study in studies.items():
    model = build_model(name, study.best_params.copy())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    comparison_results.append({
        "模型": name,
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE (%)": np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    })

comparison_df = pd.DataFrame(comparison_results).sort_values(by="MAPE (%)").reset_index(drop=True)
print("\n" + "=" * 80)
print("各最优模型在测试集上的完整性能对比")
print("=" * 80)
print(comparison_df.to_string(index=False, float_format="%.4f"))
comparison_df.to_csv(os.path.join(BASE_DIR, "model_comparison.csv"),
                     index=False, encoding="utf-8-sig")
print("\n性能对比已保存为: model_comparison.csv")
print("\n提示: 最终模型请用 train_model.py 训练（论文参数固定，可跨环境复现）。")
