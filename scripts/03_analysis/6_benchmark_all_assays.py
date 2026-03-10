import os
import sys
import shutil
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置区域 (已修复路径) =================
RAW_DATA_PATH = "data/raw_assays.csv"
LABEL_FILE_PATH = "data/labels.csv"

# [关键修复]：根据你的日志，训练脚本把结果输出到了 data/features
RESULT_DIR = "data/features"

MODEL_SAVE_ROOT = "saved_models_by_assay" # 备份目录
# =======================================================

# 化合物名称映射
NAME_MAP = {
    "Dopamine": "Dopa", "Aripiprazole": "ARI", "Brexpiprazole": "BRE",
    "Cariprazine": "CAR", "Lisuride": "Lisu", "Rotigotine": "ROT",
    "UNC2458A": "UNC", "(S)-IHCH-7084": "S84",
    "(R)-IHCH-7010": "R10", "(S)-IHCH-7010": "S10",
    "Pramipexole": "PPX", # 补充可能缺失的映射，防止报错
    "(R)-IHCH-7041": "R7041",
    "(S)-IHCH-7041": "S7041"
}

def load_and_clean_raw_data():
    """读取原始 Assay 数据并清洗列名"""
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: 找不到 {RAW_DATA_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(RAW_DATA_PATH)
    # 清洗列名（去除空格）
    df.columns = [c.strip() for c in df.columns]
    
    # 标准化 Compound 列名
    if 'Compound' not in df.columns and 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Compound'})
        
    # 应用名称映射
    df['Compound'] = df['Compound'].apply(lambda x: NAME_MAP.get(x.strip(), x.strip()))
    
    return df

def run_training():
    """调用现有的训练脚本"""
    print("  >>> Starting training process...")
    # 使用 subprocess 调用 python 2_train_model_egnn.py
    result = subprocess.run([sys.executable, "2_train_model_egnn.py"], check=True)
    if result.returncode != 0:
        print("Error: 训练脚本执行失败")
        sys.exit(1)

def backup_results(assay_name):
    """将训练结果和模型权重备份到独立文件夹"""
    # 处理 assay_name 中的非法字符 (如 β, /, 空格) 以免创建文件夹失败
    safe_name = "".join([c if c.isalnum() else "_" for c in assay_name])
    target_dir = os.path.join(MODEL_SAVE_ROOT, safe_name)
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir) # 如果存在则清空，确保是新的
    os.makedirs(target_dir)
    
    # 1. 备份 CSV 结果
    loo_csv = os.path.join(RESULT_DIR, "loo_results_final.csv")
    if os.path.exists(loo_csv):
        shutil.copy(loo_csv, os.path.join(target_dir, "loo_results.csv"))
    else:
        print(f"  [Warn] Result CSV not found at {loo_csv}")
    
    # 2. 备份模型权重 (.pth)
    if os.path.exists(RESULT_DIR):
        for f in os.listdir(RESULT_DIR):
            if f.endswith(".pth"):
                shutil.copy(os.path.join(RESULT_DIR, f), os.path.join(target_dir, f))
    else:
        print(f"  [Error] Result dir {RESULT_DIR} not found!")
            
    print(f"  >>> Backup saved to: {target_dir}")

def evaluate_metrics(assay_name):
    """读取 LOO 结果计算指标"""
    loo_csv = os.path.join(RESULT_DIR, "loo_results_final.csv")
    if not os.path.exists(loo_csv):
        return None
    
    try:
        df = pd.read_csv(loo_csv)
        if len(df) < 2: return None # 数据太少无法计算相关性
        
        y_true = df['True']
        y_pred = df['Pred']
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        pearson, _ = pearsonr(y_true, y_pred)
        
        return {"Assay": assay_name, "RMSE": rmse, "Pearson_R": pearson}
    except Exception as e:
        print(f"  [Warn] Failed to evaluate metrics: {e}")
        return None

def main():
    # 0. 读取已完成的 benchmark，避免重复计算
    completed_assays = set() 
    summary_path = "assay_benchmark_summary.csv" 
    
    if os.path.exists(summary_path): 
        df_done = pd.read_csv(summary_path) 
        if "Assay" in df_done.columns: 
            completed_assays = set(df_done["Assay"].astype(str)) 
            print(f"[Resume] Found {len(completed_assays)} completed assays.")
    # 1. 准备数据
    df_raw = load_and_clean_raw_data()
    
    # 获取所有 Assay 列名 (排除 Compound 列)
    assay_columns = [c for c in df_raw.columns if c != "Compound"]
    print(f"Found {len(assay_columns)} assays: {assay_columns}")
    
    benchmark_results = []
    
    # 2. 循环测试
    for i, assay in enumerate(assay_columns):

        if assay in completed_assays: 
            print(f"\n[Skip] Assay [{assay}] already benchmarked. Skipping.") 
            continue

        print(f"\n{'='*60}")
        print(f"Benchmark Round {i+1}/{len(assay_columns)}: Testing Assay [{assay}]")
        print(f"{'='*60}")
        
        # A. 生成临时的 labels.csv
        # 注意：这里我们假设所有 Assay 都是百分比，统一除以 100
        current_labels = df_raw[['Compound', assay]].copy()
        current_labels.columns = ['Compound', 'Efficacy']
        
        # [安全检查] 确保是数值类型，且处理可能的缺失值
        current_labels['Efficacy'] = pd.to_numeric(current_labels['Efficacy'], errors='coerce')
        current_labels = current_labels.dropna() # 丢弃空值行
        
        current_labels['Efficacy'] = current_labels['Efficacy'] / 100.0
        
        # 保存到 data/labels.csv，覆盖旧文件
        current_labels.to_csv(LABEL_FILE_PATH, index=False)
        print(f"  [Label Gen] Generated labels using {assay} (Count: {len(current_labels)})")
        
        if len(current_labels) < 3:
            print("  [Skip] Not enough data points.")
            continue

        # B. 运行训练
        try:
            run_training()
        except Exception as e:
            print(f"  [Error] Training failed for {assay}: {e}")
            continue
            
        # C. 备份权重
        backup_results(assay)
        
        # D. 记录指标
        metrics = evaluate_metrics(assay)
        if metrics:
            print(f"  [Result] RMSE: {metrics['RMSE']:.4f}, R: {metrics['Pearson_R']:.4f}")
            benchmark_results.append(metrics)

    # 3. 汇总与可视化
    print(f"\n{'='*60}")
    print("Final Benchmark Summary")
    print(f"{'='*60}")
    
    if not benchmark_results:
        print("No results collected.")
        return

    df_res = pd.DataFrame(benchmark_results)
    # 按 Pearson R 降序排列
    df_res = df_res.sort_values(by="Pearson_R", ascending=False)
    
    # 合并历史结果（如果存在）
    if os.path.exists(summary_path):
        df_old = pd.read_csv(summary_path)
        df_res = pd.concat([df_old, df_res], ignore_index=True)

    # 去重（以 Assay 为唯一键，保留最新）
    df_res = df_res.drop_duplicates(subset="Assay", keep="last")
    print(df_res)
    df_res.to_csv("assay_benchmark_summary.csv", index=False)
    
    # 绘制对比柱状图
    try:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_res, x="Assay", y="Pearson_R", palette="viridis")
        plt.title("Model Performance Across Different Assays (Pearson R)", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig("assay_benchmark_plot.svg", format="svg")
        print("\n[Done] Summary saved to 'assay_benchmark_summary.csv' and plot to 'assay_benchmark_plot.svg'")
        print(f"All model weights are backed up in '{MODEL_SAVE_ROOT}/'")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()