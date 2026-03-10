import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
import subprocess
import re
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch_geometric.data import Batch
from matplotlib.lines import Line2D

# 引入项目模块
try:
    from src.config import init_config
    from src.featurizer import PhysicsFeaturizer
    from src.model import DeltaEGNN
    import importlib.util
    # 复用之前的提取逻辑
    spec = importlib.util.spec_from_file_location("extractor", "1_extract_features_egnn.py")
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
except ImportError as e:
    print(f"Error: 模块导入失败: {e}")
    sys.exit(1)

# ================= 用户配置区域 (已修正路径) =================
PREDICT_ROOT = "predict"          # 预测数据的根目录
GLOBAL_MAX_DENSITY = 56.7638      # 用户提供的归一化因子
BEST_MODEL_DIR = "data/features" # 默认使用最好的模型

# [关键修改]：使用实际的文件夹名称，而不是简写 "Dopa"
REFERENCE_COMPOUND = "20251115_D2_Dopa_cryoEM_rebuild" 

REFERENCE_TRUE_VAL = 1.029        # Dopa 在 miniGo12 中的真实值
# ========================================================

# 绘图风格
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
plt.rcParams['font.family'] = 'DejaVu Sans'

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def auto_merge_trajectories():
    """遍历 predict 目录，自动合并 step7_*.xtc"""
    print(f"\n{'='*10} Step 1: Merging Trajectories {'='*10}")
    
    for root, dirs, files in os.walk(PREDICT_ROOT):
        if "step7_1.xtc" in files:
            if os.path.exists(os.path.join(root, "merged.xtc")):
                continue
            
            print(f"  Merging in: {root}")
            xtc_files = glob.glob(os.path.join(root, "step7_*.xtc"))
            xtc_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
            local_files = [os.path.basename(f) for f in xtc_files]
            
            cmd = ["gmx", "trjcat", "-f"] + local_files + ["-o", "merged.xtc", "-settime"]
            inputs = ["0"] + ["c"] * (len(local_files) - 1)
            input_str = "\n".join(inputs) + "\n"
            
            try:
                process = subprocess.Popen(
                    cmd, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                stdout, stderr = process.communicate(input=input_str)
                if process.returncode != 0:
                    print(f"  [Error] Merge failed in {root}:\n{stderr}")
            except FileNotFoundError:
                print("  [Error] 'gmx' command not found!")
                return

def extract_features_for_predict(config):
    """提取特征"""
    print(f"\n{'='*10} Step 2: Extracting Features {'='*10}")
    
    featurizer = PhysicsFeaturizer(config)
    # 获取 predict 下的所有化合物文件夹
    compounds = [d for d in os.listdir(PREDICT_ROOT) if os.path.isdir(os.path.join(PREDICT_ROOT, d))]
    
    class MockArgs:
        overwrite = False
    args = MockArgs()

    for cid in compounds:
        c_dir = os.path.join(PREDICT_ROOT, cid)
        print(f"Processing {cid} in {c_dir}...")
        try:
            extractor.process_compound_replicates(cid, c_dir, featurizer, GLOBAL_MAX_DENSITY, args)
        except Exception as e:
            print(f"  [Error] Feature extraction failed for {cid}: {e}")

def load_models(device, config):
    """加载模型"""
    model_paths = glob.glob(os.path.join(BEST_MODEL_DIR, "model_ensemble_*.pth"))
    if not model_paths:
        print(f"Error: No models found in {BEST_MODEL_DIR}")
        sys.exit(1)
    
    models = []
    for path in model_paths:
        model = DeltaEGNN(config).to(device)
        # 显式设置 weights_only=False 以避免 PyTorch 安全报错
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model.eval()
        models.append(model)
    return models

def get_calibration_offset(models, device, result_dir):
    """计算校准偏移量"""
    dopa_dir = os.path.join(result_dir, REFERENCE_COMPOUND) 
    if not os.path.exists(dopa_dir):
        print(f"[Warn] Reference compound '{REFERENCE_COMPOUND}' not found in {dopa_dir}.")
        print(f"       Expected path: {dopa_dir}")
        return 0.0, None

    pt_files = glob.glob(os.path.join(dopa_dir, "*", "graph_features.pt"))
    if not pt_files:
        print(f"[Warn] No .pt files found in {dopa_dir}.")
        return 0.0, None

    dopa_frames = []
    print(f"  Loading reference frames from: {dopa_dir}")
    for pt in pt_files:
        # 显式设置 weights_only=False
        dopa_frames.extend(torch.load(pt, weights_only=False))
    
    scores = []
    batch_size = 32
    # 为了快速计算，如果 Dopa 帧数太多，可以只取前 500 帧
    if len(dopa_frames) > 500:
        dopa_frames = dopa_frames[:500]

    batches = [Batch.from_data_list(dopa_frames[i:i+batch_size]).to(device) for i in range(0, len(dopa_frames), batch_size)]
    
    with torch.no_grad():
        for batch in batches:
            for model in models:
                pred, _ = model(batch, batch)
                scores.extend(pred.cpu().numpy().flatten())
    
    raw_dopa_mean = np.mean(scores)
    offset = REFERENCE_TRUE_VAL - raw_dopa_mean
    print(f"  [Calibration] {REFERENCE_COMPOUND} Raw Mean: {raw_dopa_mean:.4f}")
    print(f"  [Calibration] Target True Value: {REFERENCE_TRUE_VAL:.4f}")
    print(f"  [Calibration] Calculated Offset: {offset:.4f}")
    return offset, raw_dopa_mean

def predict_and_plot(models, offset, device, result_dir):
    """对新化合物进行预测并绘图"""
    print(f"\n{'='*10} Step 3: Prediction & Visualization {'='*10}")
    
    predict_compounds = [d for d in os.listdir(PREDICT_ROOT) if os.path.isdir(os.path.join(PREDICT_ROOT, d))]
    results = []

    for cid in predict_compounds:
        feat_dir = os.path.join(result_dir, cid)
        if not os.path.exists(feat_dir):
            continue
            
        pt_files = glob.glob(os.path.join(feat_dir, "*", "graph_features.pt"))
        if not pt_files: continue
        
        frames = []
        for pt in pt_files:
            frames.extend(torch.load(pt, weights_only=False))
            
        print(f"  Predicting {cid} ({len(frames)} frames)...")
        
        raw_preds = []
        batch_size = 32
        batches = [Batch.from_data_list(frames[i:i+batch_size]).to(device) for i in range(0, len(frames), batch_size)]
        
        with torch.no_grad():
            for batch in batches:
                batch_preds = []
                for model in models:
                    p, _ = model(batch, batch)
                    batch_preds.append(p.cpu().numpy().flatten())
                # 取系综平均
                raw_preds.extend(np.mean(batch_preds, axis=0))
        
        final_preds = np.array(raw_preds) + offset
        mean_efficacy = np.mean(final_preds)
        
        results.append({
            "Compound": cid,
            "Pred_Efficacy": mean_efficacy,
            "Scores": final_preds
        })
        print(f"  -> {cid}: Predicted Efficacy = {mean_efficacy:.4f}")

    if not results: return

    # 绘图逻辑
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(results))
    
    for i, res in enumerate(results):
        # 绘制分布
        sns.kdeplot(res['Scores'], fill=True, label=f"{res['Compound']} (Mean={res['Pred_Efficacy']:.2f})", 
                    color=colors[i], alpha=0.5, linewidth=2)
        plt.axvline(res['Pred_Efficacy'], color=colors[i], linestyle='--')

    if offset != 0.0:
        plt.axvline(REFERENCE_TRUE_VAL, color='gray', linestyle=':', linewidth=2, label=f"Ref: Dopa ({REFERENCE_TRUE_VAL})")

    plt.title(f"Predicted Efficacy Distribution (Calibrated to Dopa)", fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Efficacy (miniGo12 Scale)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    out_path = os.path.join(PREDICT_ROOT, "prediction_results.png")
    plt.savefig(out_path, dpi=300)
    print(f"\n[Done] Plot saved to {out_path}")
    
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k != 'Scores'} for r in results])
    df_res.to_csv(os.path.join(PREDICT_ROOT, "prediction_summary.csv"), index=False)

def main():
    config = init_config()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULT_DIR = config.get_path("paths.result_dir") 

    auto_merge_trajectories()
    extract_features_for_predict(config)
    
    models = load_models(DEVICE, config)
    offset, _ = get_calibration_offset(models, DEVICE, RESULT_DIR)
    predict_and_plot(models, offset, DEVICE, RESULT_DIR)

if __name__ == "__main__":
    main()