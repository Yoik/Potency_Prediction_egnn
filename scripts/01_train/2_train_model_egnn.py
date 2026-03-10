#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from torch_geometric.data import Batch
from tqdm import tqdm

# =========================================================
# 寻路补丁：定位项目根目录并加载核心库
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上退两级，从 scripts/01_train 退到 egnn 根目录
egnn_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(egnn_dir)
sys.path.append(os.path.join(egnn_dir, "core_lib"))

# =========================================================
# 引入项目模块 (无需修改任何导入路径)
# =========================================================
try:
    from src.config import init_config
    from src.dataset import MolGraphDataset, PairwiseGraphDataset, get_pairwise_loader
    from src.model import DeltaEGNN
except ImportError as e:
    print(f"Error: 模块导入失败: {e}")
    sys.exit(1)

# ================= 1. 加载配置与绝对路径分流 =================
config = init_config()

# 1. 输入数据路径 (../data/features 和 ../data/labels.csv)
FEATURE_DIR = os.path.abspath(os.path.join(egnn_dir, config.get_path("paths.result_dir") or "data/features"))
LABEL_FILE = os.path.abspath(os.path.join(egnn_dir, config.get_path("paths.label_file") or "data/labels.csv"))

# 2. 模型保存路径 (checkpoints/...)
MODEL_SAVE_BASE = os.path.abspath(os.path.join(egnn_dir, config.get_path("paths.model_path") or "checkpoints/saved_models"))
if not os.path.exists(MODEL_SAVE_BASE):
    os.makedirs(MODEL_SAVE_BASE)

# 3. 结果输出路径 (outputs/results)
OUTPUT_CSV_DIR = os.path.abspath(os.path.join(egnn_dir, "outputs/results"))
if not os.path.exists(OUTPUT_CSV_DIR):
    os.makedirs(OUTPUT_CSV_DIR)

# 训练超参
LR = config.get_float("training.learning_rate", 5e-4)
EPOCHS = config.get_int("training.num_epochs", 60)
BATCH_SIZE = config.get_int("training.batch_size", 64)
ENSEMBLE_RUNS = 5  # 固定 5 折系综

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 辅助函数 ====================

def set_seed(seed):
    """确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# === [修改] 增加绝对值正则化约束 ===
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    count = 0
    loss_fn = nn.MSELoss()

    # loader 会返回 5 个元素，把 y_a 和 y_b 也接出来
    for batch_a, batch_b, delta_y, y_a, y_b in loader:
        batch_a = batch_a.to(DEVICE)
        batch_b = batch_b.to(DEVICE)
        delta_y = delta_y.to(DEVICE)
        y_a = y_a.to(DEVICE)
        y_b = y_b.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Siamese Network
        pred_a, pred_b = model(batch_a, batch_b)
        
        # 1. 核心差分 Loss
        loss_delta = loss_fn(pred_a - pred_b, delta_y.view(-1, 1))
        
        # 2. 绝对值锚定 Loss (正则化)
        # 强迫 pred_a 和 pred_b 的数值向真实的 Emax 靠拢，防止模型学出奇怪的常数偏移
        loss_abs = loss_fn(pred_a, y_a.view(-1, 1)) + loss_fn(pred_b, y_b.view(-1, 1))
        
        # 混合 Loss (0.2 是一个经验权重，表示以差分为主，绝对值为辅)
        loss = loss_delta + 0.2 * loss_abs
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * delta_y.size(0)
        count += delta_y.size(0)
        
    return total_loss / count if count > 0 else 0

# ================= [关键] 分布均值评估逻辑 =================

def compute_mean_score(model, frames_list, batch_size=64):
    """
    计算一个化合物所有帧的平均绝对分数。
    这消除了单帧随机采样的方差，反映物理稳态效能。
    """
    model.eval()
    scores = []
    
    # 批量推理以加速
    for i in range(0, len(frames_list), batch_size):
        batch_frames = frames_list[i : i+batch_size]
        batch = Batch.from_data_list(batch_frames).to(DEVICE)
        
        with torch.no_grad():
            # 技巧：DeltaEGNN 也是 Siamese，我们输入 (batch, batch)
            # 输出 (score, score)，我们只取第一个
            pred_score, _ = model(batch, batch)
            scores.extend(pred_score.cpu().numpy().flatten())
            
    return np.mean(scores)

def evaluate_loo_distribution(models, test_cmpd, train_cmpds, base_dataset):
    """
    使用【同源锚定】策略进行评估 (Top-K Nearest Anchors)
    """
    test_frames = base_dataset.data_map[test_cmpd]
    ensemble_final_preds = []
    
    # 设定只取最相似的 K 个化合物作为锚点
    TOP_K = 3 
    
    for model in models:
        score_test = compute_mean_score(model, test_frames)
        
        # 记录每个参考化合物的 (距离, 预测出的 Emax)
        anchor_candidates = []
        
        for ref_cmpd in train_cmpds:
            ref_frames = base_dataset.data_map[ref_cmpd]
            true_ref = base_dataset.label_map[ref_cmpd]
            score_ref = compute_mean_score(model, ref_frames)
            
            delta = score_test - score_ref
            pred = true_ref + delta
            
            # 计算当前参考物与测试物在模型眼里的“物理特征距离”
            dist = abs(score_test - score_ref)
            anchor_candidates.append((dist, pred, ref_cmpd))
            
        # 根据特征距离升序排序 (越近的越相似)
        anchor_candidates.sort(key=lambda x: x[0])
        
        # 仅截取 Top-K 最相似的化合物的预测值
        top_k_preds = [x[1] for x in anchor_candidates[:TOP_K]]
        
        # 取平均
        ensemble_final_preds.append(np.mean(top_k_preds))
        
    return np.mean(ensemble_final_preds), base_dataset.label_map[test_cmpd]

# ================= 主流程 =================

def main():
    print(f"Loading Dataset from {FEATURE_DIR}...")
    if not os.path.exists(LABEL_FILE):
        print(f"Error: Label file not found at {LABEL_FILE}")
        return

    base_ds = MolGraphDataset(FEATURE_DIR, LABEL_FILE)
    all_cmpds = sorted(base_ds.get_compounds())
    print(f"Compounds ({len(all_cmpds)}): {all_cmpds}")

    if len(all_cmpds) < 2: return

    final_results = []
    
    # === Phase 1: LOO-CV (验证模式) ===
    print(f"\n{'='*10} Phase 1: LOO Cross-Validation (Distribution Mean) {'='*10}")
    
    for i, test_cmpd in enumerate(tqdm(all_cmpds, desc="LOO CV")):
        print(f"\n>>> LOO Round {i+1}/{len(all_cmpds)}: Testing {test_cmpd}")
        
        # 划分数据集
        train_cmpds = [c for c in all_cmpds if c != test_cmpd]
        train_ds = PairwiseGraphDataset(base_ds, compound_list=train_cmpds, mode='train')
        train_loader = get_pairwise_loader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        current_round_models = []
        
        # 训练 5 个系综模型
        for run in tqdm(range(ENSEMBLE_RUNS), desc=" Ensemble", leave=False):
            set_seed(42 + run)
            model = DeltaEGNN(config).to(DEVICE)
            
            # 架构检查 (只在第一个跑的时候打印)
            if i == 0 and run == 0:
                print(f"  [Arch Check] Layers: {model.n_layers}")
                if hasattr(model, 'att_pool'):
                    print("  [Arch Check] WARN: Attention Pooling detected.")
                else:
                    print("  [Arch Check] Pooling: Mean Pooling (Correct).")

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            
            # 训练循环
            for epoch in tqdm(range(EPOCHS), desc=" Epochs", leave=False):
                loss = train_one_epoch(model, train_loader, optimizer)
                scheduler.step(loss)
            
            # 保存到内存列表
            current_round_models.append(model)
        
        # 评估 (使用分布均值策略)
        pred_val, true_val = evaluate_loo_distribution(current_round_models, test_cmpd, train_cmpds, base_ds)
        
        diff = pred_val - true_val
        print(f"  -> Result: True={true_val:.4f}, Pred={pred_val:.4f}, Diff={diff:+.4f}")
        final_results.append({"Compound": test_cmpd, "True": true_val, "Pred": pred_val})

    # === Phase 2: Production (生产模式) ===
    print(f"\n{'='*10} Phase 2: Finalizing Production Models (All Data) {'='*10}")
    
    full_train_ds = PairwiseGraphDataset(base_ds, compound_list=all_cmpds, mode='train')
    full_loader = get_pairwise_loader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    for run in range(ENSEMBLE_RUNS):
        print(f"Training Production Model {run+1}/{ENSEMBLE_RUNS}...")
        set_seed(42 + run)
        model = DeltaEGNN(config).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        
        for epoch in range(EPOCHS):
            train_one_epoch(model, full_loader, optimizer)
            
        # 保存权重
        save_name = f"model_ensemble_{run}.pth"
        save_path = os.path.join(MODEL_SAVE_BASE, save_name)
        torch.save(model.state_dict(), save_path)
        print(f"  [Saved] {save_path}")

    # === [新增] Phase 2.5: 绝对基准分诊断 (Absolute Score Diagnosis) ===
    print(f"\n{'='*10} Diagnosis: Absolute Scores of Production Model {'='*10}")
    # 直接使用最后一个训练好的 model
    model.eval()
    diag_results = []
    
    for cmpd in all_cmpds:
        frames = base_ds.data_map[cmpd]
        true_val = base_ds.label_map[cmpd]
        
        # 调用现有的均值计算函数获取模型的“绝对打分”
        abs_score = compute_mean_score(model, frames, batch_size=BATCH_SIZE)
        diag_results.append((cmpd, true_val, abs_score))
        
    # 打印对比表
    print(f"{'Compound':<10} | {'True Emax':<10} | {'Abs Score':<10}")
    print("-" * 35)
    for cmpd, true_val, abs_score in diag_results:
        print(f"{cmpd:<10} | {true_val:<10.4f} | {abs_score:<10.4f}")
    
    # === Phase 3: Final Report ===
    print(f"\n{'='*10} Final LOO-CV Report {'='*10}")
    df = pd.DataFrame(final_results)
    
    rmse = np.sqrt(mean_squared_error(df["True"], df["Pred"]))
    p_corr, _ = pearsonr(df["True"], df["Pred"])
    
    # 计算 Pairwise Accuracy
    n_correct = 0; n_total = 0
    vals = df.to_dict('records')
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            if np.sign(vals[i]["True"] - vals[j]["True"]) == np.sign(vals[i]["Pred"] - vals[j]["Pred"]):
                n_correct += 1
            n_total += 1
    
    acc = n_correct / n_total if n_total > 0 else 0
    
    print(df)
    print("-" * 30)
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson R: {p_corr:.4f}")
    print(f"Pairwise Accuracy: {acc:.2%}")
    
    report_path = os.path.join(OUTPUT_CSV_DIR, "loo_results_final.csv")
    df.to_csv(report_path, index=False)
    print(f">>> Final report saved to {report_path}")

if __name__ == "__main__":
    main()