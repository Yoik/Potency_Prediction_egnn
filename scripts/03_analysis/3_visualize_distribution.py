import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
from torch_geometric.data import Batch
from tqdm import tqdm
import argparse
import glob
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# 引入项目模块
try:
    from src.config import init_config
    from src.dataset import MolGraphDataset
    from src.model import DeltaEGNN
except ImportError as e:
    print(f"Error: 模块导入失败: {e}")
    sys.exit(1)

# 设置绘图风格
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
plt.rcParams['font.family'] = 'DejaVu Sans'

# ================= 配置区域 =================
# 在这里手动设置 X 轴范围
X_AXIS_MIN = -0.6
X_AXIS_MAX = 0.6
# ===========================================

def load_ensemble_models(result_dir, config, device):
    """加载 5 个系综模型"""
    model_paths = sorted(glob.glob(os.path.join(result_dir, "model_ensemble_*.pth")))
    models = []
    print(f"[Init] Found {len(model_paths)} models.")
    for path in model_paths:
        model = DeltaEGNN(config).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models

def predict_compound(models, frames, device, stride=1):
    """
    对单个化合物进行推理
    stride: 步长，用于统一采样密度 (例如每隔 1 帧取 1 帧)
    返回: [N_selected_frames, 5] 的矩阵
    """
    # 应用步长采样
    selected_frames = frames[::stride]
    
    batch_size = 32
    all_scores = []
    
    # 预组装 Batch
    batches = []
    for i in range(0, len(selected_frames), batch_size):
        batches.append(Batch.from_data_list(selected_frames[i : i+batch_size]).to(device))
    
    with torch.no_grad():
        for batch in batches:
            batch_outputs = []
            for model in models:
                pred, _ = model(batch, batch) # Siamese input
                batch_outputs.append(pred.cpu().numpy().flatten())
            # [Models, Batch] -> [Batch, Models]
            all_scores.append(np.vstack(batch_outputs).T)
            
    if len(all_scores) == 0:
        return np.empty((0, 5))
        
    return np.vstack(all_scores)

def get_calibration_info(loo_csv_path):
    """读取 LOO 结果用于校准和排序"""
    if not os.path.exists(loo_csv_path):
        print("Error: LOO CSV not found.")
        sys.exit(1)
    
    df = pd.read_csv(loo_csv_path)
    # 建立映射: Name -> {True: val, Pred: val}
    info_map = {}
    for _, row in df.iterrows():
        info_map[str(row['Compound']).strip()] = {
            'True': float(row['True']),
            'Target_Pred': float(row['Pred'])
        }
    return info_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=1, help="Step size for frame sampling (default: 1)")
    args = parser.parse_args()

    config = init_config()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULT_DIR = config.get_path("paths.result_dir")
    LABEL_FILE = config.get_path("paths.label_file")
    LOO_CSV = os.path.join(RESULT_DIR, "loo_results_final.csv")

    # 1. 准备工作
    models = load_ensemble_models(RESULT_DIR, config, DEVICE)
    dataset = MolGraphDataset(RESULT_DIR, LABEL_FILE)
    calib_map = get_calibration_info(LOO_CSV)
    
    all_compounds = sorted(list(dataset.data_map.keys()))
    print(f"[Process] Processing {len(all_compounds)} compounds with stride={args.stride}...")

    # 用于收集所有数据的列表 (用于生成 CSV)
    csv_rows = []
    
    # 用于收集绘图数据的列表 (Long Format for Seaborn)
    plot_data = []

    # 2. 批量推理与校准
    global_min, global_max = float('inf'), float('-inf')

    # 我们需要根据 True Efficacy 对化合物进行排序 (High -> Low)
    # 过滤掉不在 calib_map 中的化合物（如果有的话）
    valid_compounds = [c for c in all_compounds if c in calib_map]
    # 排序：True Efficacy 降序
    sorted_compounds = sorted(valid_compounds, key=lambda x: calib_map[x]['True'], reverse=True)
    
    print(f"[Order] Compounds sorted by True Efficacy (High->Low): {sorted_compounds}")

    for cmpd in tqdm(sorted_compounds, desc="Inference"):
        frames = dataset.data_map[cmpd]
        
        # A. 推理: [N, 5]
        raw_scores = predict_compound(models, frames, DEVICE, stride=args.stride)
        
        # B. 校准
        raw_mean = np.mean(raw_scores)
        target_val = calib_map[cmpd]['Target_Pred']
        offset = target_val - raw_mean
        
        # calibrated_scores = raw_scores + offset
        calibrated_scores = raw_scores
        
        # 更新全局极值 (用于统一坐标轴)
        c_min = calibrated_scores.min()
        c_max = calibrated_scores.max()
        if c_min < global_min: global_min = c_min
        if c_max > global_max: global_max = c_max

        # C. 收集数据
        # 针对每一帧
        for frame_idx, row_scores in enumerate(calibrated_scores):
            # 1. 为 CSV 收集 (每行: Frame, M1..M5)
            csv_record = {
                "Compound": cmpd,
                "Frame_ID": frame_idx,
                "True_Efficacy": calib_map[cmpd]['True'],
                "Ensemble_Mean": np.mean(row_scores) # 方便快速查看
            }
            # 添加 5 个模型的分数
            for m_i, s in enumerate(row_scores):
                csv_record[f"Model_{m_i+1}"] = s
            csv_rows.append(csv_record)
            
            # 2. 为绘图收集 (只用 Ensemble Mean 代表该帧的分布位置，或者保留所有点)
            # 为了分布图更平滑，通常使用 Ensemble Mean per Frame
            plot_data.append({
                "Compound": cmpd,
                "Efficacy": np.mean(row_scores), # 这里取帧的系综平均
                "True_Label": calib_map[cmpd]['True']
            })

    # 3. 导出 CSV
    df_out = pd.DataFrame(csv_rows)
    csv_path = os.path.join(RESULT_DIR, "all_compounds_calibrated_dist.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"[Output] Data saved to {csv_path}")

    # 4. 绘制 Ridge Plot (Joy Plot)
    print("[Plotting] Generating Ridge Plot...")
    df_plot = pd.DataFrame(plot_data)
    
    # 设定画布
    # row=Compound, hue=Compound (为了颜色), aspect=纵横比, height=单行高度
    # sharex=True (统一 X 轴)
    g = sns.FacetGrid(df_plot, row="Compound", hue="Compound", aspect=10, height=0.8, 
                      palette="coolwarm_r", # 渐变色：红(高效能) -> 蓝(低效能)。_r 表示反转
                      row_order=sorted_compounds, # 强制按效能排序
                      hue_order=sorted_compounds)

    # 绘制 KDE 密度图
    # clip_on=False 允许图表略微超出子图边界，产生"山脊"堆叠效果
    g.map(sns.kdeplot, "Efficacy", clip_on=False, shade=True, alpha=0.8, lw=1.5, bw_adjust=0.8)
    
    # 绘制均值线 (白色虚线)
    def plot_mean_line(x, **kwargs):
        plt.axvline(x.mean(), color='white', linestyle='--', linewidth=1.5, alpha=0.9)
    g.map(plot_mean_line, "Efficacy")

    # 绘制参考基线 (子图底部的一条线)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # 标签处理
    # 定义一个函数，在每个子图左侧写上化合物名字和真实效能
    def label_plot(x, color, label):
        ax = plt.gca()
        # 获取当前化合物名字 (Seaborn 会传入 label)
        # 我们从 calib_map 找真实值
        true_val = calib_map[label]['True']
        # 文字写在左边 (x=0, y=0.2 坐标系是 transAxes)
        label_text = f"{label}\n(True={true_val:.2f})"
        ax.text(0, 0.2, label_text, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=10)

    g.map(label_plot, "Efficacy")

    # === [修改点 1] 设置坐标轴 & 清理单个子图标签 ===
    g.set(xlim=(X_AXIS_MIN, X_AXIS_MAX))
    g.set_titles("")     # 移除子图标题
    g.set(yticks=[])     # 移除 Y 轴刻度
    g.set_ylabels("")    # 移除每个子图的 "Density" 标签
    g.despine(bottom=True, left=True)
    plt.subplots_adjust(hspace=-0.3)

    # === [修改点 2] 全局 Y 轴标签 ===
    # 在画布最左侧添加一个大的 "Density" 标签
    g.fig.text(0.01, 0.5, 'Density (Frequency)', va='center', rotation='vertical', fontsize=14)

    # 添加总标题和 X 轴标签
    plt.xlabel("Predicted Efficacy (Calibrated)", fontsize=14, labelpad=10)
    plt.suptitle("Dynamic Efficacy Landscape of All Compounds\n(Sorted by True Efficacy)", fontsize=16, fontweight='bold', y=0.98)

    save_plot_path = os.path.join(RESULT_DIR, "all_compounds_ridge_plot.svg")
    plt.savefig(save_plot_path, format="svg", bbox_inches='tight')
    print(f"[Output] Ridge Plot saved to {save_plot_path}")

if __name__ == "__main__":
    main()