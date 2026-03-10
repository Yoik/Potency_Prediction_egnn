import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# ================= 配置 =================
DATA_DIR = "data/features"
# 挑选几个代表性分子进行详细对比
TARGET_COMPOUNDS = ["Dopa", "ARI", "LSD", "BRE"] 
# 颜色映射
CMAP = "viridis"

def load_data():
    """加载所有 .pt 数据并整理为 DataFrame"""
    print(f"Loading data from {DATA_DIR}...")
    search_path = os.path.join(DATA_DIR, "*", "*", "graph_features.pt")
    files = glob.glob(search_path)
    
    global_features = []
    atom_data_samples = {} # 存储少量样本用于3D绘图
    
    for f in tqdm(files):
        # 解析化合物名称
        parent = os.path.dirname(os.path.dirname(f))
        folder_name = os.path.basename(parent)
        
        # 简单名称匹配
        cmpd_name = "Unknown"
        for target in TARGET_COMPOUNDS:
            if target in folder_name:
                cmpd_name = target
                break
        if cmpd_name == "Unknown": continue # 只分析关注的分子，避免图太乱
            
        try:
            # 加载数据
            graph_list = torch.load(f, weights_only=False)
            
            # 1. 收集全局特征 (Global Attributes)
            for g in graph_list:
                # g.global_attr: [1, 3] -> [Cos648, Cos652, Offset]
                glob_attr = g.global_attr.numpy().flatten()
                global_features.append({
                    "Compound": cmpd_name,
                    "Cos_Angle_W648": glob_attr[0],
                    "Cos_Angle_F652": glob_attr[1],
                    "Elec_Offset": glob_attr[2]
                })
            
            # 2. 收集一帧样本用于 3D 绘图 (只存第一帧)
            if cmpd_name not in atom_data_samples and len(graph_list) > 0:
                atom_data_samples[cmpd_name] = graph_list[0]
                
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return pd.DataFrame(global_features), atom_data_samples

def plot_3d_comparison(atom_samples):
    """绘制 Dopa vs ARI 的 3D 结构对比"""
    print("Plotting 3D structures...")
    
    compounds_to_plot = [c for c in ["Dopa", "ARI"] if c in atom_samples]
    if not compounds_to_plot: return

    fig = plt.figure(figsize=(16, 8))
    
    for i, name in enumerate(compounds_to_plot):
        data = atom_samples[name]
        pos = data.pos.numpy()
        x = data.x.numpy()
        
        # 解析特征
        # x dim: 0-8(Type), 9(Weight), 10(IsLigand), 11-24(ResID)
        is_ligand = x[:, 10] == 1
        weights = x[:, 9]
        
        # 分离坐标
        lig_pos = pos[is_ligand]
        rec_pos = pos[~is_ligand]
        lig_w = weights[is_ligand]
        
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        
        # 1. 画受体锚点 (灰色骨架)
        ax.scatter(rec_pos[:, 0], rec_pos[:, 1], rec_pos[:, 2], 
                   c='lightgrey', alpha=0.3, s=20, label='Receptor Anchors')
        
        # 2. 画配体原子 (颜色深浅代表电子权重)
        # 权重越大，点越大，颜色越红
        p = ax.scatter(lig_pos[:, 0], lig_pos[:, 1], lig_pos[:, 2], 
                       c=lig_w, cmap='Reds', s=lig_w*100 + 20, 
                       edgecolors='black', linewidth=0.5, label='Ligand (Elec Weight)')
        
        # 3. 辅助线：几何中心到受体中心
        center = np.mean(lig_pos, axis=0)
        ax.scatter(center[0], center[1], center[2], c='blue', marker='x', s=100, label='Geo Center')
        
        ax.set_title(f"{name} (Aligned State)", fontsize=16, fontweight='bold')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        
        # 统一视角以便对比
        ax.view_init(elev=20, azim=45)
        
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.savefig("vis_3d_structure_compare.png", dpi=300)
    print("Saved 3D comparison to vis_3d_structure_compare.png")

def plot_global_distributions(df):
    """绘制全局物理特征的分布图"""
    print("Plotting statistical distributions...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. W6.48 夹角 (垂直 vs 平行)
    sns.kdeplot(data=df, x="Cos_Angle_W648", hue="Compound", fill=True, alpha=0.3, ax=axes[0], palette="tab10")
    axes[0].set_title("A. Orientation vs W6.48\n(0=Parallel, 1=Perpendicular)", fontsize=14)
    axes[0].set_xlabel("Cos(Theta)")
    axes[0].set_xlim(0, 1)
    
    # 2. F6.52 夹角
    sns.kdeplot(data=df, x="Cos_Angle_F652", hue="Compound", fill=True, alpha=0.3, ax=axes[1], palette="tab10")
    axes[1].set_title("B. Orientation vs F6.52", fontsize=14)
    axes[1].set_xlabel("Cos(Theta)")
    axes[1].set_xlim(0, 1)
    
    # 3. 电子重心偏移量
    sns.violinplot(data=df, x="Compound", y="Elec_Offset", ax=axes[2], palette="tab10")
    axes[2].set_title("C. Electronic Cloud Offset Magnitude", fontsize=14)
    axes[2].set_ylabel("Offset (Å)")
    
    plt.tight_layout()
    plt.savefig("vis_feature_statistics.png", dpi=300)
    print("Saved statistics to vis_feature_statistics.png")

def main():
    df, samples = load_data()
    if df.empty:
        print("No data found!")
        return
        
    print(f"Loaded {len(df)} frames for analysis.")
    
    plot_3d_comparison(samples)
    plot_global_distributions(df)

if __name__ == "__main__":
    main()