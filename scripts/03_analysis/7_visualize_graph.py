#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import argparse
from torch_geometric.data import Data

# ================= 配置 =================
ATOM_COLORS = {
    'C': 'gray', 'N': 'blue', 'O': 'red', 'S': 'gold', 
    'F': 'green', 'CL': 'lightgreen', 'BR': 'brown', 'I': 'purple', 'P': 'orange'
}
DEFAULT_COLOR = 'pink'

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'CL', 'BR', 'I', 'P']

# 模拟 EGNN 的构图半径 (Å)
EDGE_RADIUS = 5.0 

def get_atom_type(onehot):
    """从 One-Hot 编码还原原子类型"""
    idx = np.argmax(onehot)
    if idx < len(ATOM_TYPES) and onehot[idx] > 0.5:
        return ATOM_TYPES[idx]
    return "X"

def load_compound_data(target_name, data_dir="data/features"):
    """查找并加载目标化合物的第一个 graph_features.pt"""
    # 模糊搜索
    search_pattern = os.path.join(data_dir, "*")
    found_dir = None
    for d in glob.glob(search_pattern):
        if target_name.lower() in os.path.basename(d).lower():
            found_dir = d
            break
    
    if not found_dir:
        print(f"[Error] Compound '{target_name}' not found in {data_dir}")
        return None, None

    # 找第一个 .pt 文件
    pt_files = glob.glob(os.path.join(found_dir, "*", "graph_features.pt"))
    if not pt_files:
        print(f"[Error] No .pt files found in {found_dir}")
        return None, None

    print(f"[Info] Loading {pt_files[0]}...")
    data_list = torch.load(pt_files[0], weights_only=False)
    
    # 返回第一帧用于可视化
    return data_list[0], os.path.basename(found_dir)

def visualize_graph(data, compound_name, save_prefix):
    """核心可视化逻辑"""
    x = data.x.numpy()
    pos = data.pos.numpy()
    
    # 解析节点特征
    # x dim: [Type(9), Weight(1), IsLig(1), Res(N)]
    atom_types = [get_atom_type(feat[:9]) for feat in x]
    weights = x[:, 9]
    is_ligand = x[:, 10] == 1.0
    
    # 解析全局特征
    # global_attr: [Cos1, Cos2, Offset]
    # 我们没法直接画 Cos，但可以画 Offset (如果计算了 Offset 向量的话... 
    # 等等，featurizer 只存了 Offset Norm。
    # 为了可视化 Offset 向量，我们需要重新计算 GeoCenter 和 ElecCenter)
    
    # --- 1. 数据准备 ---
    lig_indices = np.where(is_ligand)[0]
    rec_indices = np.where(~is_ligand)[0]
    
    lig_pos = pos[lig_indices]
    lig_weights = weights[lig_indices]
    
    # 重新计算中心以绘制 Offset 向量
    geo_center = np.mean(lig_pos, axis=0) # 简化的几何中心
    
    # 按照 Phase 6 逻辑，GeoCenter 应该是“有效区域”的中心
    # 这里为了展示“物理注意力”，我们用 weights > 0.01 的原子算 GeoCenter
    valid_mask = lig_weights > 0.01
    if np.sum(valid_mask) > 0:
        vis_geo_center = np.mean(lig_pos[valid_mask], axis=0)
    else:
        vis_geo_center = geo_center

    if np.sum(lig_weights) > 0.001:
        elec_center = np.average(lig_pos, axis=0, weights=lig_weights)
    else:
        elec_center = vis_geo_center

    # --- 2. 绘图 ---
    fig = plt.figure(figsize=(18, 6))
    titles = ["XY Plane (Top)", "XZ Plane (Side)", "3D Perspective"]
    projections = [(0, 1), (0, 2), '3d'] # indices for X, Y, Z
    
    axes = []
    axes.append(fig.add_subplot(131))
    axes.append(fig.add_subplot(132))
    axes.append(fig.add_subplot(133, projection='3d'))

    print(f"\n[Visualizing] {compound_name}")
    print(f"  Ligand Atoms: {len(lig_indices)}")
    print(f"  Receptor Atoms: {len(rec_indices)}")
    print(f"  Max Weight: {np.max(lig_weights):.4f}")
    print(f"  Min Weight: {np.min(lig_weights):.4f}")
    print(f"  Offset Dist: {np.linalg.norm(elec_center - vis_geo_center):.4f} A")

    # 循环绘制三个视角
    for i, ax in enumerate(axes):
        is_3d = (i == 2)
        
        # A. 绘制边 (Edges) - 模拟 EGNN 视野
        # 只画 Ligand 内部以及 Ligand-Receptor 之间的边，减少杂乱
        # 简单暴力法：计算距离矩阵 (仅用于演示，大体系慎用)
        from scipy.spatial.distance import cdist
        dists = cdist(pos, pos)
        # 找到 < EDGE_RADIUS 的对，且 i < j
        src, dst = np.where((dists < EDGE_RADIUS) & (np.triu(dists) > 0))
        
        for s, d in zip(src, dst):
            # 过滤：不画受体内部的边，太乱
            if not is_ligand[s] and not is_ligand[d]: continue
            
            # 线条透明度随权重变化 (模拟信息流)
            w_edge = (weights[s] + weights[d]) / 2.0
            alpha = 0.1 + 0.4 * w_edge # 基础 0.1，最强 0.5
            
            if is_3d:
                ax.plot([pos[s,0], pos[d,0]], [pos[s,1], pos[d,1]], [pos[s,2], pos[d,2]], 
                        c='gray', alpha=alpha, linewidth=0.5)
            else:
                x_idx, y_idx = projections[i]
                ax.plot([pos[s,x_idx], pos[d,x_idx]], [pos[s,y_idx], pos[d,y_idx]], 
                        c='gray', alpha=alpha, linewidth=0.5)

        # B. 绘制节点 (Nodes)
        for idx in range(len(pos)):
            p = pos[idx]
            w = weights[idx]
            atom = atom_types[idx]
            is_lig = is_ligand[idx]
            
            color = ATOM_COLORS.get(atom, DEFAULT_COLOR)
            
            # 大小逻辑：受体固定小点，配体随权重变化
            # Base Size: 20
            # Weight Influence: w * 200
            if is_lig:
                size = 20 + w * 300
                marker = 'o'
                edgecolor = 'black'
                zorder = 10
            else:
                size = 30 # 受体节点稍微大一点以便看清位置
                marker = '^' # 三角形表示受体
                edgecolor = 'none'
                zorder = 5
                color = 'lightgray' # 受体统一灰色，避免干扰
            
            if is_3d:
                ax.scatter(p[0], p[1], p[2], s=size, c=color, marker=marker, edgecolors=edgecolor, alpha=0.8, zorder=zorder)
            else:
                x_idx, y_idx = projections[i]
                ax.scatter(p[x_idx], p[y_idx], s=size, c=color, marker=marker, edgecolors=edgecolor, alpha=0.8, zorder=zorder)
                
        # C. 绘制 Offset 向量 (Geo -> Elec)
        if is_3d:
            ax.plot([vis_geo_center[0], elec_center[0]], 
                    [vis_geo_center[1], elec_center[1]], 
                    [vis_geo_center[2], elec_center[2]], 
                    c='magenta', linestyle='--', linewidth=2, label='Offset')
            # 画中心点
            ax.scatter(vis_geo_center[0], vis_geo_center[1], vis_geo_center[2], c='black', marker='x', s=100, label='GeoCenter(Valid)')
            ax.scatter(elec_center[0], elec_center[1], elec_center[2], c='magenta', marker='*', s=200, label='ElecCenter')
        else:
            x_idx, y_idx = projections[i]
            ax.plot([vis_geo_center[x_idx], elec_center[x_idx]], 
                    [vis_geo_center[y_idx], elec_center[y_idx]], 
                    c='magenta', linestyle='--', linewidth=2)
            ax.scatter(vis_geo_center[x_idx], vis_geo_center[y_idx], c='black', marker='x', s=100)
            ax.scatter(elec_center[x_idx], elec_center[y_idx], c='magenta', marker='*', s=200)

        # 设置标题和轴
        ax.set_title(titles[i])
        if not is_3d:
            ax.set_xlabel(["X", "Y", "Z"][projections[i][0]] + " (A)")
            ax.set_ylabel(["X", "Y", "Z"][projections[i][1]] + " (A)")
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

    plt.suptitle(f"White-Box Visualization: {compound_name}\nNode Size = Electronic Weight (Phase 6 Attention)", fontsize=16)
    
    # 保存
    save_path = f"{save_prefix}_{compound_name}_vis.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Saved] Visualization saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Phase 7.0: Visualize Graph Features (.pt)")
    parser.add_argument("--target", type=str, default="S84", help="Compound name to visualize (e.g., S84, BRE, UNC)")
    args = parser.parse_args()

    data, full_name = load_compound_data(args.target)
    if data:
        visualize_graph(data, args.target, "debug_phase7")

if __name__ == "__main__":
    main()