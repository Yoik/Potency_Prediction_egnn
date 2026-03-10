#!/usr/bin/env python3
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import pandas as pd
import numpy as np
import os
import glob
import sys
from tqdm import tqdm
import warnings

# ================= 配置区域 =================

# 项目根目录 (自动向上级查找)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# 输出文件路径
OUTPUT_CSV = "ligand_rmsd_trace.csv"

# [修正] 配体名称列表 (涵盖 LIG1, R5F, DRG, 7LD 等所有可能的命名)
LIGAND_SELECTION = "resname LIG LIG1 LDP R5F DRG UNK MOL 7LD"

# 预定义的效能分组 (用于画图时自动上色，方便PPT展示)
# 这一步是可选的，如果名字对不上也不会报错，只是 Type 会变成 Unknown
EFFICACY_MAP = {
    "UNC": "Full Agonist",
    "Dopa": "Full Agonist",
    "R10": "Full Agonist",
    "S10": "Full Agonist",
    "LSD": "Full Agonist",
    "PPX": "Full Agonist",
    "ROT": "Full Agonist",
    "Lisu": "Full Agonist",
    "IHCH-7084": "Full Agonist", # 假设
    "IHCH-7041": "Full Agonist", # 假设
    "ARI": "Partial Agonist",
    "BRE": "Partial Agonist",
    "S84": "Partial Agonist",
    "CAR": "Partial Agonist"
}

# ================= 核心函数 =================

def get_compound_name(folder_name):
    """从文件夹名中提取化合物名称 (假设格式 202X_D2_{Name}_*)"""
    parts = folder_name.split('_')
    for i, p in enumerate(parts):
        if p == "D2" and i + 1 < len(parts):
            return parts[i+1]
    return "Unknown"

def calculate_rmsd_for_replicate(tpr, xtc, compound_name, rep_id):
    """
    计算单个副本的 RMSD
    """
    try:
        u = mda.Universe(tpr, xtc)
        
        # 1. 选择配体原子 (排除氢原子)
        # MDAnalysis 的 resname 支持空格分隔列表
        ligand_sel = f"({LIGAND_SELECTION}) and not name H*" 
        ligand = u.select_atoms(ligand_sel)
        
        if len(ligand) == 0:
            # 调试信息：如果没找到，打印一下当前体系里到底有哪些残基，方便排查
            resnames = set(u.residues.resnames)
            print(f"    [Warn] No ligand found in {rep_id}. Available residues: {list(resnames)[:5]}...")
            return None
            
        # 2. 设置 RMSD 计算器
        # Pre-alignment: 将 Protein CA 对齐到第 0 帧
        # Calculation: 计算 Ligand Heavy Atoms 的 RMSD
        R = rms.RMSD(u, u, 
                     select="protein and name CA", 
                     groupselections=[ligand_sel],
                     ref_frame=0)
        
        # 3. 运行计算 (静默模式)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            R.run(verbose=False)
        
        # 4. 提取数据
        # R.rmsd 结果形状: [n_frames, 3+n_groups]
        # 列 0: Frame, 列 1: Time (ps), 列 2: System RMSD, 列 3: Ligand RMSD
        rmsd_array = R.results.rmsd

        time_ns = rmsd_array[:, 1] / 1000.0
        lig_rmsd = rmsd_array[:, 3]

        
        # 构建 DataFrame
        df = pd.DataFrame({
            "Time_ns": time_ns,
            "RMSD_A": lig_rmsd,
            "Compound": compound_name,
            "Replicate": rep_id,
            "Efficacy_Type": EFFICACY_MAP.get(compound_name, "Unknown")
        })
        
        return df

    except Exception as e:
        print(f"    [Error] Failed {rep_id}: {e}")
        return None

def main():
    print(f"Searching for trajectories in: {PROJECT_ROOT}")
    
    # 查找所有符合模式的化合物文件夹
    search_path = os.path.join(PROJECT_ROOT, "202*_D2_*_*")
    compound_dirs = sorted(glob.glob(search_path))
    
    if not compound_dirs:
        print("No compound directories found!")
        return

    all_data = []
    print(f"Found {len(compound_dirs)} compound directories. Starting RMSD calculation...")

    for c_dir in tqdm(compound_dirs, desc="Compounds"):
        if not os.path.isdir(c_dir): continue
        
        compound_name = get_compound_name(os.path.basename(c_dir))
        
        # 递归查找 merged.xtc
        xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
        
        for xtc in xtcs:
            rep_dir = os.path.dirname(xtc)
            # 提取副本号
            rep_name = os.path.basename(rep_dir)
            rep_id = rep_name.split('_')[-1] if '_' in rep_name else "1"
            
            # 找 TPR
            tprs = [f for f in os.listdir(rep_dir) if f.endswith(".tpr")]
            tpr_name = next((t for t in tprs if "step7_3" in t), None) # 优先 step7_3
            if not tpr_name:
                tpr_name = next((t for t in tprs if "production" in t), None) # 其次 production
            if not tpr_name and tprs:
                tpr_name = tprs[0] # 再次任意
            
            if not tpr_name:
                continue
                
            tpr = os.path.join(rep_dir, tpr_name)
            
            # 计算
            df_rep = calculate_rmsd_for_replicate(tpr, xtc, compound_name, rep_id)
            if df_rep is not None:
                all_data.append(df_rep)

    # 合并并保存
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        
        print("\nTime-averaged RMSD per replicate (Mean ± SEM) [Angstrom]:")

        try:
            # 1️⃣ 每个 replicate 的时间平均 RMSD
            rep_mean_df = (
                final_df
                .groupby(["Compound", "Replicate"])["RMSD_A"]
                .mean()
                .reset_index(name="RMSD_mean_time")
            )

            # 2️⃣ 在 replicate 之间计算 Mean ± SEM
            summary_sem = (
                rep_mean_df
                .groupby("Compound")["RMSD_mean_time"]
                .agg(
                    Mean="mean",
                    Std="std",
                    N="count"
                )
                .reset_index()
            )

            summary_sem["SEM"] = summary_sem["Std"] / np.sqrt(summary_sem["N"])

            # 3️⃣ 排序 & 打印
            summary_sem = summary_sem.sort_values("Mean")

            for _, row in summary_sem.iterrows():
                print(
                    f"{row['Compound']:>10s} : "
                    f"{row['Mean']:.2f} ± {row['SEM']:.2f} "
                    f"(n={int(row['N'])})"
                )

        except Exception as e:
            print(f"[Warn] Failed to compute Mean ± SEM: {e}")

        print("\n" + "="*50)
        print(f"Done! Data saved to: {os.path.abspath(OUTPUT_CSV)}")
        print(f"Total frames processed: {len(final_df)}")
        print("="*50)
        
        # 简单统计预览
        print("\nRMSD Summary (Mean +/- Std) [Angstrom]:")
        try:
            summary = final_df.groupby("Compound")["RMSD_A"].agg(['mean', 'std']).sort_values('mean')
            print(summary)
        except: pass
    else:
        print("No valid data collected.")

if __name__ == "__main__":
    main()