#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import gc
import warnings
import numpy as np
import torch
import MDAnalysis as mda
from rdkit import Chem, RDConfig
from tqdm import tqdm
import tempfile
from rdkit.Chem import ChemicalFeatures
import yaml

# =========================================================
# 寻路补丁：定位项目根目录并加载核心库
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上退两级，从 scripts/01_train 退到 egnn 根目录
egnn_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(egnn_dir)
# 将新建立的 core_lib 临时加入环境变量，确保原有的 src 和 modules 导入不报错
sys.path.append(os.path.join(egnn_dir, "core_lib"))

# 引入项目模块
try:
    from src.config import init_config
    from src.featurizer import PhysicsFeaturizer
    from modules.qm_loader import load_cube_and_map, find_ligand, get_rdkit_mapping, get_dopa_global_max
    from modules.cube_parser import CubeParser
    from modules.ring_matcher import RingMatcher
except ImportError as e:
    print(f"Error: 模块导入失败: {e}")
    sys.exit(1)

# ================= 配置加载与绝对路径解析 =================
config = init_config()

# 动态获取 data 目录绝对路径
config_yaml_path = os.path.join(egnn_dir, "config.yaml")
with open(config_yaml_path, 'r', encoding='utf-8') as f:
    raw_config = yaml.safe_load(f)
relative_md_dir = raw_config.get('paths', {}).get('md_data_dir', '../data')
MD_DATA_DIR = os.path.abspath(os.path.join(egnn_dir, relative_md_dir))

INTEGRATION_RADIUS = config.get_float("data.integration_radius")
# 确保输出路径也是绝对路径，防止执行位置变动导致落盘错误
OUTPUT_BASE_DIR = config.get_path("paths.result_dir")
if not os.path.isabs(OUTPUT_BASE_DIR):
    OUTPUT_BASE_DIR = os.path.abspath(os.path.join(egnn_dir, OUTPUT_BASE_DIR))

def process_compound_replicates(cid, c_dir, featurizer, global_max, args):
    """
    处理单个化合物的所有副本
    """
    # 1. 准备 QM 数据
    cubs = glob.glob(os.path.join(c_dir, "*.cub"))
    if not cubs:
        print(f"  [Skip] {cid}: No .cub file found (QM density required).")
        return
    pdbs = glob.glob(os.path.join(c_dir, "*.pdb"))
    if not pdbs:
        print(f"  [Skip] {cid}: No .pdb file found (QM reference required).")
        return
    qm_ref_pdb = next((p for p in pdbs if "step7" not in p and "topol" not in p and "QC" not in p), None)
    if not qm_ref_pdb:
        print(f"  [Skip] {cid}: No valid reference .pdb found (excluding step7/topol/QC).")
        return
    if not cubs or not qm_ref_pdb: return

    print(f"Processing Compound: {cid}")
    qm_data = load_cube_and_map(cubs[0], qm_ref_pdb, INTEGRATION_RADIUS)
    if not qm_data: 
        print(f"  [Skip] {cid}: Failed to load QM data from .cub or map to reference.")
        return

    # 2. 准备 Ring Matcher
    qm_ring_indices = []
    try:
        rm = RingMatcher(qm_data['coords'], qm_data['elements'])
        if rm.rings and 'six_ring' in rm.rings[0]:
            qm_ring_indices = rm.rings[0]['six_ring']
        else:
            qm_ring_indices = rm.ref_ring_idx
    except: pass

    xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
    
    for xtc in xtcs:
        rd = os.path.dirname(xtc)
        charmm_gui_id = os.path.basename(os.path.dirname(rd))
        base_rn = os.path.basename(rd)
        unique_rn = f"{charmm_gui_id}_{base_rn}"
        
        save_dir = os.path.join(OUTPUT_BASE_DIR, cid, unique_rn)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "graph_features.pt")
        
        if os.path.exists(save_path) and not args.overwrite:
            print(f"  [Skip] Exists: {unique_rn}")
            continue

        tps = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".tpr")]
        topo = next((t for t in tps if "production" in t), tps[0] if tps else None)
        
        if not topo: continue

        try:
            u = mda.Universe(topo, xtc)
            lig_res = find_ligand(u)
            if not lig_res: continue
            
            graph_list = []
            stride = 5  # 可调整的帧间隔，减少计算量
            for ts in tqdm(u.trajectory[::stride], desc=f"  Extr. {unique_rn}", leave=False):
                try:
                    # === 调用 Featurizer (传入 Raw Weights，内部做衰减) ===
                    data = featurizer.process_frame(u, lig_res, qm_data, qm_ring_indices, global_max)
                    if data: graph_list.append(data)
                except Exception:
                    pass
            if graph_list:
                torch.save(graph_list, save_path)
                print(f"  [Saved] {len(graph_list)} frames -> {save_path}")
            del u
            gc.collect()
        except Exception as e:
            print(f"  [Error] Processing {unique_rn}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # 初始化 Featurizer
    print("Initializing Physics Featurizer...")
    featurizer = PhysicsFeaturizer(config)

    if not os.path.exists(OUTPUT_BASE_DIR): os.makedirs(OUTPUT_BASE_DIR)
    
    GLOBAL_MAX = get_dopa_global_max(MD_DATA_DIR, INTEGRATION_RADIUS)
    print(f"Global Normalization Factor: {GLOBAL_MAX:.4f}")

    all_dirs = glob.glob(os.path.join(MD_DATA_DIR, "*"))
    all_dirs.sort()

    for c_dir in all_dirs:
        if not os.path.isdir(c_dir): continue
        
        cid = os.path.basename(c_dir)
        process_compound_replicates(cid, c_dir, featurizer, GLOBAL_MAX, args)

if __name__ == "__main__":
    main()