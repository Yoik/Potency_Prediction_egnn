#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import MDAnalysis as mda
import glob
from rdkit import Chem
from torch_geometric.data import Data
import warnings
import tempfile

# 引入项目模块
try:
    from src.config import init_config
    from src.featurizer import PhysicsFeaturizer
    from src.model import DeltaEGNN
    from modules.qm_loader import load_cube_and_map
    from modules.cube_parser import CubeParser
    from modules.ring_matcher import RingMatcher
    # === [新增] 引入你的对齐模块 ===
    from modules.sequence_aligner import OffsetCalculator 
except ImportError as e:
    print(f"Error: 模块导入失败: {e}")
    sys.exit(1)

# ================= 辅助函数 =================
def convert_cif_to_pdb(cif_path):
    """
    使用 RDKit 将 CIF 文件转换为 PDB 文件
    返回临时 PDB 的路径
    """
    try:
        # RDKit 读取 MMCIF
        mol = Chem.MolFromMMCIFFile(cif_path)
        if mol is None:
            return None
        
        # 创建临时 PDB 文件
        fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
        os.close(fd)
        
        Chem.MolToPDBFile(mol, tmp_path)
        print(f"  [Info] Converted CIF to temporary PDB: {tmp_path}")
        return tmp_path
    except Exception as e:
        print(f"  [Warn] CIF to PDB conversion failed: {e}")
        return None
    
def get_rdkit_mapping(ref_pdb_path, mda_ligand_atoms):
    """
    计算从 Reference PDB (QM) 到 MD Analysis Ligand 的原子索引映射。
    (保持原版逻辑，未添加额外增强)
    """
    def get_skeleton(mol):
        m = Chem.Mol(mol)
        for b in m.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        for a in m.GetAtoms():
            a.SetIsAromatic(False)
        return m

    # 1. 【元素清洗】
    if not hasattr(mda_ligand_atoms.universe.atoms, 'elements'):
        mda_ligand_atoms.universe.add_TopologyAttr('elements')
        
    valid_elems = set(['H', 'C', 'N', 'O', 'S', 'F', 'P', 'CL', 'BR', 'I', 'B', 'SI', 'FE', 'ZN', 'MG', 'CA', 'NA', 'K', 'LI'])
    
    for atom in mda_ligand_atoms:
        original_elem = atom.element.upper() if atom.element else ""
        if original_elem not in valid_elems:
            name = atom.name.upper()
            guess = "".join(filter(str.isalpha, name))
            if len(guess) > 1 and guess[:2] in valid_elems: atom.element = guess[:2]
            elif len(guess) > 0 and guess[0] in valid_elems: atom.element = guess[0]
            else: atom.element = 'C'

    # 2. 加载 Reference PDB
    ref_mol = Chem.MolFromPDBFile(ref_pdb_path, removeHs=True, sanitize=False)
    if not ref_mol: return None, None

    # 3. 将 MD Ligand 转换为 RDKit Mol
    target_mol = None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mda_ligand_atoms.write(tmp_path)
        target_mol = Chem.MolFromPDBFile(tmp_path, removeHs=True, sanitize=False)
    except Exception as e:
        print(f"  [Warn] RDKit conversion failed: {e}")
        return None, None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

    if not target_mol: return None, None

    # 4. 【核心步骤】骨架匹配
    try:
        try:
            ref_mol.UpdatePropertyCache(strict=False)
            target_mol.UpdatePropertyCache(strict=False)
        except: pass
        ref_mol = Chem.RemoveHs(ref_mol, sanitize=False)
        target_mol = Chem.RemoveHs(target_mol, sanitize=False)

        ref_skel = get_skeleton(ref_mol)
        target_skel = get_skeleton(target_mol)

        if target_skel.HasSubstructMatch(ref_skel):
            match = target_skel.GetSubstructMatch(ref_skel)
            mapping = {}
            for ref_idx, target_idx in enumerate(match):
                mapping[ref_idx] = target_idx
            return mapping, ref_mol
        else:
            return None, None
    except: return None, None

def get_dopa_global_max(root_dir, integration_radius):
    # ... (保持不变) ...
    print(f"[Init] Searching for Dopa reference in {root_dir}...")
    all_dirs = glob.glob(os.path.join(root_dir, "*"))
    for c_dir in all_dirs:
        if not os.path.isdir(c_dir): continue
        if "dopa" in os.path.basename(c_dir).lower():
            cubs = glob.glob(os.path.join(c_dir, "*.cub"))
            if cubs:
                try:
                    cp = CubeParser(cubs[0])
                    integrals = cp.get_carbon_integrals(integration_radius)
                    if len(integrals) > 0: 
                        val = np.max(integrals)
                        print(f"       Found Dopa Max Integral: {val:.4f}")
                        return val
                except: pass
    print("       [Warn] Dopa reference not found. Defaulting to 1.0.")
    return 1.0

# ================= 核心预测器 =================

class EnsemblePredictor:
    def __init__(self, models_dir, project_root):
        self.config = init_config()
        self.project_root = project_root
        
        # 初始化对齐器 (用于受体识别)
        # 假设 gpcr_db.yaml 在 data/ 下
        # === [修正] 手动处理默认路径 ===
        try:
            db_path = self.config.get_path("paths.gpcr_db")
        except Exception:
            # [修正] 优先找当前 egnn/data 目录下的文件
            # 获取当前脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            local_db = os.path.join(script_dir, "data/gpcr_db.yaml")
            root_db = os.path.join(self.project_root, "data/gpcr_db.yaml")
            
            if os.path.exists(local_db):
                db_path = local_db
            else:
                db_path = root_db            
        print(f"[Init] Initializing Sequence Aligner (DB: {db_path})...")
        self.aligner = OffsetCalculator(db_path)
        
        integration_radius = self.config.get_float("data.integration_radius")
        self.global_max = get_dopa_global_max(self.project_root, integration_radius)
        
        print("[Init] Initializing Physics Featurizer...")
        self.featurizer = PhysicsFeaturizer(self.config)
        
        print(f"[Init] Loading Ensemble Models from {models_dir}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        n_res = len(self.config.get_list("residues.obp_residues"))
        input_dim = 4 + 2 + n_res
        
        model_files = glob.glob(os.path.join(models_dir, "model_ensemble_*.pth"))
        if not model_files:
            print(f"[Error] No 'model_ensemble_*.pth' found in {models_dir}")
            sys.exit(1)
            
        self.models = []
        for mf in sorted(model_files):
            model = DeltaEGNN(self.config)
            model.input_dim = input_dim
            try:
                state_dict = torch.load(mf, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"       Loaded: {os.path.basename(mf)}")
            except Exception as e:
                print(f"       [Warn] Failed to load {mf}: {e}")
        
        print(f"       Total Models Loaded: {len(self.models)}")

    def smart_isolate_receptor(self, u_raw):
        """
        利用 OffsetCalculator 智能识别受体链
        """
        best_score = -np.inf
        best_seg = None
        
        print("  [Isolation] Scanning chains for D2 Receptor...")
        
        # 遍历所有 Segment
        for seg in u_raw.segments:
            # 提取该链的原子构建临时 Universe (为了喂给 identify_receptor)
            # 注意：MDAnalysis 的 Universe 不支持直接从 AtomGroup 创建，需要一点 workaround
            # 这里我们直接从 AtomGroup 获取序列，不创建 Universe，手动调用 aligner 的内部逻辑
            
            prot_atoms = seg.atoms.select_atoms("protein and name CA")
            if len(prot_atoms) < 50: continue # 太短忽略
            
            # 手动提取序列 (参考 sequence_aligner.py _get_sim_sequence)
            seq_str = ""
            for res in prot_atoms.resnames:
                seq_str += self.aligner.three_to_one.get(res, 'X')
            
            # 与 DB 中的 D2R 进行比对
            # 假设 DB 里 D2R 的 key 是 "D2" 或 "DRD2" (取决于你的 yaml)
            # 我们遍历 DB 找最高分
            for key, data in self.aligner.db.items():
                score = self.aligner.aligner.score(data["seq"], seq_str)
                norm_score = score / len(data["seq"])
                
                # print(f"    Segment {seg.segid} vs {key}: Score={norm_score:.2f}")
                
                if norm_score > best_score:
                    best_score = norm_score
                    best_seg = seg

        if best_seg and best_score > 0.4:
            print(f"  [Isolation] Selected Chain {best_seg.segid} (Score: {best_score:.2f})")
            return best_seg.atoms.select_atoms("protein")
        else:
            print(f"  [Error] No receptor chain found! (Best Score: {best_score:.2f})")
            return None

    def predict(self, boltz_struct, qm_cub, qm_ref_pdb):
        print(f"\n[Step 1] Loading QM Density: {os.path.basename(qm_cub)}")
        integration_radius = self.config.get_float("data.integration_radius")
        qm_data = load_cube_and_map(qm_cub, qm_ref_pdb, integration_radius)
        if not qm_data:
            print("  [Error] Failed to load QM data.")
            return None

        qm_ring_indices = []
        try:
            rm = RingMatcher(qm_data['coords'], qm_data['elements'])
            if rm.rings and 'six_ring' in rm.rings[0]:
                qm_ring_indices = rm.rings[0]['six_ring']
            else:
                qm_ring_indices = rm.ref_ring_idx
        except: pass

        print(f"[Step 2] Loading Boltz Structure: {os.path.basename(boltz_struct)}")
        
        temp_pdb_file = None
        u_raw = None

        # === [核心修改] 兼容 CIF 文件 ===
        if boltz_struct.endswith(".cif"):
            print("  [Info] Detected MMCIF file. Converting to temporary PDB using RDKit...")
            temp_pdb_file = convert_cif_to_pdb(boltz_struct)
            if temp_pdb_file:
                try:
                    u_raw = mda.Universe(temp_pdb_file)
                except Exception as e:
                    print(f"  [Error] Failed to load converted PDB: {e}")
            else:
                print("  [Error] RDKit failed to convert CIF to PDB.")
        else:
            # 尝试直接加载 (PDB等)
            try:
                u_raw = mda.Universe(boltz_struct)
            except Exception as e:
                print(f"  [Error] Failed to load structure with MDAnalysis: {e}")

        if u_raw is None:
            # 清理临时文件
            if temp_pdb_file and os.path.exists(temp_pdb_file):
                os.remove(temp_pdb_file)
            return None

        try:
            # 1. 识别配体
            ligand_sel = u_raw.select_atoms("resname LIG LIG1 LDP MOL UNK")
            if len(ligand_sel) == 0:
                prot = u_raw.select_atoms("protein")
                others = u_raw.select_atoms(f"not group prot and not resname HOH TIP3 SOL")
                if len(others) > 0:
                    lig_res_raw = sorted(others.residues, key=lambda r: len(r.atoms))[-1]
                    ligand_sel = lig_res_raw.atoms
                else:
                    print("  [Error] No ligand found.")
                    return None
            else:
                lig_res_raw = ligand_sel.residues[0]
            
            print(f"  Identified Ligand: {lig_res_raw.resname} ({len(ligand_sel)} atoms)")

            # === [调用智能筛选] ===
            receptor_sel = self.smart_isolate_receptor(u_raw)
            if receptor_sel is None: return None
                
            # 创建干净的 Universe
            u_clean = mda.Merge(receptor_sel, ligand_sel)
            lig_res_clean = u_clean.select_atoms(f"resname {lig_res_raw.resname}").residues[0]
            ligand_clean = lig_res_clean.atoms
            
            u = u_clean
            lig_res = lig_res_clean
            ligand = ligand_clean 
            # ====================

            print("[Step 3] Mapping QM Weights...")
            mapping_result = get_rdkit_mapping(qm_ref_pdb, ligand)
            if mapping_result is None or mapping_result[0] is None:
                print("  [Error] Mapping failed.")
                return None
            
            mapping, _ = mapping_result 
                
            integrals = qm_data['integrals']
            norm_integrals = integrals / self.global_max
            raw_weights = np.zeros(len(ligand))
            
            for qm_idx, md_idx in mapping.items():
                if qm_idx < len(norm_integrals):
                    raw_weights[md_idx] = norm_integrals[qm_idx]

            md_ring_indices = []
            for r_idx in qm_ring_indices:
                if r_idx in mapping: md_ring_indices.append(mapping[r_idx])

            print("[Step 4] Extracting Features...")
            data = self.featurizer.process_frame(u, lig_res, raw_weights, md_ring_indices)
            if not data: return None

            print("[Step 5] Ensemble Prediction...")
            batch = data.to(self.device)
            batch.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)
            
            scores = []
            with torch.no_grad():
                for i, model in enumerate(self.models):
                    if hasattr(model, 'forward_one'):
                        pred = model.forward_one(batch)
                    else:
                        pred, _ = model(batch, batch)
                    scores.append(pred.item())
            
            scores = np.array(scores)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            return mean_score, std_score, scores
            
        finally:
            # 确保退出前清理临时文件
            if temp_pdb_file and os.path.exists(temp_pdb_file):
                try: os.remove(temp_pdb_file)
                except: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", required=True)
    parser.add_argument("--cub", required=True)
    parser.add_argument("--qm_ref", required=True)
    parser.add_argument("--models_dir", default="data/features")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    # 确保 modules 在 pythonpath
    sys.path.append(os.path.join(script_dir, "modules"))
    
    if not os.path.exists(args.models_dir):
        pass 

    predictor = EnsemblePredictor(args.models_dir, project_root)
    result = predictor.predict(args.structure, args.cub, args.qm_ref)
    
    if result:
        mean_s, std_s, all_s = result
        print("\n" + "="*40)
        print(f"ENSEMBLE PREDICTION RESULT")
        print("="*40)
        print(f"Mean Efficacy: {mean_s:.4f}")
        print(f"Uncertainty (Std): {std_s:.4f}")
        print(f"Individual Votes: {np.round(all_s, 3)}")
        print("-" * 40)
        
        if mean_s > 0.8: print(">> Classification: Full Agonist")
        elif mean_s > 0.4: print(">> Classification: Partial Agonist")
        else: print(">> Classification: Antagonist / Weak")
    else:
        print("\n[Fail] Prediction aborted.")

if __name__ == "__main__":
    main()