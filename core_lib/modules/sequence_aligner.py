"""
sequence_aligner.py
序列对齐、受体识别及 BW 编号映射模块
(支持从外部 YAML 加载 GPCR 数据库)
"""

import os
import yaml
import numpy as np
from Bio import Align

class OffsetCalculator:
    """
    智能序列对齐器：自动识别受体类型，并支持 BW 编号转换
    """
    
    def __init__(self, db_path="data/gpcr_db.yaml"):
        """
        初始化对齐器
        
        Args:
            db_path: GPCR 数据库 YAML 文件路径
        """
        self.three_to_one = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'HSD': 'H', 
            'HSE': 'H', 'HSP': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 
            'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 
            'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        # 加载外部 YAML 数据库
        self.db = self._load_database(db_path)
        
        # 初始化比对器
        self.aligner = Align.PairwiseAligner()
        self.aligner.mode = 'global'
        self.aligner.match_score = 2
        self.aligner.mismatch_score = -1
        self.aligner.open_gap_score = -0.5
        self.aligner.extend_gap_score = -0.1

    def _load_database(self, db_path):
        """加载并预处理 GPCR YAML 数据库"""
        if not os.path.exists(db_path):
            # 尝试相对于当前脚本的路径寻找（如果从其他目录运行）
            # 或者尝试项目根目录
            possible_paths = [
                db_path,
                os.path.join(os.path.dirname(__file__), "..", db_path),
                os.path.join("..", db_path)
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    db_path = p
                    break
            else:
                raise FileNotFoundError(f"[Error] GPCR Database not found at: {db_path}")

        print(f"✓ Loading GPCR Database from: {db_path}")
        
        with open(db_path, 'r', encoding='utf-8') as f:
            raw_data = yaml.safe_load(f)
            
        processed_db = {}
        for key, val in raw_data.items():
            # 预处理：移除序列中的换行和空格
            clean_seq = "".join(val["sequence"].split())
            processed_db[key] = {
                "name": val["name"],
                "seq": clean_seq,
                "bw_map": val.get("bw_map", {})
            }
        
        return processed_db

    def _get_sim_sequence(self, u):
        """从 MDAnalysis Universe 提取序列"""
        # 尝试选取 CA 原子。如果有些残基没有 CA（不太可能），可能需要调整选择逻辑
        protein = u.select_atoms("protein and name CA")
        
        if len(protein) == 0:
            raise ValueError("No protein CA atoms found in the topology!")
            
        resnames = protein.resnames
        resids = protein.resids
        
        seq_str = ""
        valid_indices = []
        for i, res in enumerate(resnames):
            code = self.three_to_one.get(res, 'X')
            seq_str += code
            valid_indices.append(resids[i])
        
        return seq_str, valid_indices

    def identify_receptor(self, u, verbose=False):
        """
        自动识别当前 Universe 属于哪个受体
        Returns: (receptor_key, sim_sequence, sim_resids)
        """
        sim_seq, sim_resids = self._get_sim_sequence(u)
        
        best_score = -np.inf
        best_receptor = None
        
        # 遍历数据库，看谁得分高
        for receptor_key, data in self.db.items():
            score = self.aligner.score(data["seq"], sim_seq)
            # 简单的归一化分数 (分数/长度)，防止长序列占便宜
            norm_score = score / len(data["seq"])
            
            if norm_score > best_score:
                best_score = norm_score
                best_receptor = receptor_key
        
        # 简单的阈值判定，防止匹配到完全不相关的蛋白
        if best_receptor is None or best_score < 0.5: # 0.5 是一个经验阈值
             print(f"[Warning] No close match found in GPCR DB. Best score: {best_score:.2f}")
        if verbose:# 缓存结果，避免同一个 Universe 重复计算（可选）
            print(f"[Auto-Detect] Identified Receptor: {best_receptor} ({self.db[best_receptor]['name']}) Score={best_score:.2f}")
        return best_receptor, sim_seq, sim_resids

    def get_real_residue_ids(self, u, bw_list):
        """
        核心功能：输入 BW 编号列表 (如 ['6.48', '3.32'])，返回模拟中的真实残基号
        """
        # 1. 识别受体
        rec_key, sim_seq, sim_resids = self.identify_receptor(u, verbose=False)
        if rec_key is None:
            return []

        ref_data = self.db[rec_key]
        ref_seq = ref_data["seq"]
        bw_map = ref_data["bw_map"]
        
        # 2. 对齐 标准序列 vs 模拟序列
        alignments = self.aligner.align(ref_seq, sim_seq)
        best_aln = alignments[0]
        
        # 建立映射: Std_ResID -> Sim_ResID
        std_to_sim_map = {}
        
        # Biopython alignment 坐标处理
        aligned_ref_intervals = best_aln.aligned[0]
        aligned_sim_intervals = best_aln.aligned[1]
        
        for (r_start, r_end), (s_start, s_end) in zip(aligned_ref_intervals, aligned_sim_intervals):
            length = r_end - r_start
            for i in range(length):
                # r_start + i 是 ref 序列中的索引 (0-based) -> +1 变 1-based resid
                std_resid = r_start + i + 1 
                
                # s_start + i 是 sim 序列中的索引 (0-based)
                sim_idx = s_start + i
                if sim_idx < len(sim_resids):
                    real_sim_resid = sim_resids[sim_idx]
                    std_to_sim_map[std_resid] = real_sim_resid

        # 3. 查表转换
        real_ids = []
        for bw in bw_list:
            # 3.1 BW -> Std
            # 兼容处理：如果 config 里写的已经是数字，就直接当做 Std ID
            # 或者是类似 "114" 的字符串
            if isinstance(bw, int):
                std_id = bw
            elif str(bw).isdigit():
                std_id = int(bw)
            else:
                # 查 BW 表
                # 将 key 强转为 str 以防万一
                bw_str = str(bw)
                if bw_str not in bw_map:
                    print(f"[Warning] BW number '{bw}' not defined for {rec_key}. Skipping.")
                    continue
                std_id = bw_map[bw_str]
            
            # 3.2 Std -> Sim
            if std_id in std_to_sim_map:
                real_id = std_to_sim_map[std_id]
                real_ids.append(real_id)
            else:
                print(f"[Warning] Residue {bw} (Std {std_id}) is missing/gapped in simulation!")
                
        return sorted(list(set(real_ids)))