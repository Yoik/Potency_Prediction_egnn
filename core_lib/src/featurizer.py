import torch
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.lib.distances import distance_array
from torch_geometric.data import Data

from modules.geometry import calculate_plane_normal
from modules.sequence_aligner import OffsetCalculator
from modules.qm_loader import get_rdkit_mapping

class PhysicsFeaturizer:
    def __init__(self, config):
        self.config = config
        
        # 1. 加载关键残基列表
        # Phe Group: 负责疏水堆积 (现在明确为 6.51, 6.52)
        self.phe_bw = config.get_list("residues.phe_residues") 
        # OBP Group: 负责全原子特征编码
        self.obp_bw = config.get_list("residues.obp_residues")
        
        # [新增] Polar Anchors: 负责极性锚定 (TM5侧壁 + 6.55)
        # 如果 config 里没有，给一个合理的默认值，防止报错
        self.polar_bw = config.get_list("residues.polar_anchors")
        if not self.polar_bw: 
            self.polar_bw = ["5.42", "5.46", "6.55"]

        self.polar_elements = {'N', 'O', 'S', 'F', 'CL', 'BR', 'P'}
        self.atom_types = ['C', 'N', 'O', 'S', 'F', 'CL', 'BR', 'I', 'P']
        
        # 2. 加载参考系
        ref_path = config.get_path("paths.reference_pdb")
        print(f"[Featurizer] Loading Reference PDB: {ref_path}")
        self.ref_u = mda.Universe(ref_path)
        # 定义原子物理化学性质 (VdW半径, 电负性, 归一化质量, 氢键能力粗略值)
        # 氢键能力: 1=Acceptor, 2=Donor/Acceptor, 0=None (简化版)
        self.atom_props = {
            'H':  [1.20, 2.20, 0.01, 0.0],
            'C':  [1.70, 2.55, 0.12, 0.0],
            'N':  [1.55, 3.04, 0.14, 2.0], # 强极性
            'O':  [1.52, 3.44, 0.16, 2.0], # 强极性
            'S':  [1.80, 2.58, 0.32, 1.0], # 中等极性 (关键！它和 C 的电负性很像)
            'F':  [1.47, 3.98, 0.19, 1.0],
            'CL': [1.75, 3.16, 0.35, 1.0], # Cl 和 S 的半径接近
            'BR': [1.85, 2.96, 0.80, 0.0],
            'I':  [1.98, 2.66, 1.27, 0.0],
            'P':  [1.80, 2.19, 0.31, 0.0]
        }
        # 默认值 (用于未知原子)
        self.default_prop = [1.70, 2.50, 0.20, 0.0]

        self.aligner = OffsetCalculator()
        # 预计算 Reference 的对齐锚点 (用于刚性对齐)
        self.ref_anchor_ids = self.aligner.get_real_residue_ids(self.ref_u, self.obp_bw)

    def _get_atom_feature(self, atom, electronic_weight=0.0, is_ligand=False, resid_onehot=None, clash_score=0.0, extra_feats=None):
            # 1. [修改] Physicochemical Properties (4 dims) 替代 Type One-Hot (9 dims)
            elem = atom.element.upper()
            
            # 查表，查不到就用默认
            props = self.atom_props.get(elem, self.default_prop)
            
            # 归一化处理 (简单缩放，使其落在 0-1 之间便于训练)
            # Radius / 2.0, Electronegativity / 4.0, Mass, HB
            norm_props = [
                props[0] / 2.0, 
                props[1] / 4.0, 
                props[2],       
                props[3] / 2.0
            ]
            
            # 2. Physics Props (QM Weight + Is_Ligand) (2 dims)
            phys_feat = [float(electronic_weight), 1.0 if is_ligand else 0.0]
            
            # 3. Clash Score (1 dim) 
            clash_feat = [clash_score]

            # 4. Residue Identity (N dims)、
            if resid_onehot is None:
                res_feat = [0.0] * len(self.obp_bw)
            else:
                res_feat = resid_onehot
            
            if extra_feats is None:
                extra_feats = [0.0, 0.0, 0.0, 0.0]
                
            return norm_props + phys_feat + clash_feat + res_feat + extra_feats
    
    def process_frame(self, u, lig_res, qm_data, qm_ring_indices, global_max=1.0):
        # === 建立轨迹级缓存，避免每帧重复计算 RDKit 匹配和序列比对 ===
        cache_key = id(u)
        if not hasattr(self, '_trajectory_cache'):
            self._trajectory_cache = {}
            
        if cache_key not in self._trajectory_cache:
            mapping, _ = get_rdkit_mapping(qm_data['pdb_path'], lig_res.atoms)
            real_obp_ids = self.aligner.get_real_residue_ids(u, self.obp_bw)
            real_phe_ids = self.aligner.get_real_residue_ids(u, self.phe_bw)
            real_polar_ids = self.aligner.get_real_residue_ids(u, self.polar_bw)
            self._trajectory_cache[cache_key] = (mapping, real_obp_ids, real_phe_ids, real_polar_ids)
        else:
            mapping, real_obp_ids, real_phe_ids, real_polar_ids = self._trajectory_cache[cache_key]

        if mapping is None: 
            return None
                    
        integrals = qm_data['integrals']
        norm_integrals = integrals / global_max
        raw_weights = np.zeros(len(lig_res.atoms))
        
        for qm_idx, md_idx in mapping.items():
            if qm_idx < len(norm_integrals):
                raw_weights[md_idx] = norm_integrals[qm_idx]
                
        md_ring_indices = []
        for r_idx in qm_ring_indices:
            if r_idx in mapping: md_ring_indices.append(mapping[r_idx])

        # === [内部处理 2] 静态化学属性与水分子提取 ===
        lig_static_chem_feats = {}
        lig_heavy_atoms = lig_res.atoms.select_atoms("not name H*")
        for atom in lig_heavy_atoms:
            bonded_atoms = atom.bonded_atoms
            num_hs = sum(1 for neighbor in bonded_atoms if neighbor.name.startswith('H') or neighbor.element == 'H')
            elem = atom.element.upper() if atom.element else "".join(filter(str.isalpha, atom.name.upper()))[:1]
            
            is_hbd = 1.0 if (elem in ['N', 'O', 'S'] and num_hs > 0) else 0.0
            is_hba = 1.0 if (elem in ['N', 'O', 'F']) else 0.0
            lig_static_chem_feats[atom.index] = [float(num_hs), is_hbd, is_hba]

        water_atoms = u.select_atoms("resname TIP3P SOL WAT and name OW")

        if not real_obp_ids: return None

        # 2. 刚性对齐 (Align to Reference)
        mobile_sel = f"resid {' '.join(map(str, real_obp_ids))} and name CA"
        ref_sel = f"resid {' '.join(map(str, self.ref_anchor_ids))} and name CA"
        align.alignto(u, self.ref_u, select=dict(mobile=mobile_sel, reference=ref_sel))

        # 3. 准备原子集合 (为了计算 Clash，提前定义 heavy atoms)
        lig_heavy = lig_res.atoms.select_atoms("not name H*")
        rec_heavy = u.select_atoms(f"resid {' '.join(map(str, real_obp_ids))} and not name H*")

        # === [核心升级 1] 化学感知衰减 (Chemical-Aware Attention) ===
        decayed_weights = np.array(raw_weights, copy=True) # 形状是 [All_Lig_Atoms]
        lig_pos_all = lig_res.atoms.positions
        
        max_decay_factors = np.zeros(len(lig_pos_all))
        
        # A. 疏水核心 (Phe): 只看距离，不做化学过滤
        if real_phe_ids:
            phe_atoms = u.select_atoms(f"resid {' '.join(map(str, real_phe_ids))} and not name H* and not name N C O CA")
            if len(phe_atoms) > 0:
                center = phe_atoms.positions.mean(axis=0)
                dists = np.linalg.norm(lig_pos_all - center, axis=1)
                factors = np.exp(- (dists / 5.0) ** 2)
                max_decay_factors = np.maximum(max_decay_factors, factors)

        # B. 极性锚点 (TM5): 增加“极性检查”
        if real_polar_ids:
            polar_atoms = u.select_atoms(f"resid {' '.join(map(str, real_polar_ids))} and not name H* and not name N C O CA")
            if len(polar_atoms) > 0:
                center = polar_atoms.positions.mean(axis=0)
                dists = np.linalg.norm(lig_pos_all - center, axis=1)
                spatial_factors = np.exp(- (dists / 5.0) ** 2)
                
                # --- [新增] 化学过滤逻辑 ---
                chem_factors = []
                for atom in lig_res.atoms:
                    elem = atom.element.upper()
                    # 如果是极性原子 (N/O/F)，保留权重 (1.0)；如果是非极性 (C)，抑制权重 (0.2)
                    chem_factor = 1.0 if elem in self.polar_elements else 0.2
                    chem_factors.append(chem_factor)
                
                chem_factors = np.array(chem_factors)
                # 最终权重 = 空间距离 * 化学属性
                final_factors = spatial_factors * chem_factors
                max_decay_factors = np.maximum(max_decay_factors, final_factors)

        decayed_weights = raw_weights * max_decay_factors

        # 计算每个配体重原子到最近受体原子的距离
        d_matrix = distance_array(lig_heavy.positions, rec_heavy.positions) # Shape [N, M]
        min_dists = d_matrix.min(axis=1) # 每个配体原子到受体的最近距离 [N]
        
        # 定义碰撞分数：距离越近(<2.5A)，分数越高 (指数级激增)
        # 用 steep gaussian: 如果 d=2.0 -> score高, d=4.0 -> score接近0
        clash_scores = np.exp(- (min_dists / 2.0) ** 2) 

        # === [新增] 动态水配位数计算 ===
        water_counts = np.zeros(len(lig_heavy))
        if water_atoms is not None and len(water_atoms) > 0:
            # 计算配体重原子到所有水分子的距离
            w_dists = distance_array(lig_heavy.positions, water_atoms.positions)
            # 统计 3.5 A 内的水分子数量
            water_counts = np.sum(w_dists < 3.5, axis=1)

        # 4. 构建节点特征
        node_feats = []
        # lig_heavy 已经在上面定义过了，这里直接用
        
        # 4.1 配体 (应用衰减后的权重 + 传入 Clash Score)
        lig_start_idx = lig_res.atoms[0].index
        for i, atom in enumerate(lig_heavy):
            local_idx = atom.index - lig_start_idx
            w = decayed_weights[local_idx] if local_idx < len(decayed_weights) else 0.0
            c_score = clash_scores[i]
            
            # === [新增] 获取隐式 H/HBD/HBA 和动态 Water ===
            static_feat = lig_static_chem_feats.get(atom.index, [0.0, 0.0, 0.0]) if lig_static_chem_feats else [0.0, 0.0, 0.0]
            w_count = float(water_counts[i])
            extra_feats = static_feat + [w_count] # 4维

            # [修改] 传入 clash_score
            node_feats.append(self._get_atom_feature(atom, w, is_ligand=True, clash_score=c_score, extra_feats=extra_feats))
            
        # 4.2 受体
        resid_to_config_idx = {rid: i for i, rid in enumerate(real_obp_ids)}
        for atom in rec_heavy:
            rid = atom.residue.resid
            cfg_idx = resid_to_config_idx.get(rid)
            onehot = [0.0] * len(self.obp_bw)
            if cfg_idx is not None: onehot[cfg_idx] = 1.0

            extra_feats = [0.0, 0.0, 0.0, 0.0] # 受体原子没有隐式 H/HBD/HBA 和水配位数
            node_feats.append(self._get_atom_feature(atom, 0.0, is_ligand=False, resid_onehot=onehot, clash_score=0.0, extra_feats=extra_feats))

        x = torch.tensor(node_feats, dtype=torch.float32)        

        # 5. 全局几何特征
        
        # 5.1 配体法向量
        lig_normal = np.array([0.0, 0.0, 1.0])
        if md_ring_indices:
            # 只有当环依然被 Attention 关注时 (权重>0.001)，才计算它的方向
            ring_weights = decayed_weights[md_ring_indices]
            if np.mean(ring_weights) > 0.001:
                try:
                    ring_pos = lig_res.atoms[md_ring_indices].positions
                    if len(ring_pos) >= 3:
                        lig_normal = calculate_plane_normal(ring_pos)
                except: pass

        # 5.2 受体法向量 (计算 6.51/6.52 的平面)
        angles = []
        phe_residues_atoms = []
        if real_phe_ids:
            for pid in real_phe_ids:
                phe_residues_atoms.append(u.select_atoms(f"resid {pid} and not name H* and not name N C O CA"))
        
        for res_ag in phe_residues_atoms:
            res_normal = np.array([0.0, 0.0, 1.0])
            try:
                if len(res_ag) >= 3: res_normal = calculate_plane_normal(res_ag.positions)
            except: pass
            angles.append(np.abs(np.dot(lig_normal, res_normal)))
        
        while len(angles) < 2: angles.append(0.0)

        # 5.3 Offset (核心：基于衰减后的权重)
        if np.sum(decayed_weights) > 0.01:
            valid_indices = np.where(decayed_weights > 0.01)[0]
            if len(valid_indices) > 0:
                # 几何中心：只计算"有效区域"的几何中心
                geo_center = lig_res.atoms.positions[valid_indices].mean(axis=0)
            else:
                geo_center = lig_res.atoms.positions.mean(axis=0)
            
            # 电子中心：加权平均
            elec_center = np.average(lig_res.atoms.positions, axis=0, weights=decayed_weights)
            offset_norm = np.linalg.norm(elec_center - geo_center)
        else:
            offset_norm = 0.0

        global_attr = torch.tensor(angles[:2] + [offset_norm], dtype=torch.float32).unsqueeze(0)
        
        # 6. 坐标拼接
        pos = torch.from_numpy(np.concatenate([lig_heavy.positions, rec_heavy.positions], axis=0)).float()

        data = Data(x=x, pos=pos)
        data.global_attr = global_attr
        return data