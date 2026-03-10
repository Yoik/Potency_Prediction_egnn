"""
modules/geometry.py
几何计算模块 - 处理平面、向量、距离等计算
"""

import numpy as np
from scipy.spatial.distance import cdist


def calculate_plane_normal(coords):
    """计算点集定义的平面的法向量"""
    c = coords - coords.mean(0)
    return np.linalg.svd(c)[2][2, :]


def get_aromatic_ring_data(residue_group):
    """
    从残基组中提取芳香环数据（质心和法向量）
    
    Args:
        residue_group: MDAnalysis residue group
        
    Returns:
        (center_of_mass, normal_vector) or (center, None)
    """
    if len(residue_group) == 0:
        return None, None
    
    rn = residue_group.resnames[0]
    ra = None
    
    # 根据残基类型选择环原子
    if rn in ['PHE', 'TYR']:
        ra = residue_group.atoms.select_atoms("name CG CD1 CD2 CE1 CE2 CZ")
    elif rn == 'TRP':
        ra = residue_group.atoms.select_atoms("name CD2 CE2 CE3 CZ2 CZ3 CH2")
    elif rn in ['HIS', 'HSD', 'HSE', 'HSP']:
        ra = residue_group.atoms.select_atoms("name CG ND1 CD2 CE1 NE2")
    
    exp_count = 5 if rn.startswith('HIS') else 6
    
    if ra and len(ra) == exp_count:
        return ra.center_of_mass(), calculate_plane_normal(ra.positions)
    
    # 备选方案：使用侧链
    side = residue_group.atoms.select_atoms("not name N CA C O")
    return (side.center_of_mass() if len(side) > 0 else residue_group.center_of_mass()), None


def calculate_carbon_angles_and_decay(carbon_positions, phe_center, phe_normal):
    """
    计算配体碳原子与Phe平面的夹角和角度衰减因子
    
    Args:
        carbon_positions: (6, 3) 数组，6个碳原子的坐标
        phe_center: (3,) 数组，Phe质心坐标
        phe_normal: (3,) 数组，Phe平面法向量
        
    Returns:
        angles: (6,) 数组，每个碳与Phe平面的夹角（度数）
        decay_factors: (6,) 数组，角度衰减因子
    """
    n_atoms = len(carbon_positions)
    angles = np.zeros(n_atoms)
    decay_factors = np.zeros(n_atoms)

    if phe_normal is None:
        # 如果没有平面法向量，无法计算
        return angles, np.ones(n_atoms)
    
    for i, c_pos in enumerate(carbon_positions):
        # 从Phe质心指向碳原子的向量
        vec_to_c = c_pos - phe_center
        
        # 计算与平面法向量的夹角
        dot_prod = np.dot(vec_to_c, phe_normal)
        norm = np.linalg.norm(vec_to_c)
        
        if norm < 1e-10:
            angles[i] = 0
            decay_factors[i] = 0
            continue
        
        cos_angle = dot_prod / norm
        angle_rad = np.arccos(np.clip(np.abs(cos_angle), -1, 1))
        angles[i] = np.degrees(angle_rad)
        
        # 角度衰减：T型接触对应90°
        angle_deviation = np.abs(angles[i] - 90.0)
        decay_factors[i] = np.exp(-((angle_deviation / 30.0) ** 2))
    
    return angles, decay_factors


def calculate_distance_decay(carbon_positions, phe_center, phe_normal):
    """
    计算碳原子到Phe平面的垂直距离和距离衰减因子
    
    Args:
        carbon_positions: (6, 3) 数组
        phe_center: (3,) 数组
        phe_normal: (3,) 数组
        
    Returns:
        perp_distances: (6,) 数组，垂直距离（Å）
        decay_factors: (6,) 数组，距离衰减因子
    """
    n_atoms = len(carbon_positions)

    if phe_normal is None:
        return np.zeros(n_atoms), np.ones(n_atoms)
    
    # 从Phe质心到各碳原子的向量
    vec_to_atoms = carbon_positions - phe_center
    
    # 垂直距离 = |向量 · 法向量|
    perp_distances = np.abs(np.dot(vec_to_atoms, phe_normal))
    
    # 距离衰减：参考距离2.0Å
    decay_factors = np.exp(-((perp_distances / 2.0) ** 2))
    
    return perp_distances, decay_factors


def calculate_combined_weight(elf_weights, angle_decay, distance_decay):
    """
    计算综合权重：ELF权重 × 角度衰减 × 距离衰减
    
    Args:
        elf_weights: (6,) 数组，ELF权重
        angle_decay: (6,) 数组，角度衰减因子
        distance_decay: (6,) 数组，距离衰减因子
        
    Returns:
        combined_weights: (6,) 数组，综合权重
    """
    return elf_weights * angle_decay * distance_decay


def calculate_weighted_average_distance(distances, weights):
    """
    计算加权平均距离
    
    Args:
        distances: (6,) 数组，碳到Phe质心的距离
        weights: (6,) 数组，权重
        
    Returns:
        weighted_avg: 加权平均距离，或 np.mean(distances) 如果权重为0
    """
    sum_weights = np.sum(weights)
    if sum_weights > 1e-10:
        return np.average(distances, weights=weights)
    else:
        return np.mean(distances)
