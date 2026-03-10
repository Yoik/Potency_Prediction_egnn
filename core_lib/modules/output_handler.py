"""
modules/output_handler.py
输出处理模块 - 管理结果文件输出和汇总
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

class OutputHandler:
    """处理分析结果的输出"""
    
    def __init__(self, compound_id, replica_name, output_base_dir="./results"):
        """初始化输出处理器"""
        self.compound_id = compound_id
        self.replica_name = replica_name
        self.output_dir = Path(output_base_dir) / compound_id / replica_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_features_exist(self, filename_suffix=""):
        """检查特征文件是否已存在"""
        filename = f"{self.compound_id}_{self.replica_name}_features{filename_suffix}.npy"
        filepath = self.output_dir / filename
        return filepath.exists()
    
    def save_timeseries(self, df, filename_suffix=""):
        """保存时间序列数据 (CSV)"""
        filename = f"{self.compound_id}_{self.replica_name}_TimeSeries{filename_suffix}.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        return filepath

    def save_features(self, feature_array, filename_suffix=""):
        """
        保存机器学习特征矩阵 (.npy)
        Shape: (Frames, Features)
        """
        filename = f"{self.compound_id}_{self.replica_name}_features{filename_suffix}.npy"
        filepath = self.output_dir / filename
        np.save(filepath, feature_array)
        print(f"     [Output] Features saved: {filepath} (Shape: {feature_array.shape})")
        return filepath
    
    def save_stats(self, stats_df, filename_suffix=""):
        """保存统计数据 (CSV)"""
        filename = f"{self.compound_id}_{self.replica_name}_Stats{filename_suffix}.csv"
        filepath = self.output_dir / filename
        stats_df.to_csv(filepath, index=False)
        return filepath
    
    def save_projection(self, filepath):
        """保存投影图像"""
        filename = f"{self.compound_id}_{self.replica_name}_projection.png"
        target_path = self.output_dir / filename
        return target_path
    
    @staticmethod
    def aggregate_timeseries(timeseries_list, output_dir, compound_id):
        """聚合多个副本的时间序列数据"""
        output_dir = Path(output_dir)
        compound_dir = output_dir / compound_id
        compound_dir.mkdir(parents=True, exist_ok=True)
        
        full_df = pd.concat(timeseries_list, ignore_index=False)
        full_df["Time"] = full_df["Time"].round(2)
        
        # 按时间点聚合
        agg_mean = full_df.groupby("Time").mean(numeric_only=True).reset_index()
        agg_std = full_df.groupby("Time").std(numeric_only=True).reset_index()
        
        result = agg_mean.copy()
        for col in agg_mean.columns:
            if col != "Time":
                result[f"{col}_SD"] = agg_std[col]
        
        output_file = compound_dir / "All_TimeSeries.csv"
        result.to_csv(output_file, index=False)
        return result
    
    @staticmethod
    def aggregate_stats(stats_list, output_dir, compound_id):
        """聚合多个副本的统计数据"""
        output_dir = Path(output_dir)
        compound_dir = output_dir / compound_id
        compound_dir.mkdir(parents=True, exist_ok=True)
        
        stats_df = pd.DataFrame(stats_list)
        
        # 计算平均值
        avg_row = stats_df.mean(numeric_only=True).to_dict()
        avg_row["Replica"] = "AVERAGE"
        avg_row["Compound"] = compound_id
        
        # 追加平均行
        result = pd.concat([stats_df, pd.DataFrame([avg_row])], ignore_index=True)
        
        output_file = compound_dir / "All_Stats.csv"
        result.to_csv(output_file, index=False)
        return result

def calculate_interaction_strength(elf_weights, angles_389, angles_390, 
                                   distance_decays_389, distance_decays_390):
    """
    计算综合相互作用强度 - 基于6个碳原子的综合评估
    
    Args:
        elf_weights: (6,) 数组，ELF权重
        angles_389: (6,) 数组，与389平面的夹角
        angles_390: (6,) 数组，与390平面的夹角
        distance_decays_389: (6,) 数组，389距离衰减
        distance_decays_390: (6,) 数组，390距离衰减
        
    Returns:
        dict 包含：
            - strength_389: 与Phe389的相互作用强度 (0-1)
            - strength_390: 与Phe390的相互作用强度 (0-1)
            - strength_combined: 综合相互作用强度 (0-1)
            - quality_score_389: Phe389对接质量分数
            - quality_score_390: Phe390对接质量分数
            - major_contributor_389: 主要贡献碳 (1-6)
            - major_contributor_390: 主要贡献碳 (1-6)
    """
    # 计算角度衰减因子
    angle_decay_389 = np.exp(-((np.abs(angles_389 - 90.0) / 30.0) ** 2))
    angle_decay_390 = np.exp(-((np.abs(angles_390 - 90.0) / 30.0) ** 2))
    
    # 获取实际原子数量 (6 或 9)
    n_atoms = len(elf_weights)

    # 综合权重
    combined_389 = elf_weights * angle_decay_389 * distance_decays_389
    combined_390 = elf_weights * angle_decay_390 * distance_decays_390
    
    # 归一化强度 (0-1)
    strength_389 = np.sum(combined_389) / (float(n_atoms) * np.max(elf_weights) + 1e-10)
    strength_390 = np.sum(combined_390) / (float(n_atoms) * np.max(elf_weights) + 1e-10)
    strength_combined = (strength_389 + strength_390) / 2.0
    
    # 质量评分（基于最佳碳原子的贡献）
    best_idx_389 = np.argmax(combined_389)
    best_idx_390 = np.argmax(combined_390)
    
    quality_389 = combined_389[best_idx_389] / np.max(elf_weights)
    quality_390 = combined_390[best_idx_390] / np.max(elf_weights)
    
    return {
        "strength_389": strength_389,
        "strength_390": strength_390,
        "strength_combined": strength_combined,
        "quality_score_389": quality_389,
        "quality_score_390": quality_390,
        "major_contributor_389": best_idx_389 + 1,
        "major_contributor_390": best_idx_390 + 1,
        "avg_angle_389": np.mean(angles_389),
        "avg_angle_390": np.mean(angles_390),
        "std_angle_389": np.std(angles_389),
        "std_angle_390": np.std(angles_390)
    }


def format_interaction_strength(strength_dict):
    """
    格式化相互作用强度为可读的字符串
    
    Returns:
        str 包含详细的相互作用信息
    """
    lines = [
        "=" * 70,
        "综合相互作用强度分析",
        "=" * 70,
        f"Phe389 相互作用强度: {strength_dict['strength_389']:.3f} (质量分数: {strength_dict['quality_score_389']:.3f})",
        f"Phe390 相互作用强度: {strength_dict['strength_390']:.3f} (质量分数: {strength_dict['quality_score_390']:.3f})",
        f"综合相互作用强度: {strength_dict['strength_combined']:.3f}",
        "",
        f"Phe389 主要贡献原子索引:{strength_dict['major_contributor_389']}",
        f"  平均夹角: {strength_dict['avg_angle_389']:.1f}° (标准差: {strength_dict['std_angle_389']:.1f}°)",
        "",
        f"Phe390 主要贡献原子索引:{strength_dict['major_contributor_390']}",
        f"  平均夹角: {strength_dict['avg_angle_390']:.1f}° (标准差: {strength_dict['std_angle_390']:.1f}°)",
        "=" * 70
    ]
    return "\n".join(lines)
