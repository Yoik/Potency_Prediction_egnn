"""
cube_parser.py
立方体文件 (Cube) 解析模块
"""

import numpy as np

BOHR_TO_ANGSTROM = 0.52917721067


class CubeParser:
    """
    解析 Cube 格式文件，提取电子密度数据和碳原子积分值
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.origin = None
        self.spacing = None
        self.dims = None
        self.atom_lines = []
        self.is_header_bohr = True
        self._load()

    def _load(self):
        """读取并解析立方体文件"""
        try:
            with open(self.filepath, 'r') as f:
                lines = f.readlines()
                
                # 解析头部信息
                parts = lines[2].split()
                natoms = int(parts[0])
                
                if natoms > 0:
                    self.is_header_bohr = True
                else:
                    self.is_header_bohr = False
                    natoms = abs(natoms)
                
                # 原点坐标
                self.origin = np.array([float(x) for x in parts[1:4]])
                
                # 网格维度和间距
                nx = int(lines[3].split()[0])
                vx = np.array([float(x) for x in lines[3].split()[1:4]])
                ny = int(lines[4].split()[0])
                vy = np.array([float(x) for x in lines[4].split()[1:4]])
                nz = int(lines[5].split()[0])
                vz = np.array([float(x) for x in lines[5].split()[1:4]])
                
                self.dims = (nx, ny, nz)
                self.spacing = np.array([vx[0], vy[1], vz[2]])
                
                # 原子信息行
                self.atom_lines = lines[6:6 + natoms]
                
                # 读取数据
                data_start = 6 + natoms
                raw_data = []
                for line in lines[data_start:]:
                    raw_data.extend([float(x) for x in line.split()])
                
                self.data = np.array(raw_data).reshape(self.dims)
        except Exception as e:
            print(f"     [Cube Error] {e}")
            self.data = None

    def get_carbon_integrals(self, radius=1.5, atom_indices=None):
        """
        计算立方体中指定重原子周围指定半径内的电子密度积分
        
        Args:
            radius: 积分半径 (Angstrom)
            atom_indices: 要计算的原子在 cube 文件中的索引列表（从 0 开始）
                         如果为 None，则计算所有重原子
            
        Returns:
            np.array: 指定重原子的积分值数组
        """
        if self.data is None:
            return np.array([])
        
        origin_ang = self.origin * BOHR_TO_ANGSTROM
        spacing_ang = self.spacing * BOHR_TO_ANGSTROM
        integrals = []
        nx, ny, nz = self.dims
        
        atom_counter = 0  # 追踪重原子的索引
        
        voxel_volume = self.spacing[0] * self.spacing[1] * self.spacing[2]
        
        for line in self.atom_lines:
            parts = line.split()
            atomic_num = int(parts[0])
            
            # 只处理重原子（原子序数 >= 6）
            if atomic_num < 6:
                continue
            
            # 如果指定了原子索引，检查是否在范围内
            if atom_indices is not None and atom_counter not in atom_indices:
                atom_counter += 1
                continue
            
            raw_coord = np.array([float(x) for x in parts[2:5]])
            atom_coord_ang = (raw_coord * BOHR_TO_ANGSTROM 
                              if self.is_header_bohr else raw_coord)
            
            # 确定积分区间
            min_idx = np.maximum(
                np.floor((atom_coord_ang - radius - origin_ang) / spacing_ang).astype(int), 
                0
            )
            max_idx = np.minimum(
                np.ceil((atom_coord_ang + radius - origin_ang) / spacing_ang).astype(int) + 1,
                [nx, ny, nz]
            )
            
            if np.any(min_idx >= max_idx):
                integrals.append(0.0)
                atom_counter += 1
                continue
            
            # 提取局部数据
            local_data = self.data[
                min_idx[0]:max_idx[0],
                min_idx[1]:max_idx[1],
                min_idx[2]:max_idx[2]
            ]
            
            # 构建网格坐标
            ix = np.arange(min_idx[0], max_idx[0])
            iy = np.arange(min_idx[1], max_idx[1])
            iz = np.arange(min_idx[2], max_idx[2])
            X, Y, Z = np.meshgrid(ix, iy, iz, indexing='ij')
            
            grid_pos_x = origin_ang[0] + X * spacing_ang[0]
            grid_pos_y = origin_ang[1] + Y * spacing_ang[1]
            grid_pos_z = origin_ang[2] + Z * spacing_ang[2]
            
            # 计算距离和掩码
            dist_sq = ((grid_pos_x - atom_coord_ang[0]) ** 2 +
                      (grid_pos_y - atom_coord_ang[1]) ** 2 +
                      (grid_pos_z - atom_coord_ang[2]) ** 2)
            mask = dist_sq < (radius ** 2)
            
            # [修改] 积分 = Sum(密度 * dV)
            raw_sum = np.sum(local_data[mask])
            physical_integral = raw_sum * voxel_volume
            
            integrals.append(physical_integral)
            atom_counter += 1
        
        return np.array(integrals)
