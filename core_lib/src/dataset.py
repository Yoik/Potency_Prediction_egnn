import os
import torch
import glob
import pandas as pd
import random
import numpy as np
from torch_geometric.data import Dataset, Batch
from torch.utils.data import Dataset as TorchDataset

class MolGraphDataset(Dataset):
    """
    基础数据集类：负责加载和管理所有图数据
    [修复点]：恢复了模糊匹配逻辑，确保文件夹名与CSV标签不完全一致时也能加载数据
    """
    def __init__(self, root_dir, label_file):
        self.root_dir = root_dir
        self.label_map = self._load_labels(label_file)
        
        # === 核心结构 ===
        # data_map: { 'ARI': [Data, Data...], 'Dopa': [...] }
        # 用于评估阶段快速获取某化合物的所有帧
        self.data_map = {} 
        
        # data_list: [Data, Data...]
        # 扁平化存储，用于兼容 PyG Dataset 接口
        self.data_list = [] 
        
        self._process()
        
        super().__init__(root_dir, transform=None, pre_transform=None)

    def _load_labels(self, label_file):
        try:
            df = pd.read_csv(label_file)
            # 假设第一列是名字，第二列是数值
            name_col = df.columns[0]
            val_col = df.columns[1]
            label_map = {str(row[name_col]).strip(): float(row[val_col]) for _, row in df.iterrows()}
            print(f"[Dataset] Loaded labels for {len(label_map)} compounds: {list(label_map.keys())}")
            return label_map
        except Exception as e:
            raise RuntimeError(f"Failed to read label file {label_file}: {e}")

    def _process(self):
        print(f"[Dataset] Loading graphs from {self.root_dir}...")
        
        # 搜索模式: root/Compound_Dir/Replica_Dir/graph_features.pt
        search_path = os.path.join(self.root_dir, "*", "*", "graph_features.pt")
        pt_files = glob.glob(search_path)
        
        if len(pt_files) == 0:
            print(f"[Warn] No .pt files found in {search_path}. Check your path!")
            return

        loaded_compounds = set()

        for pt_path in pt_files:
            # 解析路径获取文件夹名
            # 路径结构: .../Compound_Name/Replica_ID/graph_features.pt
            replica_dir = os.path.dirname(pt_path)
            compound_dir = os.path.dirname(replica_dir)
            folder_name = os.path.basename(compound_dir)
            
            # === [关键修复] 模糊匹配逻辑 ===
            matched_label = None
            
            # 1. 尝试精确匹配
            if folder_name in self.label_map:
                matched_label = folder_name
            else:
                # 2. 尝试模糊匹配 (Case-insensitive & Substring)
                # 比如 label="ARI", folder="ARI_charmm" -> 匹配成功
                for label_key in self.label_map:
                    if label_key.lower() == folder_name.lower(): # 大小写不敏感的全匹配
                         matched_label = label_key
                         break
                    if label_key.lower() in folder_name.lower(): # 子串匹配
                        matched_label = label_key
                        break
            
            if matched_label is None:
                # print(f"  [Skip] No label match for folder: {folder_name}")
                continue
                
            try:
                # 加载数据
                graph_list = torch.load(pt_path, weights_only=False)
                
                # 初始化
                if matched_label not in self.data_map:
                    self.data_map[matched_label] = []
                
                # 存入 data_map
                self.data_map[matched_label].extend(graph_list)
                
                # 存入 data_list
                self.data_list.extend(graph_list)
                
                loaded_compounds.add(matched_label)
                
            except Exception as e:
                print(f"  [Warn] Failed to load {pt_path}: {e}")

        print(f"[Dataset] Successfully loaded data for {len(loaded_compounds)} compounds: {sorted(list(loaded_compounds))}")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
    
    def get_compounds(self):
        return sorted(list(self.data_map.keys()))


class PairwiseGraphDataset(TorchDataset):
    """
    配对数据集：用于训练 (Siamese Network)
    随机采样 (Compound_A, Compound_B) 的帧进行对比
    """
    def __init__(self, base_dataset: MolGraphDataset, compound_list=None, mode='train', samples_per_epoch=2000):
        self.base_ds = base_dataset
        self.mode = mode
        self.samples_per_epoch = samples_per_epoch
        
        if compound_list is None:
            self.compound_list = base_dataset.get_compounds()
        else:
            self.compound_list = compound_list
            
        # 验证是否有数据
        self.valid_list = [c for c in self.compound_list if c in base_dataset.data_map and len(base_dataset.data_map[c]) > 0]
        
        if len(self.valid_list) < 2:
            print(f"[PairwiseDS] Warning: Not enough compounds ({len(self.valid_list)}) to form pairs!")

    def __len__(self):
        # 训练时这是一个虚拟长度，决定了一个 Epoch 跑多少次 Step
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """
        随机采样一对 (A, B)
        返回: (DataA, DataB, Delta_Y, Y_A, Y_B)
        """
        # 1. 随机选两个不同的化合物
        # (如果只有 1 个化合物，random.sample 会报错，需加保护)
        if len(self.valid_list) < 2:
             # Fallback: 自己对比自己 (Delta=0)，防止 Crash
             cmpd_a = self.valid_list[0]
             cmpd_b = self.valid_list[0]
        else:
             cmpd_a, cmpd_b = random.sample(self.valid_list, 2)
        
        # 2. 随机选它们的某一帧
        frames_a = self.base_ds.data_map[cmpd_a]
        frames_b = self.base_ds.data_map[cmpd_b]
        
        data_a = random.choice(frames_a)
        data_b = random.choice(frames_b)
        
        # 3. 获取标签
        y_a = self.base_ds.label_map[cmpd_a]
        y_b = self.base_ds.label_map[cmpd_b]
        
        delta_y = float(y_a - y_b)
        
        # 返回 5 元组
        return (
            data_a, 
            data_b, 
            torch.tensor([delta_y], dtype=torch.float32), 
            torch.tensor([y_a], dtype=torch.float32), 
            torch.tensor([y_b], dtype=torch.float32)
        )

def get_pairwise_loader(dataset, batch_size=32, shuffle=True, **kwargs):
    """
    自定义 Loader，处理 PyG Data 对象的 Batch 组装
    """
    def collate_fn(batch):
        # batch: List of (DataA, DataB, Delta, YA, YB)
        
        list_a = [item[0] for item in batch]
        list_b = [item[1] for item in batch]
        list_delta = [item[2] for item in batch]
        list_ya = [item[3] for item in batch]
        list_yb = [item[4] for item in batch]
        
        # PyG 的 Batch.from_data_list 会自动把多个图拼成一个大图
        batch_a = Batch.from_data_list(list_a)
        batch_b = Batch.from_data_list(list_b)
        
        batch_delta = torch.cat(list_delta)
        batch_ya = torch.cat(list_ya)
        batch_yb = torch.cat(list_yb)
        
        return batch_a, batch_b, batch_delta, batch_ya, batch_yb

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs)