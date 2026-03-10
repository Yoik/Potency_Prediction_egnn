#!/usr/bin/env python3
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# 引入你的项目模块
try:
    from src.config import init_config
    from src.dataset import GPCRDataset # 假设你有这个 dataset 类
    from src.model import DeltaEGNN
except ImportError:
    # 简单的 mock，如果你有现成的 dataset.py 请替换
    print("请确保 src.dataset 能够加载 Boltz 生成的 PDB 数据集")
    sys.exit(1)

def freeze_backbone(model):
    """
    冻结 EGNN 骨干，只训练分类头
    """
    # 1. 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. 解冻 MLP 部分 (假设你的模型里分类头叫 mlp 或 out_proj)
    # 根据你的 model.py 定义调整这里的名字
    for param in model.mlp.parameters():
        param.requires_grad = True
    
    # 如果有 Attention Pooling 层，也可以解冻
    if hasattr(model, 'att_pool'):
        for param in model.att_pool.parameters():
            param.requires_grad = True

    print("  [Info] Backbone frozen. Only MLP head is trainable.")

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        if hasattr(model, 'forward_one'):
            pred = model.forward_one(batch)
        else:
            pred, _ = model(batch, batch) # 兼容 Siamese 接口
            
        loss = criterion(pred.squeeze(), batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boltz_data_dir", required=True, help="Boltz 生成的训练集 PDB 目录")
    parser.add_argument("--pretrained_dir", default="data/features", help="原 MD 模型目录")
    parser.add_argument("--out_dir", default="data/boltz_models", help="微调后模型保存目录")
    parser.add_argument("--epochs", type=int, default=20, help="微调轮数 (不用太多)")
    args = parser.parse_args()

    config = init_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)

    # 1. 加载 Boltz 数据集
    # 这里假设你有一个 CSV 映射文件，或者 dataset 类能自动从文件名读标签
    # 你需要复用训练时的 Dataset 类，但指向 Boltz 的 PDB 文件夹
    print(f"[Data] Loading Boltz training data from {args.boltz_data_dir}...")
    dataset = GPCRDataset(args.boltz_data_dir, config, mode='train') 
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. 遍历 5 个模型进行微调
    model_files = glob.glob(os.path.join(args.pretrained_dir, "model_ensemble_*.pth"))
    
    for i, mf in enumerate(model_files):
        print(f"\n=== Fine-tuning Model {i+1}/{len(model_files)}: {os.path.basename(mf)} ===")
        
        # 加载模型
        n_res = len(config.get_list("residues.obp_residues"))
        model = DeltaEGNN(config)
        model.input_dim = 4 + 2 + n_res
        
        state_dict = torch.load(mf, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        
        # === 关键步骤：冻结骨干 ===
        freeze_backbone(model)
        
        # 定义优化器 (只优化 requires_grad=True 的参数)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        criterion = nn.MSELoss() # 或者你用的其他 Loss
        
        # 微调训练
        for epoch in range(args.epochs):
            loss = train_one_epoch(model, loader, optimizer, criterion, device)
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Loss = {loss:.4f}")
        
        # 保存微调后的模型
        save_path = os.path.join(args.out_dir, f"boltz_adapted_{i}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  Saved adapted model to: {save_path}")

    print("\n[Success] All models adapted for Boltz structures!")
    print("Now run '3_predict_boltz.py' with '--models_dir data/boltz_models'")

if __name__ == "__main__":
    main()