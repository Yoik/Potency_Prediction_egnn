import os
import sys
import yaml
import torch
import pandas as pd
import MDAnalysis as mda
from torch_geometric.data import Batch
from tqdm import tqdm

# 引入项目模块
from src.config import init_config
from src.model import DeltaEGNN
from src.featurizer import PhysicsFeaturizer  # <--- 我们刚才写的新类

def main():
    # 1. 初始化
    config = init_config()
    featurizer = PhysicsFeaturizer(config) # 初始化一次，复用 Reference 对齐器
    
    # 读取 YAML 配置
    with open("predict_config.yaml", 'r') as f:
        user_cfg = yaml.safe_load(f)
    
    device = torch.device(user_cfg['settings'].get('device', 'cpu'))
    
    # 2. 加载模型
    print(f"Loading Model from {user_cfg['settings']['model_path']}...")
    model = DeltaEGNN(config).to(device)
    model.load_state_dict(torch.load(user_cfg['settings']['model_path'], map_location=device))
    model.eval()
    
    # 3. 准备锚点 (Anchor)
    print("Loading Anchor Data...")
    ref_graphs = torch.load(user_cfg['reference']['feature_path'], weights_only=False)
    ref_data = ref_graphs[0].to(device) # 取第一帧作为标尺
    anchor_val = user_cfg['reference']['true_efficacy']
    
    # 4. 批量推理循环
    results = []
    print(f"\nProcessing {len(user_cfg['targets'])} targets...")
    
    for target in tqdm(user_cfg['targets']):
        name = target['name']
        pdb_path = target['pdb']
        
        if not os.path.exists(pdb_path):
            print(f"Skipping {name}: PDB not found")
            continue
            
        try:
            # === 这里的逻辑现在和训练时完全一致了 ===
            u = mda.Universe(pdb_path)
            # 一行代码完成所有特征提取
            target_data = featurizer.process_frame(u, cub_path=target.get('cub'))
            
            target_data = target_data.to(device)
            
            # 推理
            batch_target = Batch.from_data_list([target_data])
            batch_ref = Batch.from_data_list([ref_data])
            
            with torch.no_grad():
                raw_t, raw_r = model(batch_target, batch_ref)
                delta = raw_t - raw_r
                pred = anchor_val + delta
                
            results.append({
                "Compound": name,
                "Pred_Efficacy": pred.item(),
                "Delta": delta.item()
            })
            
        except Exception as e:
            print(f"Failed to process {name}: {e}")
            
    # 5. 输出结果
    df = pd.DataFrame(results).sort_values("Pred_Efficacy", ascending=False)
    print("\n" + "="*40)
    print(df.to_string(index=False, float_format="%.4f"))
    print("="*40)
    df.to_csv(user_cfg['settings']['output_csv'], index=False)

if __name__ == "__main__":
    main()