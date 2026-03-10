import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.nn import radius_graph
from torch.nn import Sequential as Seq, Linear as Lin, SiLU

class CustomEGNNConv(MessagePassing):
    """
    E(n) Equivariant Graph Convolutional Layer
    (保持不变)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=0):
        super(CustomEGNNConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        self.message_mlp = Seq(
            Lin(in_channels * 2 + 1 + edge_dim, hidden_channels),
            SiLU(),
            Lin(hidden_channels, hidden_channels),
            SiLU()
        )
        
        self.node_mlp = Seq(
            Lin(in_channels + hidden_channels, hidden_channels),
            SiLU(),
            Lin(hidden_channels, out_channels)
        )
        
        self.coord_mlp = Seq(
            Lin(hidden_channels, hidden_channels),
            SiLU(),
            Lin(hidden_channels, 1, bias=False)
        )

    def forward(self, h, pos, edge_index, edge_attr=None):
        return self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)

    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):
        rel_pos = pos_i - pos_j
        dist_sq = torch.sum(rel_pos**2, dim=-1, keepdim=True)
        
        if edge_attr is not None:
            input_feats = torch.cat([h_i, h_j, dist_sq, edge_attr], dim=-1)
        else:
            input_feats = torch.cat([h_i, h_j, dist_sq], dim=-1)
            
        msg = self.message_mlp(input_feats)
        coord_weight = self.coord_mlp(msg)
        coord_msg = rel_pos * coord_weight 
        
        return msg, coord_msg

    def aggregate(self, inputs, index, dim_size=None):
        msgs, coord_msgs = inputs
        aggr_msg = torch.zeros(dim_size, msgs.size(-1), device=msgs.device)
        aggr_msg.index_add_(0, index, msgs)
        aggr_coord = torch.zeros(dim_size, coord_msgs.size(-1), device=coord_msgs.device)
        aggr_coord.index_add_(0, index, coord_msgs)
        return aggr_msg, aggr_coord

    def update(self, aggr_out, h, pos):
        aggr_msg, aggr_coord = aggr_out
        pos_new = pos + aggr_coord
        input_feat = torch.cat([h, aggr_msg], dim=-1)
        h_new = self.node_mlp(input_feat)
        if h.shape[-1] == h_new.shape[-1]:
            h_new = h + h_new
        return h_new, pos_new

class DeltaEGNN(nn.Module):
    """
    EGNN + Explicit Global Geometry + Mean Pooling (Anti-Size-Bias)
    """
    def __init__(self, config=None):
        super(DeltaEGNN, self).__init__()
        
        if config:
            self.node_in_dim = config.get_int("model.node_in_dim", 25)
            self.global_dim = config.get_int("model.global_dim", 3)
            self.hidden_dim = config.get_int("model.hidden_dim", 64)
            self.n_layers = config.get_int("model.n_layers", 4)
            self.dropout_rate = config.get_float("model.dropout_rate", 0.1)
            self.graph_radius = config.get_float("model.graph_radius", 5.0)
        else:
            # Fallback 默认值
            self.node_in_dim = 25
            self.global_dim = 3
            self.hidden_dim = 32
            self.n_layers = 3
            self.dropout_rate = 0.3
            self.graph_radius = 5.0

        # Node Embedding
        self.node_embedding = nn.Sequential(
            Lin(self.node_in_dim, self.hidden_dim),
            SiLU(),
            Lin(self.hidden_dim, self.hidden_dim)
        )

        # EGNN Layers
        self.convs = nn.ModuleList()
        for _ in range(self.n_layers):
            self.convs.append(
                CustomEGNNConv(
                    in_channels=self.hidden_dim,
                    hidden_channels=self.hidden_dim,
                    out_channels=self.hidden_dim
                )
            )

        # === 关键修改：移除 Attention Pooling ===
        # 我们将直接使用 global_mean_pool，不需要可学习参数
        
        # Global Feature Encoder (Simple MLP)
        self.global_encoder = nn.Sequential(
            Lin(self.global_dim, 16),
            SiLU(),
            Lin(16, 16)
        )

        # Final Head (Graph Rep + Global Rep)
        # Input dim = hidden_dim (32) + global_encoded (16) = 48
        self.head = nn.Sequential(
            Lin(self.hidden_dim + 16, self.hidden_dim),
            SiLU(),
            nn.Dropout(self.dropout_rate),
            Lin(self.hidden_dim, 1)
        )

    def forward_one(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        
        # 处理全局特征
        global_feat = data.global_attr
        
        edge_index = radius_graph(pos, r=self.graph_radius, batch=batch, loop=False)
        h = self.node_embedding(x)

        for conv in self.convs:
            h, pos = conv(h=h, pos=pos, edge_index=edge_index)

        # === [修改] 物理加权池化 (Physics-Weighted Pooling) ===
        # 提取第 5 维 (索引4) 的 electronic_weight 作为每个原子的物理重要性权重
        phys_weights = x[:, 4].unsqueeze(1) # 形状: [Num_Nodes, 1]
        
        # 为了防止全 0 导致除零错误，加一个极小的 epsilon
        phys_weights = torch.clamp(phys_weights, min=1e-6)
        
        # 1. 对节点特征进行物理加权
        weighted_h = h * phys_weights
        
        # 2. 计算 Batch 中每张图的加权特征总和，以及权重总和
        sum_weighted_h = global_add_pool(weighted_h, batch)
        sum_weights = global_add_pool(phys_weights, batch)
        
        # 3. 得到真正的物理加权平均特征 (屏蔽长链尾部噪声)
        graph_repr = sum_weighted_h / sum_weights # [Batch, Hidden]
        
        # Global Encoding
        global_repr = self.global_encoder(global_feat) # [Batch, 16]
        
        # Concatenate
        combined = torch.cat([graph_repr, global_repr], dim=1) # [Batch, 48]

        out = self.head(combined)
        return out

    def forward(self, data_a, data_b=None):
        pred_a = self.forward_one(data_a)
        
        if data_b is not None:
            pred_b = self.forward_one(data_b)
            return pred_a, pred_b
        
        return pred_a