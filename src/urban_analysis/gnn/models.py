import torch
from torch_geometric.nn import GCNConv, GATConv, GAE

class GCNEncoder(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) Encoder for Graph Autoencoder.
    """
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class MultimodalResidualGATEncoder(torch.nn.Module):
    """
    景観の個性を保持するための残差接続を備えたマルチモーダルGATエンコーダ。
    """
    def __init__(self, poi_in_channels, sv_in_channels, shared_channels, out_channels):
        super(MultimodalResidualGATEncoder, self).__init__()
        self.poi_lin = torch.nn.Linear(poi_in_channels, shared_channels)
        self.sv_lin = torch.nn.Linear(sv_in_channels, shared_channels)
        
        # 共有GAT層 (Residual接続を有効化)
        self.conv1 = GATConv(shared_channels, shared_channels, heads=4, concat=True)
        self.conv2 = GATConv(shared_channels * 4, out_channels, heads=1, concat=False)
        
        # 残差用の次元調整層
        self.residual_proj = torch.nn.Linear(shared_channels, out_channels)

    def forward(self, x, edge_index, node_types):
        # 型ごとに初期射影
        x_proj = torch.zeros((x.size(0), self.poi_lin.out_features), device=x.device)
        poi_mask = (node_types == 0)
        sv_mask = (node_types == 1)
        
        if poi_mask.any():
            x_proj[poi_mask] = self.poi_lin(x[poi_mask][:, :self.poi_lin.in_features])
        if sv_mask.any():
            x_proj[sv_mask] = self.sv_lin(x[sv_mask][:, :self.sv_lin.in_features])
            
        x_initial = x_proj.relu()
        
        # グラフ畳み込み
        x = self.conv1(x_initial, edge_index).relu()
        x = self.conv2(x, edge_index)
        
        # ★ 残差接続: 元の特徴量（景観/機能の射影後）を足し合わせる
        # これにより、過剰に平均化されるのを防ぎ、個々のノードの個性を守る
        res = self.residual_proj(x_initial)
        return x + res

def create_gae_model(in_channels, out_channels, device='cpu'):
    """
    Helper function to create and move GAE model to device.
    """
    model = GAE(GCNEncoder(in_channels, out_channels))
    return model.to(device)
