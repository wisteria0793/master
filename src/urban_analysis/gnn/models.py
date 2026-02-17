import torch
from torch_geometric.nn import GCNConv, GAE

class GCNEncoder(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) Encoder for Graph Autoencoder.
    
    Args:
        in_channels (int): Input feature dimension.
        out_channels (int): Output embedding dimension.
    """
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # 2層GCN: 入力 -> 2*出力 -> 出力
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def create_gae_model(in_channels, out_channels, device='cpu'):
    """
    Helper function to create and move GAE model to device.
    """
    model = GAE(GCNEncoder(in_channels, out_channels))
    return model.to(device)
