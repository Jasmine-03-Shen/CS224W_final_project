import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class TGCNCell(nn.Module):
    """
    T-GCN cell 
    """
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # 2-layer GCN for spatial features
        self.gcn1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=True)
        gate_in_dim = hidden_channels + hidden_channels
        
        # linear layers for update, reset, and c_t
        self.W_u = nn.Linear(gate_in_dim, hidden_channels)
        self.W_r = nn.Linear(gate_in_dim, hidden_channels)
        self.W_c = nn.Linear(gate_in_dim, hidden_channels)

    def spatial(self, x_t, edge_index):
        """
        computes f(A, X_t) 
        """
        h = self.gcn1(x_t, edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, edge_index)
        h = torch.tanh(h) 
        return h
    
    def forward(self, x_t, edge_index, h_prev):
        z_t = self.spatial(x_t, edge_index)
        # concat spatial features + previous hidden
        zu = torch.cat([z_t, h_prev], dim=-1)
        # update and reset gate 
        u_t = torch.sigmoid(self.W_u(zu))
        r_t = torch.sigmoid(self.W_r(zu))

        zc = torch.cat([z_t, r_t * h_prev], dim=-1) 
        c_t = torch.tanh(self.W_c(zc)) 
        h_t = u_t * h_prev + (1.0 - u_t) * c_t 
        return h_t


class TGCN(nn.Module):
    """
    T-GCN model 
    """
    def __init__(
        self,
        node_feat_dim: int,
        past_window: int,
        future_window: int = 1,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.past_window = past_window
        self.future_window = future_window
        self.hidden_dim = hidden_dim

        self.cell = TGCNCell(
            in_channels=node_feat_dim,
            hidden_channels=hidden_dim,
        )
        self.fc = nn.Linear(hidden_dim, future_window)

    def forward(self, data: Data):
        x = data.x 
        edge_index = data.edge_index
        if x.dim() != 3:
            raise ValueError(f"wrong shape")
        N, F, T_hist = x.shape
        if T_hist != self.past_window:
            raise ValueError(f"time window mismatch")

        h = x.new_zeros((N, self.hidden_dim))  # init hidden
        
        for t in range(T_hist):
            x_t = x[:, :, t] 
            h = self.cell(x_t, edge_index, h)
       
        out = self.fc(h) # predict from last hidden
        return out
