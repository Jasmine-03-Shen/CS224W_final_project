import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import GCN, GAT, GIN
from torch.nn import GRUCell, LSTMCell, TransformerEncoderLayer

class TGNNCell(nn.Module):
	def __init__(self, in_channels: int, hidden_size: int, num_layers: int = 1, gnn_type: str = 'gcn'):
		super(TGNNCell, self).__init__()
		if gnn_type == 'gat':
			self.gnn = GAT(in_channels=in_channels, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=num_layers)
		elif gnn_type == 'gcn':
			self.gnn = GCN(in_channels=in_channels, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=num_layers)
		elif gnn_type == 'gin':
			self.gnn = GIN(in_channels=in_channels, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=num_layers)
		self.temporal_model = GRUCell(input_size=hidden_size + in_channels, hidden_size=hidden_size)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor, h: torch.tensor) -> tuple[
		torch.tensor, torch.tensor]:
		# print('TGNNCell input shape:', x.shape, torch.isnan(x).any())
		gnn_output = self.gnn(x, edge_index, edge_weight)
		# print('TGNNCell gcn_out before activation shape:', gcn_out.shape, torch.isnan(gcn_out).any())
		gnn_output = F.sigmoid(gnn_output)
		# print('TGNNCell gcn_out shape:', gcn_out.shape, torch.isnan(gcn_out).any())
		output = self.temporal_model(torch.cat([x, gnn_output], dim=-1), h)
		# print('TGNNCell output shape:', output.shape, torch.isnan(output).any())
		return output

class LSTMGCNCell(nn.Module):
	def __init__(self, in_channels: int, hidden_size: int, num_layers: int = 1, gnn_type: str = 'gcn'):
		super(LSTMGCNCell, self).__init__()
		if gnn_type == 'gat':
			self.gnn = GAT(in_channels=in_channels, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=num_layers)
		elif gnn_type == 'gcn':
			self.gnn = GCN(in_channels=in_channels, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=num_layers)
		elif gnn_type == 'gin':
			self.gnn = GIN(in_channels=in_channels, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=num_layers)
		self.temporal_model = LSTMCell(input_size=hidden_size + in_channels, hidden_size=hidden_size)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor, h_c: tuple[
		torch.tensor, torch.tensor]) -> tuple[torch.tensor, torch.tensor]:
		h, c = h_c
		gnn_output = F.sigmoid(self.gnn(x, edge_index, edge_weight))  # Act
		# print('devices in LSTMGCNCell:', x.device, edge_index.device, edge_weight.device, h.device, c.device, gnn_output.device)
		h_next, c_next = self.temporal_model(torch.cat([x, gnn_output], dim=-1), (h, c))
		return h_next, c_next


class TGCN(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, hidden_size: int, n_layers: int = 2, edge_attr_channels: int = 12, output_activation: nn.Module = None, gnn_type: str = 'gcn'):
		super(TGCN, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = max(1, n_layers)
		cells = []
		cells.append(TGNNCell(in_channels, hidden_size, gnn_type=gnn_type))
		for _ in range(1, self.n_layers):
			cells.append(TGNNCell(hidden_size, hidden_size, gnn_type=gnn_type))
		self.cells = nn.ModuleList(cells)
		# self.edge_embed = nn.Linear(edge_attr_channels, 1)
		self.out = nn.Sequential(
			nn.Linear(hidden_size, out_channels),
			output_activation if output_activation is not None else nn.Identity(),
		)
		self.gnn_type = gnn_type

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_attr: torch.tensor) -> torch.tensor:
		batch_size, num_feats, seq_len = x.shape
		# edge_weight = self.edge_embed(edge_attr).squeeze(-1)
		# if self.use_gat:
		# 	edge_weight = edge_attr
		# else:
		# 	edge_weight = edge_attr[:, -1]
		edge_weight = edge_attr[:, -1] # corr
		# print('edge_weight shape:', edge_weight, torch.isnan(edge_weight).any())
		h_prev = [
			torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.n_layers)
		]
		for t in range(seq_len):
			h = x[:, :, t] 
			for i, cell in enumerate(self.cells):
				h = cell(h, edge_index, edge_weight, h_prev[i])
				h_prev[i] = h
				# print('h shape at time', t, 'layer', i, ':', h.shape, torch.isnan(h).any(), torch.isnan(edge_weight).any())
		return self.out(h_prev[-1])

class LSTMGCN(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, hidden_size: int, n_layers: int = 2, edge_attr_channels: int = 12, output_activation: nn.Module = None, gnn_type: str = 'gcn'):
		super(LSTMGCN, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = max(1, n_layers)
		cells = []
		cells.append(LSTMGCNCell(in_channels, hidden_size, gnn_type=gnn_type))
		for _ in range(1, self.n_layers):
			cells.append(LSTMGCNCell(hidden_size, hidden_size, gnn_type=gnn_type))
		self.cells = nn.ModuleList(cells)
		# self.edge_embed = nn.Linear(edge_attr_channels, 1)
		self.out = nn.Sequential(
			nn.Linear(hidden_size, out_channels),
			output_activation if output_activation is not None else nn.Identity(),
		)
		self.gnn_type = gnn_type

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_attr: torch.tensor) -> torch.tensor:
		batch_size, num_feats, seq_len = x.shape
		# edge_weight = self.edge_embed(edge_attr).squeeze(-1)
		# if self.use_gat:
		# 	edge_weight = edge_attr
		# else:
		# 	edge_weight = edge_attr[:, -1]
		edge_weight = None
		h_prev = [
			(torch.zeros(batch_size, self.hidden_size).to(x.device), torch.zeros(batch_size, self.hidden_size).to(x.device)) for _ in range(self.n_layers)
		]
		for t in range(seq_len):
			h = x[:, :, t]  
			for i, cell in enumerate(self.cells):
				h, c = cell(h, edge_index, edge_weight, h_prev[i])
				h_prev[i] = (h, c)
		return self.out(h_prev[-1][0])
	
class LSTM(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, hidden_size: int, n_layers: int = 2, output_activation: nn.Module = None):
		super(LSTM, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = max(1, n_layers)
		self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
		self.out = nn.Sequential(
			nn.Linear(hidden_size, out_channels),
			output_activation if output_activation is not None else nn.Identity(),
		)

	def forward(self, x: torch.tensor, edge_index: torch.tensor = None, edge_attr: torch.tensor = None) -> torch.tensor:
		batch_size, num_feats, seq_len = x.shape
		x = x.permute(0, 2, 1)  # (batch_size, seq_len, num_feats)
		lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
		return self.out(lstm_out[:, -1, :])  # Use the last time step's output