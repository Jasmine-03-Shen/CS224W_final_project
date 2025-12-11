import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import GCN, GAT, GIN
from torch.nn import GRUCell, LSTMCell, TransformerEncoderLayer

class TGNNCell(nn.Module):
	"""
	T-GCN cell
	args:
		in_channels: int, input feature dimension
		hidden_size: int, hidden state dimension
		num_layers: int, number of GNN layers
		gnn_type: str, type of GNN ('gcn', 'gat', 'gin')
	model composition:
		self.gnn: GNN model (GCN, GAT, or GIN)
		GRU gates: update gate W_u, reset gate W_r, candidate hidden state W_c
	"""
	def __init__(self, in_channels: int, hidden_size: int, num_layers: int = 1, gnn_type: str = 'gcn'):
		super(TGNNCell, self).__init__()
		if gnn_type == 'gat':
			self.gnn = GAT(in_channels=in_channels, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=num_layers)
		elif gnn_type == 'gcn':
			self.gnn = GCN(in_channels=in_channels, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=num_layers)
		elif gnn_type == 'gin':
			self.gnn = GIN(in_channels=in_channels, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=num_layers)
		gate_in_dim = 2 * hidden_size + in_channels
		self.W_u = nn.Linear(gate_in_dim, hidden_size)
		self.W_r = nn.Linear(gate_in_dim, hidden_size)
		self.W_c = nn.Linear(gate_in_dim, hidden_size)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor, h: torch.tensor) -> tuple[
		torch.tensor, torch.tensor]:
		# get GNN output
		gnn_output = self.gnn(x, edge_index, edge_weight)
		gnn_output = F.sigmoid(gnn_output)
		# concat input, gnn output, and previous hidden state
		zu = torch.cat([x, gnn_output, h], dim=-1)
		# GRU gates
		u_t = torch.sigmoid(self.W_u(zu))
		r_t = torch.sigmoid(self.W_r(zu))
		zc = torch.cat([x, gnn_output, r_t * h], dim=-1)
		c_t = torch.tanh(self.W_c(zc))
		h_next = u_t * h + (1.0 - u_t) * c_t
		return h_next

class LSTMGCNCell(nn.Module):
	"""
	LSTM-GCN cell
	args:
		in_channels: int, input feature dimension
		hidden_size: int, hidden state dimension
		num_layers: int, number of GNN layers
		gnn_type: str, type of GNN ('gcn', 'gat', 'gin')
	model composition:
		self.gnn: GNN model (GCN, GAT, or GIN)
		LSTM gates: forget gate W_f, input gate W_i, output gate W_o, candidate cell W_c
	"""
	def __init__(self, in_channels: int, hidden_size: int, num_layers: int = 1, gnn_type: str = 'gcn'):
		super(LSTMGCNCell, self).__init__()
		if gnn_type == 'gat':
			self.gnn = GAT(in_channels=in_channels, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=num_layers)
		elif gnn_type == 'gcn':
			self.gnn = GCN(in_channels=in_channels, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=num_layers)
		elif gnn_type == 'gin':
			self.gnn = GIN(in_channels=in_channels, hidden_channels=hidden_size, out_channels=hidden_size, num_layers=num_layers)
		gate_in_dim = in_channels + 2 * hidden_size
		self.W_f = nn.Linear(gate_in_dim, hidden_size)
		self.W_i = nn.Linear(gate_in_dim, hidden_size)
		self.W_o = nn.Linear(gate_in_dim, hidden_size)
		self.W_c = nn.Linear(gate_in_dim, hidden_size)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor, h_c: tuple[
		torch.tensor, torch.tensor]) -> tuple[torch.tensor, torch.tensor]:
		h, c = h_c
		# get GNN output
		gnn_output = F.sigmoid(self.gnn(x, edge_index, edge_weight))
		# concat input, gnn output, and previous hidden state
		zu = torch.cat([x, gnn_output, h], dim=-1)
		# LSTM gates
		f_t = torch.sigmoid(self.W_f(zu))
		i_t = torch.sigmoid(self.W_i(zu))
		o_t = torch.sigmoid(self.W_o(zu))
		c_t = torch.tanh(self.W_c(zu))
		c_next = f_t * c + i_t * c_t
		h_next = o_t * torch.tanh(c_next)
		return h_next, c_next


class TGCN(nn.Module):
	"""
	TGCN
	args:
		in_channels: int, input feature dimension
		out_channels: int, output feature dimension
		hidden_size: int, hidden state dimension
		n_layers: int, number of T-GCN layers
		output_activation: nn.Module, activation function for output layer
		gnn_layers: int, number of GNN layers in each T-GCN cell
		gnn_type: str, type of GNN ('gcn', 'gat', 'gin')
	model composition:
		self.cells: list of T-GCN cells
		self.out: output layer
	"""
	def __init__(self, in_channels: int, out_channels: int, hidden_size: int, n_layers: int = 2, output_activation: nn.Module = None, gnn_layers: int = 1, gnn_type: str = 'gcn'):
		super(TGCN, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = max(1, n_layers)
		cells = []
		cells.append(TGNNCell(in_channels, hidden_size, num_layers=gnn_layers, gnn_type=gnn_type))
		for _ in range(1, self.n_layers):
			cells.append(TGNNCell(hidden_size, hidden_size, num_layers=gnn_layers, gnn_type=gnn_type))
		self.cells = nn.ModuleList(cells)
		self.out = nn.Sequential(
			nn.Linear(hidden_size, out_channels),
			output_activation if output_activation is not None else nn.Identity(),
		)
		self.gnn_type = gnn_type

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_attr: torch.tensor) -> torch.tensor:
		batch_size, num_feats, seq_len = x.shape
		edge_weight = edge_attr[:, -1] # corr
		hidden = [
			torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.n_layers)
		]
		# for each time step, update hidden states using T-GCN cells
		for t in range(seq_len):
			h = x[:, :, t] 
			for i, cell in enumerate(self.cells):
				h = cell(h, edge_index, edge_weight, hidden[i])
				hidden[i] = h
		return self.out(hidden[-1])

class A3TGCN(nn.Module):
	"""
	A3TGCN
	args:
		in_channels: int, input feature dimension
		out_channels: int, output feature dimension
		hidden_size: int, hidden state dimension
		n_layers: int, number of T-GCN layers
		output_activation: nn.Module, activation function for output layer
		gnn_type: str, type of GNN ('gcn', 'gat', 'gin')
		agg: str, aggregation method for attention output ('mean', 'sum')
	model composition:
		self.cells: list of T-GCN cells
		self.attention: multi-head attention layer
		self.out: output layer
	"""
	def __init__(self, in_channels: int, out_channels: int, hidden_size: int, n_layers: int = 2, output_activation: nn.Module = None, gnn_type: str = 'gcn', agg: str = 'mean'):
		super(A3TGCN, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = max(1, n_layers)
		cells = []
		cells.append(TGNNCell(in_channels, hidden_size, gnn_type=gnn_type))
		for _ in range(1, self.n_layers):
			cells.append(TGNNCell(hidden_size, hidden_size, gnn_type=gnn_type))
		self.cells = nn.ModuleList(cells)
		self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2)
		self.out = nn.Sequential(
			nn.Linear(hidden_size, out_channels),
			output_activation if output_activation is not None else nn.Identity(),
		)
		self.gnn_type = gnn_type
		self.agg = agg

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_attr: torch.tensor) -> torch.tensor:
		batch_size, num_feats, seq_len = x.shape
		edge_weight = edge_attr[:, -1] # corr
		hidden = [
			torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.n_layers)
		]
		h_final = torch.zeros(batch_size, seq_len, self.hidden_size).to(x.device)
		# for each time step, update hidden states using T-GCN cells
		for t in range(seq_len):
			h = x[:, :, t] 
			for i, cell in enumerate(self.cells):
				h = cell(h, edge_index, edge_weight, hidden[i])
				hidden[i] = h
			h_final[:, t, :] = h
		# apply attention layer
		h_final = h_final.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
		attn_output, attn_weights = self.attention(h_final, h_final, h_final)
		attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
		# aggregate attention output
		if self.agg == 'mean':
			h_attn = attn_output.mean(dim=1)  # (batch_size, hidden_size)
		elif self.agg == 'sum':
			h_attn = attn_output.sum(dim=1)  # (batch_size, hidden_size)
		elif self.agg == 'last':
			h_attn = attn_output[:, -1, :]  # (batch_size, hidden_size)
		return self.out(h_attn)

class LSTMGCN(nn.Module):
	"""
	LSTMGCN
	args:
		in_channels: int, input feature dimension
		out_channels: int, output feature dimension
		hidden_size: int, hidden state dimension
		n_layers: int, number of LSTM-GCN layers
		output_activation: nn.Module, activation function for output layer
		gnn_layers: int, number of GNN layers in each LSTM-GCN cell
		gnn_type: str, type of GNN ('gcn', 'gat', 'gin')
	model composition:
		self.cells: list of LSTM-GCN cells
		self.out: output layer
	"""
	def __init__(self, in_channels: int, out_channels: int, hidden_size: int, n_layers: int = 2, output_activation: nn.Module = None, gnn_layers: int = 1, gnn_type: str = 'gcn'):
		super(LSTMGCN, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = max(1, n_layers)
		cells = []
		cells.append(LSTMGCNCell(in_channels, hidden_size, num_layers=gnn_layers, gnn_type=gnn_type))
		for _ in range(1, self.n_layers):
			cells.append(LSTMGCNCell(hidden_size, hidden_size, num_layers=gnn_layers, gnn_type=gnn_type))
		self.cells = nn.ModuleList(cells)
		self.out = nn.Sequential(
			nn.Linear(hidden_size, out_channels),
			output_activation if output_activation is not None else nn.Identity(),
		)
		self.gnn_type = gnn_type

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_attr: torch.tensor) -> torch.tensor:
		batch_size, num_feats, seq_len = x.shape
		edge_weight = edge_attr[:, -1] # corr
		hidden = [
			(torch.zeros(batch_size, self.hidden_size).to(x.device), torch.zeros(batch_size, self.hidden_size).to(x.device)) for _ in range(self.n_layers)
		]
		# for each time step, update hidden states using LSTM-GCN cells
		for t in range(seq_len):
			h = x[:, :, t]  
			for i, cell in enumerate(self.cells):
				h, c = cell(h, edge_index, edge_weight, hidden[i])
				hidden[i] = (h, c)
		return self.out(hidden[-1][0])

class GRU(nn.Module):
	"""
	GRU
	args:
		in_channels: int, input feature dimension
		out_channels: int, output feature dimension
		hidden_size: int, hidden state dimension
		n_layers: int, number of GRU layers
		output_activation: nn.Module, activation function for output layer
	model composition:
		self.gru: GRU layer
		self.out: output layer
	"""
	def __init__(self, in_channels: int, out_channels: int, hidden_size: int, n_layers: int = 2, output_activation: nn.Module = None):
		super(GRU, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = max(1, n_layers)
		self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
		self.out = nn.Sequential(
			nn.Linear(hidden_size, out_channels),
			output_activation if output_activation is not None else nn.Identity(),
		)

	def forward(self, x: torch.tensor, edge_index: torch.tensor = None, edge_attr: torch.tensor = None) -> torch.tensor:
		batch_size, num_feats, seq_len = x.shape
		x = x.permute(0, 2, 1)  # (batch_size, seq_len, num_feats)
		h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(x.device)
		gru_out, h_n = self.gru(x, h_0)  # gru_out: (batch_size, seq_len, hidden_size)
		return self.out(gru_out[:, -1, :])  # Use the last time step's output
	
class LSTM(nn.Module):
	"""
	LSTM
	args:
		in_channels: int, input feature dimension
		out_channels: int, output feature dimension
		hidden_size: int, hidden state dimension
		n_layers: int, number of LSTM layers
		output_activation: nn.Module, activation function for output layer
	model composition:
		self.lstm: LSTM layer
		self.out: output layer
	"""
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
		h_0, c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(x.device), torch.zeros(self.n_layers, batch_size, self.hidden_size).to(x.device)
		lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))  # lstm_out: (batch_size, seq_len, hidden_size)
		return self.out(lstm_out[:, -1, :])  # Use the last time step's output
	
class GNN_only(nn.Module):
	"""
	GNN_only: only use GNN for prediction
	args:
		in_channels: int, input feature dimension
		out_channels: int, output feature dimension
		hidden_size: int, hidden state dimension
		n_layers: int, number of GNN layers
		gnn_type: str, type of GNN ('gcn', 'gat', 'gin')
		output_activation: nn.Module, activation function for output layer
		agg: str, aggregation method for input features ('mean', 'last', 'sum')
	model composition:
		self.gnn: GNN model (GCN, GAT, or GIN)
		self.out: output layer
	"""
	def __init__(self, in_channels: int, out_channels: int, hidden_size: int, n_layers: int = 2, gnn_type: str = 'gcn', output_activation: nn.Module = None, agg: str = 'mean'):
		super(GNN_only, self).__init__()
		if gnn_type == 'gat':
			self.gnn = GAT(in_channels=in_channels, hidden_channels=hidden_size, out_channels=out_channels, num_layers=n_layers)
		elif gnn_type == 'gcn':
			self.gnn = GCN(in_channels=in_channels, hidden_channels=hidden_size, out_channels=out_channels, num_layers=n_layers)
		elif gnn_type == 'gin':
			self.gnn = GIN(in_channels=in_channels, hidden_channels=hidden_size, out_channels=out_channels, num_layers=n_layers)
		self.out = nn.Sequential(
			nn.Linear(out_channels, out_channels),
			output_activation if output_activation is not None else nn.Identity(),
		 )
		self.agg = agg

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_attr: torch.tensor) -> torch.tensor:
		# aggregate input features over time dimension
		if self.agg == 'mean':
			x = x.mean(dim=-1)
		elif self.agg == 'last':
			x = x[:, :, -1]
		elif self.agg == 'sum':
			x = x.sum(dim=-1)
		gnn_out = self.gnn(x, edge_index, edge_attr[:, -1])
		return self.out(gnn_out)