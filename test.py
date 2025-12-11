import torch
from torch_geometric.loader import DataLoader
from dataset import SP500

from model import TGCN, LSTMGCN
from tqdm import tqdm
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--past_window', type=int, default=25, help='number of past time steps to use')
argparser.add_argument('--future_window', type=int, default=1, help='number of future time steps to predict')
argparser.add_argument('--hidden_size', type=int, default=16, help='hidden size of the model')
argparser.add_argument('--n_layers', type=int, default=1, help='number of layers in the model')
argparser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
argparser.add_argument('--gnn_type', type=str, default='gat', help='type of GCN model to use: gcn or gat')
argparser.add_argument('--temporal_model', type=str, default='LSTMGCN', help='type of temporal model to use: TGCN or LSTMGCN')
argparser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
argparser.add_argument('--device', type=str, default='cuda:4', help='device to use for training')
args = argparser.parse_args()

past_window = args.past_window
future_window = args.future_window
hidden_size = args.hidden_size
n_layers = args.n_layers
batch_size = args.batch_size
gnn_type = args.gnn_type
temporal_model_type = args.temporal_model
dataset = SP500(past_window=past_window)
print(dataset)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))

train_dataset, val_dataset, test_dataset = dataset[:train_size], dataset[train_size: train_size + val_size], dataset[train_size + val_size:]

print(f'train size: {len(train_dataset)} val size: {len(val_dataset)} test size: {len(test_dataset)}')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

in_channels = dataset[0].x.shape[-2]
out_channels = 1
if temporal_model_type == 'TGCN':
    model = TGCN(in_channels, out_channels, hidden_size, n_layers, gnn_type=gnn_type)
elif temporal_model_type == 'LSTMGCN':
    model = LSTMGCN(in_channels, out_channels, hidden_size, n_layers, gnn_type=gnn_type)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

criterion = torch.nn.MSELoss()

model.load_state_dict(torch.load(f'{temporal_model_type}_best_model_{n_layers}_{gnn_type}.pth'))
model = model.to(device)
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_dataloader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(out, batch.y)
        test_loss += loss.item() * batch.num_graphs
avg_test_loss = test_loss / len(test_dataloader.dataset)
print(f'Test Loss: {avg_test_loss:.4f}')