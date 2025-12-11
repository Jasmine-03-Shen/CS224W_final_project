import torch
from torch_geometric.data import Dataset, Data

class SP500(Dataset):
    def __init__(self, 
                 past_window=25,
                 fut_window=1,
                 feat_file = './data/processed/feature.pt',
                 label_file = './data/processed/daily_log_return_y.pt',
                 edge_index_file = './data/processed/edge_index.pt',
                 edge_attr_file = './data/processed/edge_attr.pt',
                 root = './data/processed', 
                 transform = None, 
                 pre_transform = None, 
                 pre_filter = None, 
                 log = True, 
                 force_reload = False):
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.past_window = past_window
        self.fut_window = fut_window
        self.feats = torch.load(feat_file)
        self.label = torch.load(label_file)
        self.edge_index = torch.load(edge_index_file)
        self.edge_attr = torch.load(edge_attr_file)
        print(f"Dataset loaded with features shape {self.feats.shape}, label shape {self.label.shape}, edge_index shape {self.edge_index.shape}, edge_attr shape {self.edge_attr.shape}")
    
    def len(self):
        return self.feats.shape[-1] - self.past_window - self.fut_window
    
    def get(self, idx: int):
        # for x, use all features ('NormClose', 'DailyLogReturn', 'ALR1W', 'ALR2W', 'ALR1M', 'ALR2M', 'RSI', 'MACD') except the first two ('Close', 'Volume')
        # for y, use 'NormClose'
        return Data(x=self.feats[:, 2:, idx:idx+self.past_window], edge_index=self.edge_index, edge_attr=self.edge_attr, y=self.feats[:, 2, idx+self.past_window:idx+self.past_window+self.fut_window])
    
if __name__ == "__main__":
    dataset = SP500(feat_file='./data/processed/feature_small.pt',
                label_file='./data/processed/daily_log_return_y_small.pt',
                edge_attr_file='./data/processed/edge_attr_small.pt',
                edge_index_file='./data/processed/edge_index_small.pt')
    
    print(f"length {len(dataset)}")
    
    first_data = dataset[0]
    print(first_data)
    print('x:', first_data.x.shape)
    print('edge_index:', first_data.edge_index.shape)
    print('edge_attr:', first_data.edge_attr.shape)
    print('y:', first_data.y.shape)