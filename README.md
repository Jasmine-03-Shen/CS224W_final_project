# CS224W_final_project

## Dataset

features(x): ['Close', 'Volume', 'NormClose', 'DailyLogReturn', 'ALR1W', 'ALR2W', 'ALR1M', 'ALR2M', 'RSI', 'MACD']
y: 'DailyLogReturn'
past_window: 25
future_window: 1
number of edges: 31356
number of nodes: 489
edge_features: edge_attr[:, :n_sectors] -- If the endpoints of the edge come from the same industry sector, this part is the one-hot embedding of that industry sector; edge_attr[:, -1] -- If the correlation of weekly returns is greater than r_crit (0.5289, alpha=0.05), this part is the correlation of weekly returns.