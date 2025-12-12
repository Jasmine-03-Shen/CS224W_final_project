# CS224W_final_project

## Data

https://drive.google.com/drive/folders/1viKTZOvkastV--JKocjphmGMZYd03EID?usp=sharing

## Dataset

features(x): ['Close', 'Volume', 'NormClose', 'DailyLogReturn', 'ALR1W', 'ALR2W', 'ALR1M', 'ALR2M', 'RSI', 'MACD']

y: 'DailyLogReturn'

past_window: 25

future_window: 1

number of edges: 31356

number of nodes: 489

edge_features: edge_attr[:, :n_sectors] -- If the endpoints of the edge come from the same industry sector, this part is the one-hot embedding of that industry sector; edge_attr[:, -1] -- If the correlation of weekly returns is greater than r_crit (0.5289, alpha=0.05), this part is the correlation of weekly returns.

## data.ipynb

This notebook prepares and analyzes S&P 500 stock market data. It gets 5 years of historical price data and computes financial features.

### 1. Data Collection
- Loads S&P 500 company metadata from `archive/sp500_companies.csv`
- Gets 5-year historical data using yfinance for each stock

### 2. Compute Features
Computes the following features for each stock:
- **Normalized Close**: Standardized closing price
- **Daily Log Return**: Logarithmic daily returns, normalized
- **Aggregated Log Returns**: Rolling window returns at multiple time scales
  - ALR1W: 1-week averaged log returns
  - ALR2W: 2-week averaged log returns
  - ALR1M: 1-month averaged log returns
  - ALR2M: 2-month averaged log returns
- **Technical Indicators**: RSI and MACD

### 3. Exploratory Data Analysis (EDA)
- Visualizes distributions of log returns (daily / aggregated)
- Computes correlation matrix of weekly returns (from Jan 2021)

### 4. Output
- Saves historical data to `./data/raw/history.csv`
- Saves weekly returns to `data/raw/weekly_returns.csv` 

## graph_feature.ipynb

This notebook constructs stock correlation graphs and prepares graph-based data for temporal graph neural networks. It builds adjacency matrices based on stock weekly return correlations and sector relationships.

### 1. Data Loading
- Loads S&P 500 company metadata and weekly stock returns
- Computes correlation matrix of weekly returns

### 2. Statistical Analysis
- Analyzes correlation distribution with normal approximation
- Calculates significance threshold ($\alpha = 0.05$) to filter meaningful correlations

### 3. Graph Construction
Builds two types of adjacency matrices:

**Correlation Graph (`adj_wr_corr`)**
- Edges created between stocks with statistically significant correlations
- Edge weights are correlation coefficients
- Weighted by magnitude of correlation

**Sector Graph (`adj_sector`)**
- Edges connect stocks in the same sector
- Binary adjacency matrix

**Merged Graph**
- Combines correlation and sector information
- Final edge attributes: sector one-hot encoding + correlation value

### 4. Data Reshaping
- Reshapes time-series features into $[N_{nodes}, T_{timesteps}, N_{features}]$ tensor format
- Features include: `['Close', 'Volume', 'NormClose', 'DailyLogReturn', 'ALR1W', 'ALR2W', 'ALR1M', 'ALR2M', 'RSI', 'MACD']`

### 5. PyTorch Data Serialization
Saves the following files for model training:
- `edge_index.pt`: Graph connectivity (full dataset)
- `edge_attr.pt`: Edge attributes including sector info and correlations
- `feature.pt`: Node features across time steps

## model.py

This module contains various temporal graph neural network architectures for stock price prediction. Also contains GRU, LSTM, and GNN only models for comparison

### Cell Modules (Base Building Blocks)

**TGNNCell**
- Combines a GNN layer with GRU gating mechanism
- Supports GCN, GAT, or GIN as the GNN backbone

**LSTMGCNCell**
- Combines a GNN layer with LSTM gating mechanism
- Supports GCN, GAT, or GIN as the GNN backbone

### Temporal Graph Models

**TGCN** - Temporal Graph Convolutional Network
- Stacks multiple TGNNCell layers (usually only use 1 layer)
- Uses hidden state from last time step for prediction

**A3TGCN** - Attention-based Adaptive 3D Temporal GCN
- Extends TGCN with multi-head attention mechanism
- Applies attention after TGCN cells
- Aggregates output via mean, sum, or last (only use the embedding of the last time step) pooling

**LSTMGCN** - LSTM-Graph Convolutional Network
- Stacks multiple LSTMGCNCell layers (usually only use 1 layer)
- Uses hidden state from last time step for prediction
