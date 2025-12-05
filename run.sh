python train.py --past_window 25 --future_window 1 \
       --hidden_size 16 --n_layers 1 --batch_size 32 \
       --gnn_type gat --temporal_model TGCN \
       --epochs 200 --device cuda:0
python train.py --past_window 25 --future_window 1 \
       --hidden_size 16 --n_layers 1 --batch_size 32 \
       --gnn_type gin --temporal_model TGCN \
       --epochs 200 --device cuda:0
python train.py --past_window 25 --future_window 1 \
       --hidden_size 16 --n_layers 1 --batch_size 32 \
       --gnn_type gcn --temporal_model LSTMGCN \
       --epochs 200 --device cuda:0