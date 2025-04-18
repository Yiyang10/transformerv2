import torch
import torch.nn as nn
from dataset import SlidingWindowDataset
from model import TransformerModel  
from model import LSTMModel
from train import train_model
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from focal_loss import FocalLoss
import wandb
from test import evaluate_model

# def get_data_loaders_by_segment(
#     pickle_path, 
#     sequence_length=32, 
#     batch_size=32, 
#     train_ratio=0.8, 
#     shuffle=True, 
#     random_seed=42
# ):
#     with open(pickle_path, "rb") as f:
#         big_df = pickle.load(f)
#     if not isinstance(big_df, pd.DataFrame):
#         raise ValueError("The loaded pickle does not contain a DataFrame.")

#     segments = big_df["segment_name"].unique()
#     np.random.seed(random_seed)
#     np.random.shuffle(segments)
#     train_count = int(len(segments) * train_ratio)
#     train_segments = segments[:train_count]
#     test_segments = segments[train_count:]

#     train_df = big_df[big_df["segment_name"].isin(train_segments)].reset_index(drop=True)
#     test_df = big_df[big_df["segment_name"].isin(test_segments)].reset_index(drop=True)

#     train_dataset = SlidingWindowDataset(train_df, window_size=sequence_length)
#     test_dataset = SlidingWindowDataset(test_df, window_size=sequence_length)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader
def get_data_loaders_by_segment(
    train_pickle_path,
    test_pickle_path,
    sequence_length=32, 
    batch_size=32, 
    shuffle=True
):
    # 加载训练数据
    with open(train_pickle_path, "rb") as f:
        train_df = pickle.load(f)
    train_dataset = SlidingWindowDataset(train_df, window_size=sequence_length, return_indices=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # 加载测试数据
    with open(test_pickle_path, "rb") as f:
        test_df = pickle.load(f)
    test_dataset = SlidingWindowDataset(test_df, window_size=sequence_length, return_indices=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pickle", required=True,
                        help="训练集 DataFrame pickle 路径")
    parser.add_argument("--test_pickle",  required=True,
                        help="测试集  DataFrame pickle 路径")
    parser.add_argument("--seq_len",      default=64,  type=int)
    parser.add_argument("--batch_size",   default=32,  type=int)
    parser.add_argument("--epochs",       default=12,  type=int)
    parser.add_argument("--lr",           default=5e-4, type=float)
    parser.add_argument("--model",        choices=["transformer", "lstm"],
                        default="transformer")
    parser.add_argument("--out_csv",      default="final_test.csv")
    args = parser.parse_args()

    # —— W&B
    wandb.init(project="forceboard_anomaly",
               name=f"{args.model}-run", config=vars(args))

    # —— device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # —— DataLoaders
    train_loader, test_loader = get_data_loaders_by_segment(
        args.train_pickle, args.test_pickle,
        seq_len=args.seq_len, batch_size=args.batch_size)

    # —— Model
    input_dim = 35
    if args.model == "transformer":
        model = TransformerModel(
            input_dim, d_model=64, nhead=4,
            num_layers=2, dim_feedforward=128,
            dropout=0.1
        ).to(device)
    else:
        from models import LSTMModel
        model = LSTMModel(input_dim, hidden_dim=64,
                          num_layers=2, dropout=0.1).to(device)

    wandb.watch(model, log="all", log_freq=100)

    # —— Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # —— Train
    model = train_model(model, train_loader, test_loader,
                        criterion, optimizer, device,
                        num_epochs=args.epochs,
                        threshold=0.5,
                        save_best=True)

    # —— Final full‑test
    evaluate_model(model, test_loader, device,
                   threshold=0.5,
                   save_csv_path=args.out_csv,
                   epoch=args.epochs)

    wandb.finish()

if __name__ == "__main__":
    main()
