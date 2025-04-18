# run_test_only.py
import argparse, pickle, torch
import pandas as pd
from model import TransformerModel, LSTMModel 
from dataset import SlidingWindowDataset
from test import evaluate_model
from torch.utils.data import DataLoader
import os
import wandb
wandb.init(mode="disabled") 
def run_test(test_pickle, model_path, output_csv, model_type="transformer"):
    with open(test_pickle, "rb") as f:
        test_df = pickle.load(f)

    dataset = SlidingWindowDataset(test_df, window_size=64, return_indices=True)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 35
    if model_type == "transformer":
        model = TransformerModel(
            input_dim=input_dim, d_model=64, nhead=4,
            num_layers=2, dim_feedforward=128, dropout=0.1
        ).to(device)
    else:
        from model import LSTMModel
        model = LSTMModel(input_dim, hidden_dim=64, num_layers=2, dropout=0.1).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    evaluate_model(model, test_loader, device, threshold=0.5, save_csv_path=output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pickle", required=True)
    parser.add_argument("--model_path",  default="best_model.pt")
    parser.add_argument("--output_csv",  default="test_result.csv")
    parser.add_argument("--model",       choices=["transformer", "lstm"], default="transformer")
    args = parser.parse_args()

    run_test(
        test_pickle=args.test_pickle,
        model_path=args.model_path,
        output_csv=args.output_csv,
        model_type=args.model
    )
