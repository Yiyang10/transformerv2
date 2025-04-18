import torch
import numpy as np
from test import evaluate_model  # 调用 test.py 中的评估函数
import wandb

def train_model(model, train_loader, test_loader,
                criterion, optimizer, device,
                num_epochs=12, threshold=0.5,
                save_best=True):
    """
    若 save_best=True 则以 F1 为监控指标保存最佳权重，
    函数结尾返回回滚后的最佳模型；否则返回最后 epoch 的权重。
    """
    best_f1 = -1
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for X, y, *_ in train_loader:      # 兼容有无 seg_idx/offset
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)              # [B, seq_len, 1]
            logits = logits.squeeze(-1)    # [B, seq_len]

            loss = criterion(logits, y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        wandb.log({"Train Loss": avg_loss, "epoch": epoch + 1})
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_loss:.4f}")

        # —— 在线评估
        precision, recall, f1 = evaluate_model(
            model, test_loader, device,
            threshold,
            save_csv_path=None,
            epoch=epoch
        )
        if save_best and f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()

    # —— 训练循环结束
    if save_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (F1={best_f1:.4f})")
    return model
