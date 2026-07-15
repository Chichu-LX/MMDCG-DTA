import os
import pickle
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time

from torch.utils.data import Dataset, DataLoader

# 从 Utils 文件夹下的 metrics.py 导入评估函数
from Utils.metrics import evaluate_metrics
# 从 Model 文件夹下的 MMDCG_DTA.py 导入模型
from Model.MMDCG_DTA import MMDCGDTAModel


#######################
# 自定义 Dataset
#######################
class GraphDataset(Dataset):
    """
    将数据字典 (key: sample_id, value: sample_dict) 转为 Dataset，以便使用 DataLoader。
    """

    def __init__(self, data_dict):
        # 这里将 data_dict 的所有值(样本)转为列表
        self.samples = list(data_dict.values())
        self.keys = list(data_dict.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 每次返回一个样本
        return self.samples[idx]


#######################
# 加载配置
#######################
def load_config(config_file="default.yaml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


#######################
# 划分训练/验证集
#######################
def split_train_val(dataset, val_ratio=0.1, seed=42):
    """
    将 dataset 列表按 val_ratio 划分训练集和验证集。
    """
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    val_size = int(len(dataset) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    return train_data, val_data


#######################
# 日志函数
#######################
def save_log(log_file, message):
    """
    在 log_file 中追加写一行，同时打印到控制台
    """
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)


#######################
# 主训练流程
#######################
def main():
    # 1. 检查数据文件
    if not os.path.exists("refined_set_graphs.pkl"):
        print("Error: refined_set_graphs.pkl 不存在，请先生成图数据。")
        return
    if not os.path.exists("core_set_graphs.pkl"):
        print("Error: core_set_graphs.pkl 不存在，请先生成图数据。")
        return

    # 2. 加载数据
    with open("refined_set_graphs.pkl", "rb") as f:
        refined_data = pickle.load(f)  # 用于训练(并划分验证)
    with open("core_set_graphs.pkl", "rb") as f:
        core_data = pickle.load(f)  # 用于测试

    # 将 refined_data 的样本转换成列表，并划分 10% 为验证集
    refined_samples = list(refined_data.values())
    train_list, val_list = split_train_val(refined_samples, val_ratio=0.1)

    # 将划分结果构造为 Dataset
    train_dataset = GraphDataset({i: s for i, s in enumerate(train_list)})
    val_dataset = GraphDataset({i: s for i, s in enumerate(val_list)})
    test_dataset = GraphDataset(core_data)

    # DataLoader（batch_size=1，shuffle=True 仅对训练集）
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. 加载超参数配置
    config = load_config("default.yaml")
    learning_rate = config["learning_rate"]
    max_epochs = config["max_epochs"]
    patience = config["patience"]  # 例如 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. 初始化模型、损失函数、优化器
    model = MMDCGDTAModel(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. 创建日志目录、模型保存目录
    model_save_dir = os.path.join("Log", "Models")
    log_dir = "Log"
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    best_test_rmse = float('inf')
    best_epoch = 0
    no_improve_epochs = 0

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        #######################
        # 训练阶段
        #######################
        model.train()
        train_losses = []
        for batch_data in train_loader:
            # batch_data 是一个样本（字典）
            sample = batch_data
            # 将其中的图对象移动到 GPU
            for key in sample:
                if key.endswith("graph"):
                    sample[key] = sample[key].to(device)

            # 前向传播
            y_pred = model(sample)  # shape [1]
            # 真实标签假设保存在 sample["label"]
            y_true = torch.tensor([sample["label"]], dtype=torch.float32, device=device)

            loss = criterion(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        #######################
        # 验证阶段
        #######################
        model.eval()
        val_losses = []
        all_val_true = []
        all_val_pred = []
        with torch.no_grad():
            for batch_data in val_loader:
                sample = batch_data
                for key in sample:
                    if key.endswith("graph"):
                        sample[key] = sample[key].to(device)
                y_pred = model(sample)
                y_true = torch.tensor([sample["label"]], dtype=torch.float32, device=device)
                val_loss = criterion(y_pred, y_true)
                val_losses.append(val_loss.item())
                all_val_true.append(y_true.item())
                all_val_pred.append(y_pred.item())

        avg_val_loss = np.mean(val_losses)
        val_metrics = evaluate_metrics(np.array(all_val_true), np.array(all_val_pred))

        #######################
        # 测试阶段
        #######################
        test_losses = []
        all_test_true = []
        all_test_pred = []
        with torch.no_grad():
            for batch_data in test_loader:
                sample = batch_data
                for key in sample:
                    if key.endswith("graph"):
                        sample[key] = sample[key].to(device)
                y_pred = model(sample)
                y_true = torch.tensor([sample["label"]], dtype=torch.float32, device=device)
                test_loss = criterion(y_pred, y_true)
                test_losses.append(test_loss.item())
                all_test_true.append(y_true.item())
                all_test_pred.append(y_pred.item())

        avg_test_loss = np.mean(test_losses)
        test_metrics = evaluate_metrics(np.array(all_test_true), np.array(all_test_pred))

        current_test_rmse = test_metrics["RMSE"]

        # 若测试集 RMSE 有所改善，则保存模型
        if current_test_rmse < best_test_rmse:
            best_test_rmse = current_test_rmse
            best_epoch = epoch
            no_improve_epochs = 0
            model_save_path = os.path.join(model_save_dir, f"best_model_epoch{epoch}.pt")
            torch.save(model.state_dict(), model_save_path)
        else:
            no_improve_epochs += 1

        # 记录日志
        log_msg = (
            f"Epoch {epoch}/{max_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Test Loss: {avg_test_loss:.4f} | "
            f"Test RMSE: {test_metrics['RMSE']:.4f} | "
            f"Test MAE: {test_metrics['MAE']:.4f} | "
            f"Test Pearson: {test_metrics['Pearson']:.4f} | "
            f"Test SD: {test_metrics['SD']:.4f}"
        )
        save_log(log_file, log_msg)

        # Early Stopping 判断：如果在测试集上 RMSE 连续 patience 个 epoch 无改善，则停止
        if no_improve_epochs >= patience:
            save_log(log_file, f"Early stopping at epoch {epoch}. No improvement in Test RMSE for {patience} epochs.")
            break

    total_time = time.time() - start_time
    final_msg = f"Training finished in {total_time:.2f}s. Best Test RMSE: {best_test_rmse:.4f} at epoch {best_epoch}."
    save_log(log_file, final_msg)


if __name__ == "__main__":
    main()
