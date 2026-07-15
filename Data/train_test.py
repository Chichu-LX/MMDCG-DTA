import os
import pickle
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import math
from torch.utils.data import Dataset, DataLoader

# 尝试导入评估指标
try:
    from metrics import evaluate_metrics
except ImportError:
    def evaluate_metrics(y_true, y_pred):
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Pearson 增强版
        vx = y_true - np.mean(y_true)
        vy = y_pred - np.mean(y_pred)
        
        # 检查方差是否为 0 (即是否为常数)
        std_x = np.std(y_true)
        std_y = np.std(y_pred)
        
        if std_x < 1e-6 or std_y < 1e-6:
            # 如果模型预测输出恒定值，相关性无法计算，设为 0
            pearson = 0.0
        else:
            pearson = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-8)
            
        # SD
        error = y_true - y_pred
        sd = np.std(error)
        return {"RMSE": rmse, "MAE": mae, "Pearson": pearson, "SD": sd}

# 导入模型
from MMDCG_DTA import MMDCGDTAModel


#######################
# 配置加载
#######################
def load_config(config_file="default.yaml"):
    # 这里的默认值只在找不到文件时生效
    default_config = {
        "batch_size": 1, "learning_rate": 0.001, "max_epochs": 100, "patience": 30,
        "d_atom": 8, "d_res": 8, "d_sub": 6,
        "l_intra": 3, "l_inter": 3, "l_atom": 2, "l_sub": 2,
        "embedding_dim": 128, "inter_negative_slope": 0.2, "sub_x_dim": 5,
        "raw_atom_dim": 5, "prot_res_dim": 1,
        "dropout_rate": 0.1, 
        "weight_decay": 1e-5 
    }
    
    if not os.path.exists(config_file):
        print(f"Warning: {config_file} not found. Using default hyperparameters.")
        return default_config
        
    with open(config_file, "r") as f:
        loaded_config = yaml.safe_load(f)
        
    # 强制覆盖默认配置
    final_config = default_config.copy()
    final_config.update(loaded_config)
    
    return final_config


#######################
# 自定义 Dataset
#######################
class GraphDataset(Dataset):
    def __init__(self, data_dict, dataset_name="unknown"):
        self.samples = []
        self.dataset_name = dataset_name
        self.invalid_samples = []
        self.empty_graph_samples = []
        
        # 键名必须与 build_graph_dataset.py 生成的一致
        self.graph_keys = [
            'ligand_atom_graph', 
            'protein_atom_graph', 
            'atom_interaction_graph', 
            'ligand_fragment_graph', 
            'protein_residue_graph', 
            'substructure_interaction_graph'
        ]
        
        print(f"Processing {dataset_name} dataset...")
        
        for key, val in data_dict.items():
            if val.get('label') is None:
                continue
            
            label = val.get('label')
            if label is None or math.isnan(label) or math.isinf(label):
                self.invalid_samples.append(key)
                continue
            
            has_empty_graph = False
            for g_key in self.graph_keys:
                if g_key in val:
                    graph = val[g_key]
                    if hasattr(graph, 'num_nodes') and graph.num_nodes() == 0:
                        self.empty_graph_samples.append((key, g_key))
                        has_empty_graph = True
                        break 
            
            if has_empty_graph:
                continue
            
            self.samples.append(val)
        
        print(f"  {dataset_name}: Loaded {len(self.samples)} valid samples")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


#######################
# Collate Function
#######################
def collate_single_sample(batch):
    return batch[0]


#######################
# 模型输出检查
#######################
def check_model_output(output, sample_idx=None):
    if torch.isnan(output).any() or torch.isinf(output).any():
        if sample_idx is not None:
            print(f"Warning: Model output contains NaN/Inf for sample {sample_idx}")
        return False
    return True


#######################
# 训练/验证/测试 封装函数 (带梯度累积)
#######################
def run_epoch(model, loader, optimizer, criterion, device, is_train=True, epoch=None):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []
    
    step_count = 0
    skipped_samples = 0
    clip_value = 5.0
    
    # =================== 重要：梯度累积配置 ===================
    accumulation_steps = 32  # 每 accumulation_steps 个样本更新一次参数
    # ========================================================

    graph_keys = [
        'ligand_atom_graph', 
        'protein_atom_graph', 
        'atom_interaction_graph', 
        'ligand_fragment_graph', 
        'protein_residue_graph', 
        'substructure_interaction_graph'
    ]

    context = torch.enable_grad() if is_train else torch.no_grad()
    
    # 在循环开始前清零梯度
    if is_train:
        optimizer.zero_grad()

    with context:
        for batch_idx, sample in enumerate(loader):
            try:
                if sample.get('label') is None: 
                    skipped_samples += 1
                    continue
                
                y_true_val = sample["label"]
                y_true = torch.tensor([y_true_val], dtype=torch.float32, device=device)
                
                # 确保图数据在 GPU 上
                for g_key in graph_keys:
                    if g_key in sample:
                        sample[g_key] = sample[g_key].to(device)
                
                # 3. 前向传播
                y_pred = model(sample)
                
                if not check_model_output(y_pred, batch_idx):
                    skipped_samples += 1
                    continue
                
                # 4. 计算损失
                loss = criterion(y_pred.view(-1), y_true.view(-1))
                
                # 5. 反向传播 (关键：梯度累积)
                if is_train:
                    # 除以 accumulation_steps，得到平均梯度
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    # 只有在累积了足够步数后，才更新参数
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                        # 更新参数
                        optimizer.step()
                        # 清空梯度
                        optimizer.zero_grad()

                # 记录数据 (原始 loss, 未除以 accumulation_steps)
                loss_val = loss.item() * accumulation_steps
                
                if not math.isnan(loss_val) and not math.isinf(loss_val):
                    total_loss += loss_val
                    all_true.append(y_true.item())
                    all_pred.append(y_pred.item())
                else:
                    skipped_samples += 1
                    
                step_count += 1
                
            except Exception as e:
                skipped_samples += 1
                if epoch is not None and batch_idx < 3:
                    print(f"Epoch {epoch}: Error processing sample {batch_idx}: {e}")
                continue

    avg_loss = total_loss / max(1, step_count) if step_count > 0 else float('nan')
    
    metrics = {}
    if len(all_true) > 0:
        all_true_array = np.array(all_true)
        all_pred_array = np.array(all_pred)
        
        # 修正 Pearson 系数计算：增加极小值保护
        if np.any(np.isnan(all_true_array)) or np.any(np.isnan(all_pred_array)):
            metrics = {"RMSE": float('nan'), "MAE": float('nan'), "Pearson": 0.0, "SD": float('nan')}
        else:
            metrics = evaluate_metrics(all_true_array, all_pred_array)
    else:
        metrics = {"RMSE": float('nan'), "MAE": float('nan'), "Pearson": 0.0, "SD": float('nan')}
        
    return avg_loss, metrics, skipped_samples


#######################
# 初始化权重
#######################
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


#######################
# 主程序
#######################
def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    refined_path = "refined_set_graphs.pkl"
    core_path = "core_set_graphs.pkl"
    
    if not os.path.exists(refined_path) or not os.path.exists(core_path):
        print(f"Error: 数据文件缺失。")
        return

    print("Loading datasets...")
    with open(refined_path, "rb") as f:
        refined_data = pickle.load(f)
    with open(core_path, "rb") as f:
        core_data = pickle.load(f)
    
    train_samples_list = list(refined_data.values())
    core_samples_list = list(core_data.values())
    random.shuffle(core_samples_list)
    
    split_point = int(len(core_samples_list) * 0.8)
    val_samples_list = core_samples_list[:split_point]
    test_samples_list = core_samples_list[split_point:]
    
    train_dataset = GraphDataset({i: s for i, s in enumerate(train_samples_list)}, "Training")
    val_dataset = GraphDataset({i: s for i, s in enumerate(val_samples_list)}, "Validation")
    test_dataset = GraphDataset({i: s for i, s in enumerate(test_samples_list)}, "Testing")
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                              collate_fn=collate_single_sample, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            collate_fn=collate_single_sample, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                             collate_fn=collate_single_sample, pin_memory=True, num_workers=0)

    # 加载配置
    config = load_config("default.yaml")
    
    # 强制覆盖维度设置以匹配 featurize.py
    config["raw_atom_dim"] = 5 
    config["sub_x_dim"] = 5
    config["prot_res_dim"] = 1
    
    print(f"\nModel Config: {config}")
    
    model = MMDCGDTAModel(config).to(device)
    model.apply(init_weights)
    
    # 使用 config 里的学习率和权重衰减
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    criterion = nn.MSELoss()

    log_dir = "Log"
    model_save_dir = os.path.join(log_dir, "Models")
    os.makedirs(model_save_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_log.txt")
    
    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log(f"Start Training on {device}")
    
    best_val_rmse = float('inf')
    best_test_rmse_at_val = float('inf')
    patience_counter = 0

    for epoch in range(1, config["max_epochs"] + 1):
        start_t = time.time()
        
        train_loss, train_metrics, train_skipped = run_epoch(model, train_loader, optimizer, criterion, device, is_train=True, epoch=epoch)
        val_loss, val_metrics, val_skipped = run_epoch(model, val_loader, optimizer, criterion, device, is_train=False, epoch=epoch)
        test_loss, test_metrics, test_skipped = run_epoch(model, test_loader, optimizer, criterion, device, is_train=False, epoch=epoch)
        
        if not math.isnan(val_loss):
            scheduler.step(val_loss)
        
        epoch_time = time.time() - start_t
        
        val_rmse = val_metrics.get('RMSE', float('inf'))
        test_rmse = test_metrics.get('RMSE', float('inf'))
        val_pearson = val_metrics.get('Pearson', 0.0)

        log_msg = (f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s | "
                   f"Train Loss: {train_loss:.4f} | "
                   f"Val RMSE: {val_rmse:.4f} (Rp:{val_pearson:.3f}) | "
                   f"Test RMSE: {test_rmse:.4f}")
        
        if train_skipped > 0: log_msg += f" | Skp: {train_skipped}"
        
        log(log_msg)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_test_rmse_at_val = test_rmse
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model.pt"))
            log(f"  >>> Best Model Saved! (Val RMSE: {best_val_rmse:.4f}, Test RMSE: {best_test_rmse_at_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                log("Early Stopping.")
                break
    
    log(f"Best Val RMSE: {best_val_rmse:.4f} (Test: {best_test_rmse_at_val:.4f})")

if __name__ == "__main__":
    main()