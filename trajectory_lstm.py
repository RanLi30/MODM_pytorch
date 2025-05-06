import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class TrajectoryDataset(Dataset):
    def __init__(self, csv_file, seq_len=10, normalize=True):
        self.seq_len = seq_len
        self.normalize = normalize
        df = pd.read_csv(csv_file, header=None)

        # 将每行字符串转换为 ndarray
        parsed_data = df[0].apply(lambda s: np.fromstring(s.strip('[]'), sep=' '))
        coords = np.stack(parsed_data.values)

        self.positions = coords[:, :2].astype(np.float32)

        if self.normalize:
            self.data_min = self.positions.min(axis=0)
            self.data_max = self.positions.max(axis=0)
            self.data_range = self.data_max - self.data_min
            self.positions = (self.positions - self.data_min) / self.data_range
        else:
            self.data_min = 0
            self.data_range = 1

        self.samples = []
        for i in range(len(self.positions) - seq_len):
            seq = self.positions[i:i + seq_len]
            label = self.positions[i + seq_len]
            self.samples.append((seq, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def denormalize(self, norm_coords):
        return norm_coords * self.data_range + self.data_min

class LSTMPositionPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

def create_dataloaders(csv_path, seq_len=10, batch_size=64, normalize=True):
    dataset = TrajectoryDataset(csv_path, seq_len=seq_len, normalize=normalize)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, dataset

def predict_next_bounding_box(bbox_seq, model_path="trajectory_model.pth", device="cpu"):
    """
    使用已训练的 LSTM 模型预测下一帧的 bounding box（基于中心点轨迹）

    参数：
        bbox_seq: List[np.array] 或 np.ndarray，形状为 (t, 4)，每个元素是 [x, y, w, h]
        model_path: 已训练模型的路径
        device: 'cpu' 或 'cuda'

    返回：
        predicted_bbox: [x, y, w, h]，预测的 bounding box
    """
    bbox_seq = np.array(bbox_seq, dtype=np.float32)
    if bbox_seq.shape[1] != 4:
        raise ValueError("每个 bounding box 应包含 [x, y, w, h] 共4项")

    centers = bbox_seq[:, :2] + bbox_seq[:, 2:] / 2
    data_min = centers.min(axis=0)
    data_max = centers.max(axis=0)
    data_range = data_max - data_min
    norm_centers = (centers - data_min) / data_range

    input_seq = torch.tensor(norm_centers, dtype=torch.float32).unsqueeze(0).to(device)

    model = LSTMPositionPredictor()
    load_model(model, model_path, device=device)
    model.eval()

    with torch.no_grad():
        pred_norm = model(input_seq)
        pred_norm = pred_norm.squeeze(0).cpu().numpy()

    p2 = pred_norm * data_range + data_min
    p1_last = centers[-1]
    p3 = (p1_last + p2) / 2

    last_w, last_h = bbox_seq[-1, 2], bbox_seq[-1, 3]
    x3 = p3[0] - last_w / 2
    y3 = p3[1] - last_h / 2

    predicted_bbox = [x3, y3, last_w, last_h]
    return predicted_bbox
