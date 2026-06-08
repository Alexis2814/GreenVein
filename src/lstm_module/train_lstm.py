import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
SEQ_LENGTH = 7      # Nhìn lại 7 ngày quá khứ
MAX_CAPACITY = 15000.0 # Dùng để chuẩn hóa dữ liệu
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 0.002

class WasteDataset(Dataset):
    def __init__(self, df, seq_length):
        self.X, self.y = [], []
        
        for bin_id in df['Bin_ID'].unique():
            bin_data = df[df['Bin_ID'] == bin_id]['Waste_Weight_Kg'].values
            # 🌟 BÍ QUYẾT: Chuẩn hóa dữ liệu về [0, 1] để LSTM học tốt nhất
            bin_data_norm = bin_data / MAX_CAPACITY 
            
            for i in range(len(bin_data_norm) - seq_length - 1):
                self.X.append(bin_data_norm[i : i + seq_length])
                self.y.append(bin_data_norm[i + seq_length])
                
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(-1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class WasteLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(WasteLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model():
    print("🚀 BƯỚC 2: Đang bắt đầu huấn luyện mạng LSTM...")
    df = pd.read_csv("data/dongda_waste_real.csv")
    
    dataset = WasteDataset(df, SEQ_LENGTH)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WasteLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_history = []
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.6f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/lstm_model.pth")
    print("✅ Đã lưu trọng số (Bộ não AI) tại: weights/lstm_model.pth")

if __name__ == "__main__":
    train_model()