import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Trỏ đường dẫn về thư mục gốc để nạp Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import Config

# 🌟 1. CẤU HÌNH THÔNG SỐ HYPERPARAMETERS
SEQ_LENGTH = 12       # Nhìn lại 3 giờ quá khứ (12 bước x 15 phút)
PREDICT_STEPS = 1     # Dự đoán 15 phút tương lai
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

# 🌟 2. XÂY DỰNG DATASET THEO CỬA SỔ TRƯỢT
class WasteDataset(Dataset):
    def __init__(self, data, seq_length):
        self.X = []
        self.y = []
        
        for bin_id in data['Bin_ID'].unique():
            bin_data = data[data['Bin_ID'] == bin_id]['Fill_Level_Percent'].values
            bin_data = bin_data / 100.0 # Chuẩn hóa về [0, 1]
            
            for i in range(len(bin_data) - seq_length - PREDICT_STEPS):
                self.X.append(bin_data[i : i + seq_length])
                self.y.append(bin_data[i + seq_length : i + seq_length + PREDICT_STEPS])
                
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 🌟 3. XÂY DỰNG KIẾN TRÚC MẠNG LSTM
class WasteForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(WasteForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, PREDICT_STEPS)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_model():
    print("🚀 ĐANG NẠP DỮ LIỆU TỪ FILE CSV...")
    # Lấy file waste_history.csv từ thư mục gốc
    csv_path = os.path.join(Config.BASE_DIR, "waste_history.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file dữ liệu tại: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 🌟 4. CHUẨN BỊ DATALOADER
    dataset = WasteDataset(df, SEQ_LENGTH)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 🌟 5. KHỞI TẠO MODEL & TRAINING
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WasteForecaster().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"🤖 BẮT ĐẦU HUẤN LUYỆN AI TIÊN TRI (Thiết bị: {device})...")
    print("-" * 50)
    
    loss_history = []
    for epoch in range(EPOCHS):
        model.train()
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
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Sai số Loss (MSE): {avg_loss:.6f}")

    # 🌟 6. LƯU BỘ NÃO VÀO ĐÚNG THƯ MỤC CHUẨN
    model_dir = os.path.join(Config.BASE_DIR, 'src', 'lstm_module')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'waste_forecaster.pth')
    torch.save(model.state_dict(), model_path)
    print("-" * 50)
    print(f"✅ HOÀN THÀNH HUẤN LUYỆN! Trọng số đã lưu tại: {model_path}")
    
    # Vẽ biểu đồ Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='purple', linewidth=2)
    plt.title('Biểu đồ Tiết giảm Sai số (Training Loss) của mạng LSTM')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    
    plot_path = os.path.join(model_dir, 'lstm_loss.png')
    plt.savefig(plot_path)
    print(f"🎨 Đã lưu biểu đồ Loss tại: {plot_path}")

if __name__ == "__main__":
    train_model()