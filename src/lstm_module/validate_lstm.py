import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. BÊ NGUYÊN KIẾN TRÚC MẠNG TỪ FILE TRAIN SANG ĐỂ LOAD TRỌNG SỐ
class WasteForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(WasteForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1) # PREDICT_STEPS = 1

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def validate_model():
    print("🔍 ĐANG KHỞI ĐỘNG BỘ NÃO TIÊN TRI...")
    
    # Cấu hình
    seq_length = 12 # 3 giờ (12 bước x 15 phút)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. LOAD MÔ HÌNH
    model = WasteForecaster().to(device)
    model_path = '../models_lstm/waste_forecaster.pth'
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy file model. Hãy chạy train_lstm.py trước!")
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() # Chuyển sang chế độ suy luận (không train nữa)

    # 3. LOAD DỮ LIỆU
    df = pd.read_csv("waste_history.csv")
    
    # Lấy thử 1 thùng rác Khu Dân Cư (Residential) để test
    # Thay bằng ID thùng rác của fen nếu cần
    test_bin_id = df[df['Zone_Type'] == 'residential']['Bin_ID'].iloc[0]
    bin_data = df[df['Bin_ID'] == test_bin_id]['Fill_Level_Percent'].values
    
    # Lấy 48 giờ (192 steps) + 12 steps mồi ban đầu
    test_data_raw = bin_data[:192 + seq_length] 
    test_data_norm = test_data_raw / 100.0 # Chuẩn hóa
    
    actuals = []
    predictions = []

    print(f"📊 Đang dự báo cho thùng rác ID: {test_bin_id} (Khu Dân Cư)...")

    # 4. TRƯỢT CỬA SỔ ĐỂ DỰ BÁO
    with torch.no_grad(): # Tắt tính đạo hàm cho nhanh
        for i in range(len(test_data_norm) - seq_length):
            # Cắt 12 bước quá khứ
            window = test_data_norm[i : i + seq_length]
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            
            # AI dự báo bước tiếp theo
            pred = model(window_tensor).item()
            
            # Lưu lại (nhân 100 để trả về % rác thực tế)
            predictions.append(pred * 100.0)
            actuals.append(test_data_norm[i + seq_length] * 100.0)

    # 5. VẼ BIỂU ĐỒ SO SÁNH
    plt.figure(figsize=(14, 6))
    plt.style.use('ggplot')
    
    # Trục X là thời gian (Giờ)
    x_axis = np.arange(len(actuals)) / 4 
    
    plt.plot(x_axis, actuals, label='Thực tế (Actual)', color='#2ca02c', linewidth=2.5)
    plt.plot(x_axis, predictions, label='AI Dự báo (Predicted)', color='#d62728', linestyle='--', linewidth=2)
    
    plt.title(f'KIỂM CHỨNG ĐỘ CHÍNH XÁC CỦA MẠNG LSTM (Thùng: {test_bin_id})', fontsize=16, fontweight='bold')
    plt.xlabel('Thời gian (Giờ)', fontsize=12)
    plt.ylabel('Mức độ đầy rác (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    
    output_img = '../models_lstm/lstm_validation.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ Đã vẽ xong! Biểu đồ lưu tại: {output_img}")
    plt.show()

if __name__ == "__main__":
    validate_model()