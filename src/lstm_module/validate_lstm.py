import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import kiến trúc mạng từ file train
from train_lstm import WasteLSTM, SEQ_LENGTH, MAX_CAPACITY

def validate_and_plot():
    print("🔍 BƯỚC 3: Đang kiểm chứng và vẽ đồ thị...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Nạp bộ não
    model = WasteLSTM().to(device)
    model.load_state_dict(torch.load("weights/lstm_model.pth", map_location=device, weights_only=True))
    model.eval()
    
    # 2. Đọc dữ liệu
    df = pd.read_csv("data/dongda_waste_real.csv")
    test_bin = "bin_van_phong_01" # Test thùng rác khu văn phòng
    
    bin_data = df[df['Bin_ID'] == test_bin]['Waste_Weight_Kg'].values
    bin_data_norm = bin_data / MAX_CAPACITY
    
    actuals, predictions = [], []
    
    # Lấy 60 ngày để test
    test_days = 60
    with torch.no_grad():
        for i in range(test_days):
            window = bin_data_norm[i : i + SEQ_LENGTH]
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            
            pred_norm = model(window_tensor).item()
            
            # Nhân ngược lại để ra số Kg
            predictions.append(pred_norm * MAX_CAPACITY)
            actuals.append(bin_data_norm[i + SEQ_LENGTH] * MAX_CAPACITY)

    # 3. Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    plt.plot(actuals, label='Khối lượng Thực tế', color='green', linewidth=2, marker='o', alpha=0.7)
    plt.plot(predictions, label='LSTM Dự đoán', color='red', linestyle='--', linewidth=2, marker='x')
    
    plt.title(f'KIỂM CHỨNG ĐỘ CHÍNH XÁC LSTM (Khu vực Văn Phòng)', fontsize=14, fontweight='bold')
    plt.xlabel('Thời gian (Ngày)', fontsize=12)
    plt.ylabel('Khối lượng rác (Kg)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("lstm_validation_result.png", dpi=300, bbox_inches='tight')
    print("✅ Đã vẽ xong! Hãy mở file 'lstm_validation_result.png' để xem kết quả tuyệt vời nhé!")

if __name__ == "__main__":
    validate_and_plot()