import pandas as pd
import numpy as np
import datetime
import os

def generate_perfect_dataset():
    print("🔄 BƯỚC 1: Đang tạo Dataset giả lập chuẩn IoT...")
    
    # Dữ liệu 1 năm (365 ngày) để LSTM có đủ chu kỳ học
    dates = [datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i) for i in range(365)]
    
    bins = [
        {"id": "bin_van_phong_01", "zone": 1}, # 1: Văn phòng
        {"id": "bin_van_phong_02", "zone": 1},
        {"id": "bin_dan_cu_01", "zone": 0},    # 0: Dân cư
        {"id": "bin_dan_cu_02", "zone": 0}
    ]
    
    MAX_CAPACITY = 15000.0 # 15 Tấn mỗi khu vực
    data = []
    
    for date in dates:
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        for b in bins:
            if b["zone"] == 1: # Văn phòng
                base_percent = np.random.uniform(10, 25) if is_weekend else np.random.uniform(70, 95)
            else: # Dân cư
                base_percent = np.random.uniform(80, 98) if is_weekend else np.random.uniform(30, 50)
                
            # Thêm nhiễu ngẫu nhiên
            final_percent = np.clip(base_percent + np.random.normal(0, 5), 0, 100)
            real_kg = (final_percent / 100.0) * MAX_CAPACITY
            
            data.append({
                "Date": date.strftime("%Y-%m-%d"),
                "DayOfWeek": day_of_week,
                "IsWeekend": is_weekend,
                "Zone_Type": b["zone"],
                "Bin_ID": b["id"],
                "Waste_Weight_Kg": round(real_kg, 1)
            })
            
    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dongda_waste_real.csv", index=False)
    print("✅ Xong Bước 1! File dữ liệu đã lưu tại: data/dongda_waste_real.csv")

if __name__ == "__main__":
    generate_perfect_dataset()