import pandas as pd
import numpy as np
import datetime
import os

def generate_waste_data():
    print("🔄 Đang nội suy dữ liệu rác Đống Đa từ số liệu URENCO 4...")
    
    # --- THÔNG SỐ VĨ MÔ ---
    # Quận Đống Đa: 350.000 kg rác/ngày. 
    # Giả sử bản đồ SUMO của bạn có 50 điểm tập kết (POI).
    # Trung bình 1 điểm tập kết gánh: 350.000 / 50 = 7000 kg/ngày.
    AVG_WEIGHT_PER_POI = 7000 
    
    start_date = datetime.datetime(2025, 1, 1)
    dates = [start_date + datetime.timedelta(days=i) for i in range(180)] # Dữ liệu 6 tháng
    
    # Giả sử bạn cắm 4 thùng rác đại diện trên bản đồ SUMO để test trước
    bins = [
        {"id": "bin_van_phong_01", "zone": "Office"},
        {"id": "bin_van_phong_02", "zone": "Office"},
        {"id": "bin_dan_cu_01", "zone": "Residential"},
        {"id": "bin_dan_cu_02", "zone": "Residential"}
    ]
    
    dataset = []
    
    for date in dates:
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        for b in bins:
            if b["zone"] == "Office":
                # Văn phòng: Cuối tuần vắng người -> rác giảm mạnh (còn 20%)
                base_weight = AVG_WEIGHT_PER_POI * 0.2 if is_weekend else AVG_WEIGHT_PER_POI * 1.5
            else:
                # Dân cư: Cuối tuần ở nhà dọn dẹp -> rác tăng mạnh (lên 140%)
                base_weight = AVG_WEIGHT_PER_POI * 1.4 if is_weekend else AVG_WEIGHT_PER_POI * 0.8
                
            # Thêm nhiễu (Noise) ngẫu nhiên dao động +-15% để AI không học vẹt
            noise = np.random.uniform(-0.15, 0.15)
            final_weight = max(0, base_weight * (1 + noise))
            
            dataset.append({
                "Date": date.strftime("%Y-%m-%d"),
                "DayOfWeek": day_of_week,
                "IsWeekend": is_weekend,
                "Zone_Type": 1 if b["zone"] == "Office" else 0, # 1: Văn phòng, 0: Dân cư
                "Bin_ID": b["id"],
                "Waste_Weight_Kg": round(final_weight, 1)
            })

    df = pd.DataFrame(dataset)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dongda_waste_real_sim.csv", index=False)
    print("✅ Đã tạo xong: lstm_module/data/dongda_waste_real_sim.csv")

if __name__ == "__main__":
    generate_waste_data()