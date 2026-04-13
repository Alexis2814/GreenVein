import json
import csv
import os
from waste_generator import RealWasteGenerator

def generate_7_days_dataset():
    print("🌟 BẮT ĐẦU CÀY DỮ LIỆU SINH RÁC TRONG 7 NGÀY (GIẢ LẬP)...")
    
    # 1. Đọc file cấu hình zone_config.json mà fen vừa làm ở Task 1.1
    config_path = os.path.join(os.path.dirname(__file__), 'zone_config.json')
    try:
        with open(config_path, 'r') as f:
            zone_config = json.load(f)
    except FileNotFoundError:
        print("❌ LỖI: Không tìm thấy file 'zone_config.json'. Hãy làm Task 1.1 trước!")
        return

    # Khởi tạo Generator cho từng thùng rác
    generators = {}
    bin_levels = {}
    for edge_id, zone_type in zone_config.items():
        generators[edge_id] = RealWasteGenerator(zone_type=zone_type)
        bin_levels[edge_id] = 0.0 # Bắt đầu ngày 1 với 0% rác

    # 2. Thiết lập thông số thời gian
    days = 7
    total_steps = days * 24 * 3600 # Tổng số giây trong 7 ngày (604,800 bước)
    record_interval = 900 # Lưu dữ liệu mỗi 15 phút (900 giây)
    
    csv_filename = "waste_history.csv"
    
    # 3. Chạy vòng lặp thời gian và lưu CSV
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Ghi Header cho file CSV
        writer.writerow(["Timestamp_Step", "Day", "Hour", "Bin_ID", "Zone_Type", "Fill_Level_Percent"])
        
        for step in range(total_steps):
            # Tính toán lượng rác sinh ra tại step này
            for edge_id in zone_config.keys():
                growth = generators[edge_id].get_fill_rate(step)
                bin_levels[edge_id] = min(100.0, bin_levels[edge_id] + growth)
                
                # Nếu thùng đầy 100%, giả sử có xe đi qua dọn dẹp (Reset về 0)
                # (Điều này giúp dữ liệu có hình răng cưa thực tế, thay vì chạm nóc 100% rồi nằm im)
                if bin_levels[edge_id] >= 100.0:
                    bin_levels[edge_id] = 0.0 
            
            # Lưu dữ liệu mỗi 15 phút
            if step % record_interval == 0:
                current_day = (step // (24 * 3600)) + 1
                current_hour = (step // 3600) % 24
                
                for edge_id, zone_type in zone_config.items():
                    # Làm tròn số rác đến 2 chữ số thập phân
                    fill_level = round(bin_levels[edge_id], 2)
                    writer.writerow([step, current_day, current_hour, edge_id, zone_type, fill_level])
            
            # In tiến độ cho đỡ chán
            if step % (24 * 3600) == 0:
                print(f"⏳ Đã giả lập xong Ngày { (step // (24 * 3600)) + 1 }...")

    print(f"✅ ĐÃ HOÀN THÀNH! Dữ liệu được lưu tại: {os.path.abspath(csv_filename)}")
    print("Sẵn sàng mang file này đi train mạng LSTM!")

if __name__ == "__main__":
    generate_7_days_dataset()