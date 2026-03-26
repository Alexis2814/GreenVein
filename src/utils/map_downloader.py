# src/utils/map_downloader.py
import osmnx as ox
import os
import sys

# Khai báo đường dẫn gốc để gọi file config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config

def download_dongda_map():
    # Định nghĩa nơi lưu file bản đồ thô
    save_path = os.path.join(Config.BASE_DIR, 'data', 'dongda_map.graphml')
    
    # Cấu hình osmnx: Bật log để xem tiến trình tải, bật cache để tiết kiệm mạng
    ox.settings.log_console = True
    ox.settings.use_cache = True
    
    place_name = "Dong Da District, Hanoi, Vietnam" 
    
    print("📡 Đang kết nối máy chủ OpenStreetMap...")
    print(f"📍 Đang tải mạng lưới giao thông khu vực: {place_name}...")
    print("⏳ Vui lòng kiên nhẫn, quá trình này có thể mất 1-2 phút...")
    
    try:
        # Tải dữ liệu dạng đồ thị (chỉ lấy đường cho phép ô tô/xe rác chạy)
        G = ox.graph_from_place(place_name, network_type='drive')
        
        # Lưu đồ thị dưới dạng định dạng GraphML (chuẩn tốt nhất cho SUMO)
        ox.save_graphml(G, filepath=save_path)
        
        print(f"\n✅ Tải THÀNH CÔNG! Bản đồ đã được lưu tại: {save_path}")
        
    except Exception as e:
        print(f"\n❌ Có lỗi xảy ra trong quá trình tải: {e}")

if __name__ == "__main__":
    download_dongda_map()