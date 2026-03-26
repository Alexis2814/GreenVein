# src/utils/build_network.py
import os
import sys
import requests
import subprocess

# Khai báo đường dẫn gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config

def build_sumo_network():
    osm_path = os.path.join(Config.BASE_DIR, 'data', 'dongda_map.osm')
    net_path = os.path.join(Config.BASE_DIR, 'envs', 'map.net.xml')
    
    print("📡 1. Đang tải dữ liệu tọa độ chi tiết (.osm) cho SUMO...")
    # Tọa độ Bounding Box khu vực Đống Đa (min_lon, min_lat, max_lon, max_lat)
    bbox = "105.815,21.000,105.840,21.025"
    url = f"https://overpass-api.de/api/map?bbox={bbox}"
    
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        with open(osm_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Đã lưu file vệ tinh thô tại: {osm_path}")
    except Exception as e:
        print(f"❌ Lỗi khi tải file: {e}")
        return

    print("\n⚙️ 2. Đang biên dịch bản đồ sang định dạng 3D của SUMO...")
    # Lệnh netconvert với các tham số TỐI ƯU HÓA THỰC TẾ
    cmd = [
        "netconvert",
        "--osm-files", osm_path,
        "-o", net_path,
        "--geometry.remove", "true",       # Làm mượt đường
        "--roundabouts.guess", "true",     # Tự động nhận diện vòng xuyến
        "--junctions.join", "true",        # Gộp các ngã tư phức tạp (tránh kẹt xe ảo)
        "--tls.guess-signals", "true",     # Lấy dữ liệu đèn giao thông từ vệ tinh
        "--tls.discard-simple", "true",    # Bỏ đèn tín hiệu ở ngõ hẻm nhỏ
        "--tls.join", "true",              # Gộp các cụm đèn ở ngã tư lớn cho đồng bộ
        "--sidewalks.guess", "true",       # BẬT: Tự động dự đoán và vẽ vỉa hè
        "--crossings.guess", "true" ,       # BẬT: Tự động vẽ vạch kẻ đường cho người đi bộ
        "--output.street-names", "true"
    ]
    
    try:
        # Gọi thẳng tiến trình hệ thống để chạy lệnh
        subprocess.run(cmd, check=True)
        print(f"\n🎉 THÀNH CÔNG! Đã tạo xong mạng lưới giao thông SUMO tại: {net_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Lỗi khi biên dịch: {e}")

if __name__ == "__main__":
    build_sumo_network()