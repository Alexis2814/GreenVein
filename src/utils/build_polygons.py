# src/utils/build_polygons.py
import os
import sys
import subprocess

# Khai báo đường dẫn gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config

def build_landscape():
    net_path = os.path.join(Config.BASE_DIR, 'envs', 'map.net.xml')
    osm_path = os.path.join(Config.BASE_DIR, 'data', 'dongda_map.osm')
    poly_path = os.path.join(Config.BASE_DIR, 'envs', 'map.poly.xml')
    
    if 'SUMO_HOME' not in os.environ:
        print("❌ Lỗi: Chưa nhận diện được SUMO_HOME.")
        return
        
    # File quy chuẩn màu sắc tòa nhà, cây cối mặc định của SUMO
    typemap_path = os.path.join(os.environ['SUMO_HOME'], 'data', 'typemap', 'osmPolyconvert.typ.xml')
    
    print("🌳 Đang dựng cảnh quan (Tòa nhà, Hồ nước, Công viên) cho Quận Đống Đa...")
    
    cmd = [
        "polyconvert",
        "--net-file", net_path,
        "--osm-files", osm_path,
        "--type-file", typemap_path,
        "-o", poly_path,
        "--discard", "true"  # Lọc bỏ các khối rác để bản đồ không bị nặng
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ THÀNH CÔNG! Đã tạo xong file cảnh quan tại: {poly_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi dựng cảnh quan: {e}")

if __name__ == "__main__":
    build_landscape()