# src/utils/generate_bins.py
import os
import sys
import random
import sumolib
from xml.dom import minidom
import xml.etree.ElementTree as ET

# Khai báo đường dẫn gốc để gọi config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config

def generate_pois():
    # Đường dẫn file đầu vào (bản đồ) và đầu ra (file chứa rác)
    net_path = os.path.join(Config.BASE_DIR, 'envs', 'map.net.xml')
    out_path = os.path.join(Config.BASE_DIR, 'envs', 'bins.add.xml')

    print("🗺️ 1. Đang đọc cấu trúc đồ thị giao thông SUMO...")
    try:
        # Đọc bản đồ bằng sumolib
        net = sumolib.net.readNet(net_path)
    except Exception as e:
        print(f"❌ Lỗi không tìm thấy bản đồ: {e}")
        return
    
    # Lấy tất cả các con đường (edges) cho phép ô tô/xe tải chạy
    edges = [e for e in net.getEdges() if e.allows("passenger") or e.allows("truck")]
    
    if len(edges) < Config.NUM_BINS + 1:
        print("❌ Lỗi: Bản đồ quá nhỏ, không đủ đường để đặt thùng rác!")
        return

    print(f"🎲 2. Đang rải ngẫu nhiên {Config.NUM_BINS} thùng rác và 1 bãi tập kết...")
    
    # Cố định random seed (ví dụ: 42) để lần nào chạy mã cũng ra đúng các vị trí này (dễ test AI sau này)
    random.seed(42) 
    selected_edges = random.sample(edges, Config.NUM_BINS + 1)
    
    # Tách 1 đường làm Depot, 50 đường còn lại làm Bins
    depot_edge = selected_edges[0]
    bin_edges = selected_edges[1:]

    # Khởi tạo thẻ XML gốc <additional> (chuẩn của SUMO)
    root = ET.Element("additional")
    
    # 1. Tạo Bãi tập kết (Depot) - Màu Đỏ, kích thước to (5x5)
    ET.SubElement(root, "poi", id=Config.DEPOT_NODE, type="depot", color="255,0,0", 
                  lane=depot_edge.getLanes()[0].getID(), pos="10.0", width="5.0", height="5.0")
    
    # 2. Tạo các Thùng rác (Bins) - Màu Xanh lá, kích thước nhỏ (2x2)
    for i, edge in enumerate(bin_edges):
        bin_id = f"bin_{i:03d}"
        lane_id = edge.getLanes()[0].getID()
        # Đặt thùng rác ở một vị trí bất kỳ dọc theo lề đường
        pos = str(round(random.uniform(5.0, edge.getLength() - 5.0), 2))
        ET.SubElement(root, "poi", id=bin_id, type="trash_bin", color="0,255,0", 
                      lane=lane_id, pos=pos, width="2.0", height="2.0")

    # Format file XML cho đẹp và lưu lại
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xmlstr)
        
    print(f"✅ THÀNH CÔNG! Đã rải xong rác. File được lưu tại: {out_path}")

if __name__ == "__main__":
    generate_pois()