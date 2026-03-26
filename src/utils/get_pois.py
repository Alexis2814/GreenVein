# src/utils/get_pois.py
import os
import sys
import requests

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    print("❌ Lỗi: Chưa nhận diện được SUMO_HOME.")
    sys.exit(1)

import sumolib
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config

def extract_all_pois():
    net_path = os.path.join(Config.BASE_DIR, 'envs', 'map.net.xml')
    pois_path = os.path.join(Config.BASE_DIR, 'envs', 'pois.add.xml')
    
    print("📡 Đang quét toàn bộ địa điểm và nhúng tham số Tên hiển thị...")
    
    query = """
    [out:json];
    (
      node["amenity"~"hospital|clinic|pharmacy|school|university|college|kindergarten|fuel|cafe|restaurant|fast_food|bank|police|fire_station|marketplace"](21.000,105.815,21.025,105.840);
      way["amenity"~"hospital|clinic|pharmacy|school|university|college|kindergarten|fuel|cafe|restaurant|fast_food|bank|police|fire_station|marketplace"](21.000,105.815,21.025,105.840);
      node["shop"~"supermarket|mall|convenience|clothes|electronics"](21.000,105.815,21.025,105.840);
      way["shop"~"supermarket|mall|convenience|clothes|electronics"](21.000,105.815,21.025,105.840);
      node["leisure"~"park|sports_centre|stadium"](21.000,105.815,21.025,105.840);
      way["leisure"~"park|sports_centre|stadium"](21.000,105.815,21.025,105.840);
    );
    out center;
    """
    
    try:
        response = requests.get("https://overpass-api.de/api/interpreter", data=query, timeout=90)
        data = response.json()
        net = sumolib.net.readNet(net_path)
        
        colors = {
            "hospital": "255,50,50", "clinic": "255,100,100", "pharmacy": "255,150,150",
            "school": "50,150,255", "university": "0,100,255", "college": "50,150,255", "kindergarten": "100,200,255",
            "fuel": "255,150,0", "police": "0,0,200", "fire_station": "200,0,0",
            "cafe": "139,69,19", "restaurant": "255,200,50", "fast_food": "255,170,0",
            "marketplace": "150,50,200", "supermarket": "150,50,200", "mall": "150,50,200", "convenience": "180,100,220",
            "bank": "0,255,255",
            "park": "50,200,50", "stadium": "0,150,0", "sports_centre": "0,180,0"
        }
        
        poi_xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<additional>\n'
        
        count = 0
        for el in data.get('elements', []):
            tags = el.get('tags', {})
            name = tags.get('name', 'Unknown')
            if name == 'Unknown': continue
            
            # Xử lý các ký tự đặc biệt để không bị lỗi file XML
            name = name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")
            
            poi_type = tags.get('amenity') or tags.get('shop') or tags.get('leisure')
            if not poi_type: continue
            
            if el['type'] == 'node':
                lat, lon = el['lat'], el['lon']
            else:
                lat, lon = el['center']['lat'], el['center']['lon']
                
            x, y = net.convertLonLat2XY(lon, lat)
            color = colors.get(poi_type, "200,200,200")
            
            # ĐÂY LÀ ĐIỂM KHÁC BIỆT: Ép nhúng thẻ <param key="name"> vào bên trong
            poi_xml_content += f'    <poi id="{poi_type}_{el["id"]}" type="{poi_type}" color="{color}" layer="10" x="{x}" y="{y}" width="15" height="15">\n'
            poi_xml_content += f'        <param key="name" value="{name}"/>\n'
            poi_xml_content += f'    </poi>\n'
            count += 1
            
        poi_xml_content += '</additional>\n'
        
        with open(pois_path, 'w', encoding='utf-8') as f:
            f.write(poi_xml_content)
            
        print(f"✅ THÀNH CÔNG! Đã ép thông số Tên cho {count} địa điểm.")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    extract_all_pois()