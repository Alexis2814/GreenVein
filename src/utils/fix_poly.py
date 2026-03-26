import os
import re
import html

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
poly_path = os.path.join(base_dir, 'envs', 'map.poly.xml')

if not os.path.exists(poly_path):
    poly_path = os.path.join(os.getcwd(), 'envs', 'map.poly.xml')

def clean_keep_vietnamese(text):
    if not text: return ""
    # 1. Giải mã các ký tự XML bị lỗi
    text = html.unescape(text)
    text = text.replace('&apos;s', '').replace('&apos;', '').replace('&amp;', '_')
    
    # 2. VŨ KHÍ MỚI: Dùng [^\w] để GIỮ LẠI toàn bộ chữ Tiếng Việt có dấu và số. 
    # Bất kỳ thứ gì không phải chữ/số (dấu cách, #, ?, khoảng trắng...) sẽ thành gạch dưới.
    safe_text = re.sub(r'[^\w]', '_', text)
    
    # 3. Dọn dẹp gạch dưới thừa
    return re.sub(r'_+', '_', safe_text).strip('_')

def fix_sumo_ids():
    print(f"🔍 Đang tối ưu hóa hiển thị bản đồ tại: {poly_path}")
    
    if not os.path.exists(poly_path):
        print("❌ Lỗi: Không tìm thấy file map.poly.xml!")
        return

    try:
        with open(poly_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        final_lines = []
        used_names = set()

        for line in lines:
            if '<poly ' not in line and '<poi ' not in line:
                final_lines.append(line)
                continue

            match_id = re.search(r'id="([^"]+)"', line)
            if not match_id:
                final_lines.append(line)
                continue
                
            orig_id = match_id.group(1)
            
            # Khôi phục Tiếng Việt có dấu an toàn
            safe_id = clean_keep_vietnamese(orig_id)

            # 1. TIÊU DIỆT SỐ RÁC & CÔNG TRÌNH VÔ DANH
            if not safe_id or safe_id.isdigit() or 'poly' in safe_id.lower() or 'way' in safe_id.lower():
                continue

            # 2. XÓA GIAN HÀNG TRONG TRUNG TÂM THƯƠNG MẠI
            match_type = re.search(r'type="([^"]+)"', line)
            if match_type:
                orig_type = match_type.group(1).lower()
                # Nếu là shop, cửa hàng nhỏ, cafe, fast_food... -> Xóa ngay không vẽ lên bản đồ
                if 'shop' in orig_type or 'kiosk' in orig_type or 'fast_food' in orig_type or 'cafe' in orig_type:
                    continue
                
                # Sửa luôn Type cho an toàn
                safe_type = clean_keep_vietnamese(orig_type)
                line = line.replace(f'type="{match_type.group(1)}"', f'type="{safe_type}"')

            # 3. CHỐNG CHỒNG CHÉO (GỘP TÊN TÒA NHÀ)
            base_name = re.sub(r'_[0-9]+$', '', safe_id)
            base_name = re.sub(r'_+', '_', base_name).strip('_')

            if not base_name or base_name.isdigit():
                continue

            if base_name in used_names:
                continue
                
            used_names.add(base_name)

            # Cập nhật ID mới, chuẩn Tiếng Việt vào dòng
            line = line.replace(f'id="{orig_id}"', f'id="{base_name}"')
            final_lines.append(line)

        with open(poly_path, 'w', encoding='utf-8') as f:
            f.write("".join(final_lines))
            
        print("✅ XUẤT SẮC! Đã khôi phục Tiếng Việt có dấu, ẩn gian hàng TTTM và tối ưu hiển thị!")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    fix_sumo_ids()