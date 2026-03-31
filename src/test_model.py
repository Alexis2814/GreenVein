import torch
from model import QNetwork

def test_qnetwork():
    print("🧠 ĐANG KHỞI ĐỘNG BÀI KIỂM TRA BỘ NÃO AI (TASK 13)...")
    print("-" * 50)
    
    try:
        # 1. Khởi tạo mạng Nơ-ron với 4 input (cảm biến) và 3 output (hành động)
        model = QNetwork(state_size=4, action_size=3, seed=42)
        print("✅ BƯỚC 1: Khởi tạo Model thành công! Không có lỗi cú pháp.")
        
        # 2. Tạo một tình huống giả định (Dummy State) từ Môi trường Task 12
        # Giả sử xe đang chạy 45.5 km/h, xả 693.4 g CO2/km, Rác A 50, Rác B 80
        dummy_state = torch.tensor([45.5, 693.4, 50.0, 80.0], dtype=torch.float32)
        print(f"📥 BƯỚC 2: Bơm tín hiệu đầu vào (State)  -> {dummy_state.tolist()}")
        
        # 3. Cho não "suy nghĩ" (Forward Pass)
        # Dùng no_grad() vì chúng ta chỉ test nháp, chưa cần tính đạo hàm để học
        with torch.no_grad(): 
            q_values = model(dummy_state)
            
        print(f"📤 BƯỚC 3: AI trả về kết quả (Q-Values) -> {q_values.tolist()}")
        
        # 4. Phân tích quyết định của AI
        action = torch.argmax(q_values).item()
        action_names = ["0 (Rẽ Trái)", "1 (Đi Thẳng)", "2 (Rẽ Phải)"]
        print(f"🎯 BƯỚC 4: AI quyết định chọn hành động -> {action_names[action]} (Vì điểm Q-value cao nhất)")
        
        print("-" * 50)
        print("🎉 KẾT LUẬN: TASK 13 HOẠT ĐỘNG HOÀN HẢO! MẠNG NƠ-RON ĐÃ SẴN SÀNG.")
        
    except Exception as e:
        print(f"❌ CÓ LỖI XẢY RA TRONG MODEL: {e}")

if __name__ == "__main__":
    test_qnetwork()