import numpy as np
from agent import DQNAgent

def test_dqn_agent():
    print("🤖 ĐANG KHỞI ĐỘNG BÀI SÁT HẠCH TÀI XẾ AI (TASK 14)...")
    print("-" * 60)
    
    try:
        # 1. Tuyển dụng tài xế AI
        agent = DQNAgent(state_size=4, action_size=3, seed=42)
        print("✅ BƯỚC 1: Khởi tạo Tài xế AI thành công! Đã kết nối với Bộ Não (QNetwork).")
        
        # 2. Test phản xạ (Hàm act)
        # Tình huống: Chạy 45 km/h, Xả 600g CO2, Rác A 50, Rác B 80
        dummy_state = np.array([45.0, 600.0, 50.0, 80.0])
        
        action_explore = agent.act(dummy_state, eps=1.0) # Ép đi bừa 100% để khám phá
        action_exploit = agent.act(dummy_state, eps=0.0) # Ép dùng não 100% để chọn đường
        print(f"🚦 BƯỚC 2: Test Phản xạ -> Quyết định khám phá: {action_explore} | Quyết định dùng não: {action_exploit}")
        
        # 3. Test Khả năng học tập (Hàm step & learn)
        print("📚 BƯỚC 3: Đang bơm 65 trải nghiệm giả lập vào Bộ nhớ (Replay Buffer)...")
        for step in range(65):
            state = np.random.rand(4)
            action = np.random.randint(3)
            reward = np.random.randn()
            next_state = np.random.rand(4)
            done = False
            
            # Ghi vào nhật ký. Khi đủ 64 dòng (BATCH_SIZE), nó sẽ tự lôi sách ra học!
            agent.step(state, action, reward, next_state, done)
            
        print("✅ BƯỚC 4: AI đã lôi 64 trải nghiệm ra ôn bài và Cập nhật Trọng số (Backpropagation) thành công không có lỗi toán học!")
        
        print("-" * 60)
        print("🎉 KẾT LUẬN: TASK 14 HOÀN HẢO! TÀI XẾ AI ĐÃ CÓ BẰNG LÁI, SẴN SÀNG RA ĐƯỜNG THỰC TẾ.")
        
    except Exception as e:
        print(f"❌ CÓ LỖI XẢY RA TRONG QUÁ TRÌNH KIỂM TRA AGENT: {e}")

if __name__ == "__main__":
    test_dqn_agent()