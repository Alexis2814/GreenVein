import torch
import numpy as np
from collections import deque
from environment import GreenVeinEnv
from agent import DQNAgent
import os

def train_marl():
    print("🚀 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MADQN (MULTI-AGENT GREENVEIN)")
    print("=" * 60)

    # 1. Khởi tạo Môi trường (Trường đua)
    env = GreenVeinEnv()

    # 2. Khởi tạo 3 "Tài xế AI" độc lập với 3 seed khác nhau để chúng không suy nghĩ giống hệt nhau
    state_size = 4
    action_size = 3
    agents = {
        truck_id: DQNAgent(state_size=state_size, action_size=action_size, seed=i*10)
        for i, truck_id in enumerate(env.truck_ids)
    }

    # 3. CÁC THÔNG SỐ HUẤN LUYỆN CHÍNH (Hyperparameters)
    n_episodes = 300       # Tổng số chuyến đi (Chạy 300 vòng)
    max_steps = 150        # Số bước tối đa mỗi vòng (Tránh xe đi lạc vòng vo mãi không chịu về)
    
    eps_start = 1.0        # Vòng 1: Đi bừa 100% để khám phá bản đồ
    eps_end = 0.05         # Giới hạn sự đi bừa: Tối thiểu chỉ đi bừa 5%
    eps_decay = 0.985      # Sau mỗi vòng, tốc độ đi bừa giảm đi 1.5%

    eps = eps_start
    # Cửa sổ lưu 20 điểm số gần nhất để tính Trung bình (Xem AI có đang tiến bộ không)
    scores_window = {truck_id: deque(maxlen=20) for truck_id in env.truck_ids} 

    # Tạo thư mục lưu "Bằng lái" (Trọng số Model) nếu chưa có
    if not os.path.exists('models'):
        os.makedirs('models')

    # ==========================================
    # BẮT ĐẦU CHẠY VÒNG LẶP HUẤN LUYỆN
    # ==========================================
    for episode in range(1, n_episodes + 1):
        # Đưa 3 xe về Trạm điều hành
        states, _ = env.reset()
        scores = {truck_id: 0.0 for truck_id in env.truck_ids}
        dones = {truck_id: False for truck_id in env.truck_ids}

        # Xe bắt đầu chạy từng bước trong phố
        for step in range(max_steps):
            actions = {}

            # A. Tài xế nhìn đường và vặn vô lăng
            for truck_id in env.truck_ids:
                if not dones[truck_id]: # Chỉ lấy quyết định nếu xe chưa về đích
                    actions[truck_id] = agents[truck_id].act(states[truck_id], eps)
                else:
                    actions[truck_id] = 1 # Xe về đích rồi thì action ảo, bỏ qua

            # B. Xe di chuyển, Môi trường trả về GPS, CO2 và Điểm Reward
            next_states, rewards, terminated, truncated, _ = env.step(actions)

            # C. Tài xế ghi chép lại và Học bài (Quan trọng nhất)
            for truck_id in env.truck_ids:
                if not dones[truck_id]:
                    # Ghi điểm thưởng vào sổ TRƯỚC
                    scores[truck_id] += rewards[truck_id]
                    # Nạp kinh nghiệm và học
                    agents[truck_id].step(states[truck_id], actions[truck_id], rewards[truck_id], next_states[truck_id], terminated[truck_id])
                    # Cập nhật trạng thái về đích SAU KHI đã cộng điểm
                    dones[truck_id] = terminated[truck_id]

            states = next_states

            # Dừng sớm vòng này nếu TẤT CẢ các xe đều đã báo cáo hoàn thành nhiệm vụ
            if all(dones.values()):
                break

        # Sau 1 vòng, giảm tỷ lệ đi bừa xuống một chút
        eps = max(eps_end, eps_decay * eps)

        # Ghi nhận điểm số để làm báo cáo
        for truck_id in env.truck_ids:
            scores_window[truck_id].append(scores[truck_id])

        # 🌟 IN BÁO CÁO SAU MỖI CHUYẾN ĐI 🌟
        print(f"\n🏁 TỔNG KẾT EPISODE {episode}/{n_episodes} | Epsilon (Tỷ lệ đi bừa): {eps:.3f}")
        for truck_id in env.truck_ids:
            avg_score = np.mean(scores_window[truck_id])
            print(f"   [{truck_id}] Điểm chuyến này: {scores[truck_id]:>7.2f} | Điểm TB (20 vòng): {avg_score:>7.2f}")

        # Lưu lại "Trọng số Não" (Model file .pth) cứ sau mỗi 50 vòng
        if episode % 50 == 0:
            for truck_id in env.truck_ids:
                torch.save(agents[truck_id].qnetwork_local.state_dict(), f'models/checkpoint_{truck_id}_ep{episode}.pth')
            print(f"💾 ĐÃ LƯU CHECKPOINT TẠI EPISODE {episode} VÀO THƯ MỤC 'models/'")

    env.close()
    print("\n🎉 HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN!")

if __name__ == "__main__":
    train_marl()