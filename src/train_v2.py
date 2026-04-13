import os
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from environment import GreenVeinEnv
from agent import DQNAgent

# 🌟 GIAI ĐOẠN TU LUYỆN: Tăng lên 500 vòng, giảm Epsilon chậm lại (0.99)
def train_madqn(n_episodes=500, max_t=2500, eps_start=1.0, eps_end=0.01, eps_decay=0.99):
    print("\n" + "🌟"*25, flush=True)
    print(" BẮT ĐẦU CHIẾN DỊCH TU LUYỆN MADQN (500 VÒNG) ".center(50), flush=True)
    print("🌟"*25 + "\n", flush=True)
    
    env = GreenVeinEnv()

    # KHỞI TẠO 3 BỘ NÃO
    agents = {
        truck_id: DQNAgent(state_size=5, action_size=3, seed=42)
        for truck_id in env.truck_ids
    }

    scores_window = {t: deque(maxlen=20) for t in env.truck_ids}
    scores_history = {t: [] for t in env.truck_ids}
    eps = eps_start

    # Tạo thư mục lưu bằng lái mới
    os.makedirs('models_v2', exist_ok=True)

    for i_episode in range(1, n_episodes + 1):
        print(f"\n🚀 --- BẮT ĐẦU EPISODE {i_episode}/{n_episodes} ---", flush=True)
        states, _ = env.reset()
        scores = {t: 0.0 for t in env.truck_ids}
        dones = {t: False for t in env.truck_ids}

        for t_step in range(max_t):
            
            # ====================================================
            # 🌟 BÁO CÁO NHỊP TIM MỖI 50 BƯỚC
            # ====================================================
            if t_step > 0 and t_step % 50 == 0:
                alive_trucks = [t for t in env.truck_ids if not dones[t]]
                if alive_trucks:
                    print(f"   ⏳ [Tiến độ {t_step}/{max_t}] Các xe đang hoạt động: {', '.join(alive_trucks)}...", flush=True)

            actions = {}
            for t in env.truck_ids:
                if not dones[t]:
                    actions[t] = agents[t].act(states[t], eps)
                else:
                    actions[t] = 1 

            next_states, rewards, terminated, _, _ = env.step(actions)

            for t in env.truck_ids:
                if not dones[t]:
                    agents[t].step(states[t], actions[t], rewards[t], next_states[t], terminated[t])
                    scores[t] += rewards[t]
                    dones[t] = terminated[t]

            states = next_states
            if all(dones.values()):
                break

        # Giảm tỷ lệ khám phá (Epsilon) chậm lại để học sâu hơn
        eps = max(eps_end, eps_decay * eps)
        
        print(f'\n📊 BẢNG TỔNG KẾT EPISODE {i_episode}:', flush=True)
        print("=" * 90, flush=True)
        for t in env.truck_ids:
            scores_window[t].append(scores[t])
            scores_history[t].append(scores[t])
            
            dist = env.trip_distance.get(t, 0.0)
            co2_total = env.trip_co2.get(t, 0.0)
            co2_km = (co2_total / dist) if dist > 0.001 else 0.0
            waste = env.total_collected.get(t, 0.0)
            zone_name = env.zone_names.get(t, "Không xác định")
            
            print(f'🚛 [{t}] - Phụ trách: {zone_name}', flush=True)
            print(f'   ↳ Điểm AI: {scores[t]:>7.2f} | Lượng rác thu: {waste:>6.1f} kg | Đi: {dist:>5.2f} km | CO2: {co2_km:>6.1f} g/km', flush=True)
            print("-" * 90, flush=True)

        # Lưu checkpoint và vẽ biểu đồ mượt mỗi 10 vòng
        if i_episode % 10 == 0:
            for t in env.truck_ids:
                torch.save(agents[t].qnetwork_local.state_dict(), f'models_v2/checkpoint_{t}_ep{i_episode}.pth')
            print(f"💾 Đã lưu file trọng số (.pth) vào thư mục 'models_v2/'", flush=True)
            
            # 🌟 TỐI ƯU CẬP NHẬT: Vẽ biểu đồ Real-time mỗi 10 vòng
            plt.figure(figsize=(12, 6))
            plt.style.use('ggplot')
            colors = {'XeRac_AI_1': '#ff7f0e', 'XeRac_AI_2': '#1f77b4', 'XeRac_AI_3': '#2ca02c'}
            
            for t in env.truck_ids:
                smoothed_scores = [np.mean(scores_history[t][max(0, i-10):i+1]) for i in range(len(scores_history[t]))]
                plt.plot(smoothed_scores, label=f"{t} (Smoothed)", color=colors[t], linewidth=2, alpha=0.9)
                plt.plot(scores_history[t], color=colors[t], linewidth=1, alpha=0.15)
                
            plt.title(f'TIẾN TRÌNH HUẤN LUYỆN MADQN (Đang chạy vòng {i_episode}/{n_episodes})', fontsize=16, fontweight='bold')
            plt.xlabel('Vòng (Episode)', fontsize=12)
            plt.ylabel('Điểm số (Reward)', fontsize=12)
            plt.axhline(0, color='red', linestyle='--', alpha=0.8)
            plt.legend()
            plt.tight_layout()
            plt.savefig('models_v2/training_progress_realtime.png', dpi=300)
            plt.close() # Đóng plot để tránh tràn RAM bộ nhớ máy tính

    env.close()

    # ====================================================
    # 🏆 TỔNG KẾT CHIẾN DỊCH SAU KHI KẾT THÚC 500 VÒNG
    # ====================================================
    print("\n" + "🏆"*25, flush=True)
    print(" TỔNG KẾT CHIẾN DỊCH SAU KHI KẾT THÚC ".center(50), flush=True)
    print("🏆"*25 + "\n", flush=True)

    # 1. Tạo file Text Báo cáo
    with open('models_v2/final_report.txt', 'w', encoding='utf-8') as f:
        report = f"BÁO CÁO TỔNG KẾT CHIẾN DỊCH GREENVEIN ({n_episodes} EPISODES)\n"
        report += "="*70 + "\n"
        
        for t in env.truck_ids:
            scores_array = np.array(scores_history[t])
            best_ep = np.argmax(scores_array) + 1
            max_score = np.max(scores_array)
            
            # Lấy trung bình 50 vòng cuối thay vì 20 để đánh giá chuẩn xác hơn
            avg_last_50 = np.mean(scores_array[-50:]) if len(scores_array) >= 50 else np.mean(scores_array)
            
            truck_report = f"🚛 [{t}] - Phụ trách: {env.zone_names[t]}\n"
            truck_report += f"   ↳ Kỷ lục cá nhân cao nhất : {max_score:>7.2f} điểm (Đạt được ở Episode {best_ep})\n"
            truck_report += f"   ↳ Điểm TB cuối khóa (50 Ep cuối): {avg_last_50:>7.2f} điểm\n"
            if avg_last_50 > 0:
                truck_report += f"   ↳ Đánh giá: AI ĐÃ TỐT NGHIỆP XUẤT SẮC! Biết gom rác và né kẹt xe.\n"
            else:
                truck_report += f"   ↳ Đánh giá: AI cần cày cuốc thêm. Vẫn còn bị kẹt đường oan uổng.\n"
            truck_report += "-"*70 + "\n"
            
            print(truck_report, flush=True)
            report += truck_report
            
        f.write(report)
    print("📄 Đã lưu file Báo cáo chi tiết tại: models_v2/final_report.txt", flush=True)

    # 2. Biểu đồ bản Final
    print("🎨 Đang vẽ biểu đồ tiến trình học tập cuối cùng...", flush=True)
    plt.figure(figsize=(12, 6))
    plt.style.use('ggplot')
    colors = {'XeRac_AI_1': '#ff7f0e', 'XeRac_AI_2': '#1f77b4', 'XeRac_AI_3': '#2ca02c'}
    
    for t in env.truck_ids:
        # Đường Smoothed bản cuối lấy trung bình 20 vòng cho thật mượt
        smoothed_scores = [np.mean(scores_history[t][max(0, i-20):i+1]) for i in range(len(scores_history[t]))]
        plt.plot(smoothed_scores, label=f"{t} (Smoothed)", color=colors[t], linewidth=2, alpha=0.9)
        plt.plot(scores_history[t], color=colors[t], linewidth=1, alpha=0.15)
        
    plt.title(f'TIẾN TRÌNH HUẤN LUYỆN MADQN {n_episodes} VÒNG (HỆ THỐNG THU GOM 5 TẤN)', fontsize=16, fontweight='bold')
    plt.xlabel('Vòng (Episode)', fontsize=12)
    plt.ylabel('Điểm số (Reward)', fontsize=12)
    plt.axhline(0, color='red', linestyle='--', alpha=0.8) 
    plt.legend()
    plt.tight_layout()
    plt.savefig('models_v2/training_progress_final.png', dpi=300)
    print("✅ Đã lưu biểu đồ thành công tại: models_v2/training_progress_final.png", flush=True)

if __name__ == "__main__":
    train_madqn()