import os
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from environment import GreenVeinEnv
from agent import DQNAgent

# 🌟 GIAI ĐOẠN TU LUYỆN: Tăng lên 500 vòng, nới rộng thời gian (max_t=5000)
# eps_decay=0.995 giúp AI thoát khỏi giai đoạn chạy random mù lòa nhanh hơn
def train_madqn(n_episodes=500, max_t=5000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    print("\n" + "🌟"*25, flush=True)
    print(" BẮT ĐẦU CHIẾN DỊCH TU LUYỆN MADQN (500 VÒNG) ".center(50), flush=True)
    print("🌟"*25 + "\n", flush=True)
    
    env = GreenVeinEnv()

    # ====================================================
    # 🌟 TỐI ƯU: KHỞI TẠO DUY NHẤT 1 BỘ NÃO CHO CẢ 3 XE
    # Chuẩn state_size=6 (đã thêm Fuel)
    # ====================================================
    global_agent = DQNAgent(state_size=6, action_size=3, seed=42)

    scores_window = {t: deque(maxlen=20) for t in env.truck_ids}
    scores_history = {t: [] for t in env.truck_ids}
    eps = eps_start

    # Tạo thư mục lưu bằng lái mới
    os.makedirs('models_v2', exist_ok=True)

    for i_episode in range(1, n_episodes + 1):
        print(f"\n🚀 --- BẮT ĐẦU EPISODE {i_episode}/{n_episodes} | Epsilon: {eps:.3f} ---", flush=True)
        states, _ = env.reset()
        scores = {t: 0.0 for t in env.truck_ids}
        dones = {t: False for t in env.truck_ids}

        for t_step in range(max_t):
            
            # ====================================================
            # 🌟 BÁO CÁO NHỊP TIM MỖI 500 BƯỚC (Giảm tần suất để đỡ trôi Log)
            # ====================================================
            if t_step > 0 and t_step % 500 == 0:
                alive_trucks = [t for t in env.truck_ids if not dones[t]]
                if alive_trucks:
                    print(f"   ⏳ [Tiến độ {t_step}/{max_t}] Các xe đang hoạt động: {', '.join(alive_trucks)}...", flush=True)

            # Lấy hành động từ NÃO CHUNG
            actions = {}
            for t in env.truck_ids:
                if not dones[t]:
                    actions[t] = global_agent.act(states[t], eps)
                else:
                    actions[t] = 1 # Xe chết/xong việc thì phanh tay

            next_states, rewards, terminated, _, _ = env.step(actions)

            # Nhồi kinh nghiệm vào NÃO CHUNG
            for t in env.truck_ids:
                if not dones[t]:
                    global_agent.step(states[t], actions[t], rewards[t], next_states[t], terminated[t])
                    scores[t] += rewards[t]
                    dones[t] = terminated[t]

            states = next_states
            if all(dones.values()):
                break

        # Giảm tỷ lệ khám phá (Epsilon)
        eps = max(eps_end, eps_decay * eps)
        
        print(f'\n📊 BẢNG TỔNG KẾT EPISODE {i_episode}:', flush=True)
        print("=" * 90, flush=True)
        for t in env.truck_ids:
            scores_window[t].append(scores[t])
            scores_history[t].append(scores[t])
            
            dist = env.trip_distance.get(t, 0.0)
            co2_total_kg = env.trip_co2.get(t, 0.0) / 1000.0 # Đổi ra Kg
            co2_kg_per_km = (co2_total_kg / dist) if dist > 0.001 else 0.0
            fuel = env.current_fuel.get(t, 0.0)
            waste = env.total_collected.get(t, 0.0)
            zone_name = env.zone_names.get(t, "Không xác định")
            
            print(f'🚛 [{t}] - Phụ trách: {zone_name}', flush=True)
            print(f'   ↳ Điểm AI: {scores[t]:>7.2f} | Rác: {waste:>6.1f} kg | Đi: {dist:>5.2f} km | Xăng: {fuel:>4.1f}% | CO2: {co2_kg_per_km:>6.2f} kg/km', flush=True)
            print("-" * 90, flush=True)

        # Lưu checkpoint và vẽ biểu đồ mượt mỗi 10 vòng
        if i_episode % 10 == 0:
            # Nhân bản Não chung ra thành 3 file để tương thích lúc Test
            for t in env.truck_ids:
                torch.save(global_agent.qnetwork_local.state_dict(), f'models_v2/checkpoint_{t}_ep{i_episode}.pth')
            print(f"💾 Đã lưu file trọng số (.pth) vào thư mục 'models_v2/'", flush=True)
            
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
            plt.close() 

    env.close()

    # ====================================================
    # 🏆 TỔNG KẾT CHIẾN DỊCH SAU KHI KẾT THÚC 500 VÒNG
    # ====================================================
    print("\n" + "🏆"*25, flush=True)
    print(" TỔNG KẾT CHIẾN DỊCH SAU KHI KẾT THÚC ".center(50), flush=True)
    print("🏆"*25 + "\n", flush=True)

    with open('models_v2/final_report.txt', 'w', encoding='utf-8') as f:
        report = f"BÁO CÁO TỔNG KẾT CHIẾN DỊCH GREENVEIN ({n_episodes} EPISODES)\n"
        report += "="*70 + "\n"
        
        for t in env.truck_ids:
            scores_array = np.array(scores_history[t])
            best_ep = np.argmax(scores_array) + 1
            max_score = np.max(scores_array)
            
            avg_last_50 = np.mean(scores_array[-50:]) if len(scores_array) >= 50 else np.mean(scores_array)
            
            truck_report = f"🚛 [{t}] - Phụ trách: {env.zone_names[t]}\n"
            truck_report += f"   ↳ Kỷ lục cá nhân : {max_score:>7.2f} điểm (Tại Episode {best_ep})\n"
            truck_report += f"   ↳ Điểm TB (50 Ep cuối): {avg_last_50:>7.2f} điểm\n"
            if avg_last_50 > 0:
                truck_report += f"   ↳ Đánh giá: AI ĐÃ TỐT NGHIỆP XUẤT SẮC! Biết gom rác và né kẹt xe.\n"
            else:
                truck_report += f"   ↳ Đánh giá: AI cần cày cuốc thêm. Vẫn còn bị kẹt đường oan uổng.\n"
            truck_report += "-"*70 + "\n"
            
            print(truck_report, flush=True)
            report += truck_report
            
        f.write(report)
    print("📄 Đã lưu file Báo cáo chi tiết tại: models_v2/final_report.txt", flush=True)

    print("🎨 Đang vẽ biểu đồ tiến trình học tập cuối cùng...", flush=True)
    plt.figure(figsize=(12, 6))
    plt.style.use('ggplot')
    colors = {'XeRac_AI_1': '#ff7f0e', 'XeRac_AI_2': '#1f77b4', 'XeRac_AI_3': '#2ca02c'}
    
    for t in env.truck_ids:
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