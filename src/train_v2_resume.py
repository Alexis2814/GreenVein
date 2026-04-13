import os
import torch
import numpy as np
from environment import GreenVeinEnv
from agent import DQNAgent
from collections import deque
import matplotlib.pyplot as plt

# 🌟 Cấu hình học bổ túc: Từ vòng 1001 đến vòng 1300
def resume_training(start_episode=1001, n_episodes=1300, max_t=2500):
    print("\n" + "🌟"*25, flush=True)
    print(f" BẮT ĐẦU TU LUYỆN IoT (TỪ VÒNG {start_episode} ĐẾN {n_episodes}) ".center(50), flush=True)
    print("🌟"*25 + "\n", flush=True)
    
    env = GreenVeinEnv()

    # KHỞI TẠO 3 BỘ NÃO
    agents = {t: DQNAgent(state_size=5, action_size=3, seed=42) for t in env.truck_ids}

    # 🌟 NẠP LẠI BỘ NÃO CŨ (KẾT QUẢ CỦA 1000 VÒNG TRƯỚC)
    for t in env.truck_ids:
        path = f'models_v2/checkpoint_{t}_ep1000.pth'
        if os.path.exists(path):
            agents[t].qnetwork_local.load_state_dict(torch.load(path))
            print(f"✅ Đã nạp thành công tri thức Expert 1000 vòng cho {t}", flush=True)
        else:
            print(f"⚠️ Không tìm thấy file {path}. Sẽ bắt đầu học mới cho {t}.", flush=True)

    # 🌟 GIẢM ĐỘ NGÁO: Chỉ khám phá 20% thời gian, 80% còn lại dùng trí khôn
    eps = 0.2 
    eps_end = 0.01
    eps_decay = 0.995

    scores_window = {t: deque(maxlen=20) for t in env.truck_ids}
    scores_history = {t: [] for t in env.truck_ids}
    
    os.makedirs('models_v2', exist_ok=True)

    for i_episode in range(start_episode, n_episodes + 1):
        print(f"\n🚀 --- TU LUYỆN NÂNG CAO EPISODE {i_episode}/{n_episodes} ---", flush=True)
        states, _ = env.reset()
        scores = {t: 0.0 for t in env.truck_ids}
        dones = {t: False for t in env.truck_ids}

        for t_step in range(max_t):
            # Cập nhật tiến độ mỗi 50 bước
            if t_step > 0 and t_step % 50 == 0:
                alive_trucks = [t for t in env.truck_ids if not dones[t]]
                if alive_trucks:
                    print(f"   ⏳ [Tiến độ {t_step}/{max_t}] Các xe đang hoạt động: {', '.join(alive_trucks)}...", flush=True)

            actions = {t: agents[t].act(states[t], eps) if not dones[t] else 1 for t in env.truck_ids}
            next_states, rewards, terminated, _, _ = env.step(actions)

            for t in env.truck_ids:
                if not dones[t]:
                    agents[t].step(states[t], actions[t], rewards[t], next_states[t], terminated[t])
                    scores[t] += rewards[t]
                    dones[t] = terminated[t]

            states = next_states
            if all(dones.values()): break

        # Giảm tỷ lệ khám phá
        eps = max(eps_end, eps_decay * eps)
        
        # ====================================================
        # 🌟 KHÔI PHỤC BẢNG TỔNG KẾT EPISODE
        # ====================================================
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

        # ====================================================
        # 🌟 LƯU BỘ NÃO VÀ VẼ BIỂU ĐỒ REAL-TIME MỖI 10 VÒNG
        # ====================================================
        if i_episode % 10 == 0:
            for t in env.truck_ids:
                torch.save(agents[t].qnetwork_local.state_dict(), f'models_v2/checkpoint_{t}_ep{i_episode}.pth')
            print(f"💾 Đã lưu file trọng số vòng {i_episode} vào thư mục 'models_v2/'", flush=True)
            
            plt.figure(figsize=(12, 6))
            plt.style.use('ggplot')
            colors = {'XeRac_AI_1': '#ff7f0e', 'XeRac_AI_2': '#1f77b4', 'XeRac_AI_3': '#2ca02c'}
            
            for t in env.truck_ids:
                smoothed_scores = [np.mean(scores_history[t][max(0, i-10):i+1]) for i in range(len(scores_history[t]))]
                # Điều chỉnh trục X bắt đầu từ start_episode
                x_axis = range(start_episode, start_episode + len(scores_history[t]))
                
                plt.plot(x_axis, smoothed_scores, label=f"{t} (Smoothed)", color=colors[t], linewidth=2, alpha=0.9)
                plt.plot(x_axis, scores_history[t], color=colors[t], linewidth=1, alpha=0.15)
                
            plt.title(f'TIẾN TRÌNH TU LUYỆN IoT (Vòng {start_episode} đến {i_episode})', fontsize=16, fontweight='bold')
            plt.xlabel('Vòng (Episode)', fontsize=12)
            plt.ylabel('Điểm số (Reward)', fontsize=12)
            plt.axhline(0, color='red', linestyle='--', alpha=0.8)
            plt.legend()
            plt.tight_layout()
            plt.savefig('models_v2/training_progress_resume_realtime.png', dpi=300)
            plt.close() 

    env.close()

    # ====================================================
    # 🏆 TỔNG KẾT CHIẾN DỊCH SAU KHI KẾT THÚC (VÒNG 1300)
    # ====================================================
    print("\n" + "🏆"*25, flush=True)
    print(" TỔNG KẾT GIAI ĐOẠN TU LUYỆN IoT ".center(50), flush=True)
    print("🏆"*25 + "\n", flush=True)

    with open('models_v2/final_report_resume.txt', 'w', encoding='utf-8') as f:
        report = f"BÁO CÁO TỔNG KẾT TU LUYỆN IoT (VÒNG {start_episode}-{n_episodes})\n"
        report += "="*70 + "\n"
        
        for t in env.truck_ids:
            scores_array = np.array(scores_history[t])
            best_relative_ep = np.argmax(scores_array) 
            best_ep = start_episode + best_relative_ep
            max_score = np.max(scores_array)
            
            avg_last_50 = np.mean(scores_array[-50:]) if len(scores_array) >= 50 else np.mean(scores_array)
            
            truck_report = f"🚛 [{t}] - Phụ trách: {env.zone_names[t]}\n"
            truck_report += f"   ↳ Kỷ lục cá nhân mới : {max_score:>7.2f} điểm (Đạt được ở Episode {best_ep})\n"
            truck_report += f"   ↳ Điểm TB cuối khóa (50 Ep cuối): {avg_last_50:>7.2f} điểm\n"
            if avg_last_50 > 0:
                truck_report += f"   ↳ Đánh giá: AI ĐÃ TIẾN HÓA THÀNH CHUYÊN GIA IOT!\n"
            else:
                truck_report += f"   ↳ Đánh giá: Vẫn còn khó khăn ở các điểm kẹt xe.\n"
            truck_report += "-"*70 + "\n"
            
            print(truck_report, flush=True)
            report += truck_report
            
        f.write(report)
    print("📄 Đã lưu file Báo cáo chi tiết tại: models_v2/final_report_resume.txt", flush=True)

if __name__ == "__main__":
    resume_training()