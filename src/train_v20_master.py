import os
import sys

# 🌟 TẮT ĐỒ HỌA 3D ĐỂ ÉP XUNG HUẤN LUYỆN (Dùng Libsumo cho tốc độ x100)
os.environ["USE_GUI"] = "0"

# Đảm bảo Python nhận diện đúng thư mục src để import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from core.environment import GreenVeinEnv
from rl_agents.agent import DQNAgent
from collections import deque
import matplotlib.pyplot as plt

# Tối ưu hóa backend PyTorch
torch.set_num_threads(1) 
try:
    torch.set_float32_matmul_precision('high')
except:
    pass

# 🌟 TĂNG max_t lên 2500 ĐỂ XE CÓ ĐỦ THỜI GIAN GOM 54 ĐIỂM RÁC THẬT
# 🌟 CHỈNH eps_decay = 0.99 ĐỂ AI KHÁM PHÁ ĐỦ CHU KỲ 7 NGÀY CỦA LSTM
def train_v20_decentralized(n_episodes=500, max_t=2500, eps_start=1.0, eps_end=0.01, eps_decay=0.99):
    print("\n" + "🌟"*25, flush=True)
    print(" BẮT ĐẦU HUẤN LUYỆN V20 PRO MAX (HỢP THỂ LSTM) ".center(50), flush=True)
    print("🌟"*25 + "\n", flush=True)
    
    env = GreenVeinEnv()

    env.sumo_cmd = [
        "sumo", "-c", env.sumo_cfg, 
        "--no-warnings", 
        "--no-step-log", 
        "--time-to-teleport", "-1",
        "--error-log", "sumo_error_train.log"
    ]

    agents = {
        truck_id: DQNAgent(state_size=6, action_size=3, seed=42)
        for truck_id in env.truck_ids
    }

    eps = eps_start 

    scores_window = {t: deque(maxlen=20) for t in env.truck_ids}
    scores_history = {t: [] for t in env.truck_ids}
    
    os.makedirs('models_v20_pro', exist_ok=True)

    for i_episode in range(1, n_episodes + 1):
        # Tính toán ngày trong tuần để in ra log cho dễ theo dõi
        day_names = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]
        current_day = day_names[(i_episode - 1) % 7]
        
        print(f"\n🚀 --- VÒNG {i_episode}/{n_episodes} ({current_day}) | Epsilon: {eps:.3f} ---", flush=True)
        
        states, _ = env.reset(current_episode=i_episode)
        
        scores = {t: 0.0 for t in env.truck_ids}
        dones = {t: False for t in env.truck_ids}
        finished_early = False

        for t_step in range(max_t):
            if t_step > 0 and t_step % 500 == 0:
                alive = [t for t in env.truck_ids if not dones[t]]
                if alive: print(f"    ⏳ [Bước {t_step}/{max_t}] Xe đang dọn: {', '.join(alive)}...", flush=True)

            with torch.no_grad():
                actions = {}
                for t in env.truck_ids:
                    if not dones[t]:
                        actions[t] = agents[t].act(states[t], eps)
                    else:
                        actions[t] = 1 # Kéo phanh tay đỗ lại nếu đã xong việc
            
            next_states, rewards, terminated, _, _ = env.step(actions)

            for t in env.truck_ids:
                if not dones[t]:
                    agents[t].step(states[t], actions[t], rewards[t], next_states[t], terminated[t])
                    scores[t] += rewards[t]
                    dones[t] = terminated[t]

            states = next_states
            if all(dones.values()): 
                finished_early = True
                break

        eps = max(eps_end, eps_decay * eps)
        
        print(f'\n📊 BẢNG TỔNG KẾT VÒNG {i_episode} ({current_day}):', flush=True)
        if finished_early:
            print("🏁 Trạng thái: HoÀN THÀNH SỚM (Đã dọn sạch 54 điểm).")
        else:
            print("⏳ Trạng thái: HẾT CA LÀM VIỆC (Bị cắt ngang do hết giờ).")

        for t in env.truck_ids:
            scores_window[t].append(scores[t])
            scores_history[t].append(scores[t])
            
            dist = env.trip_distance.get(t, 0.0)
            fuel = env.current_fuel.get(t, 0.0)
            co2_kg = env.trip_co2.get(t, 0.0) / 1000.0 
            zone_name = env.zone_names.get(t, "Không rõ")
            
            print(f'🚛 [{t}] ({zone_name}) - Điểm: {scores[t]:>7.2f} | Rác: {env.total_collected.get(t,0):>5.1f}kg | Đi: {dist:>5.2f}km | Xăng: {fuel:>4.1f}% | CO2: {co2_kg:>5.2f}kg')

        # 🌟 LƯU TRỌNG SỐ SAU MỖI 10 VÒNG
        if i_episode % 10 == 0:
            for t in env.truck_ids:
                torch.save(agents[t].qnetwork_local.state_dict(), f'models_v20_pro/brain_{t}_ep{i_episode}.pth')
            print(f"💾 Đã lưu 3 bộ Não Độc lập vòng {i_episode} vào thư mục 'models_v20_pro/'", flush=True)
            
            # Vẽ biểu đồ tiến trình
            plt.figure(figsize=(12, 6))
            plt.style.use('ggplot')
            colors = {'XeRac_AI_1': '#ff7f0e', 'XeRac_AI_2': '#1f77b4', 'XeRac_AI_3': '#2ca02c'}
            for t in env.truck_ids:
                smoothed = [np.mean(scores_history[t][max(0, i-10):i+1]) for i in range(len(scores_history[t]))]
                plt.plot(smoothed, label=f"{t} ({env.zone_names[t]})", color=colors[t], linewidth=2)
            plt.axhline(0, color='red', linestyle='--')
            plt.title(f'TIẾN TRÌNH TU LUYỆN RL (MÔI TRƯỜNG ĐỘNG LSTM) - Vòng {i_episode}', fontsize=14, fontweight='bold')
            plt.xlabel('Vòng (Episode)')
            plt.ylabel('Điểm số (Reward)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('models_v20_pro/training_progress.png')
            plt.close() 

    env.close()
    
    print("\n" + "🏆"*25, flush=True)
    print(" TỔNG KẾT CHIẾN DỊCH SAU KHI KẾT THÚC ".center(50), flush=True)
    print("🏆"*25 + "\n", flush=True)

    with open('models_v20_pro/final_report.txt', 'w', encoding='utf-8') as f:
        report = f"BÁO CÁO TỔNG KẾT CHIẾN DỊCH GREENVEIN HYBRID AI ({n_episodes} EPISODES)\n"
        report += "="*70 + "\n"
        for t in env.truck_ids:
            scores_array = np.array(scores_history[t])
            best_ep = np.argmax(scores_array) + 1
            max_score = np.max(scores_array)
            avg_last_50 = np.mean(scores_array[-50:]) if len(scores_array) >= 50 else np.mean(scores_array)
            
            truck_report = f"🚛 [{t}] - Phụ trách: {env.zone_names[t]}\n"
            truck_report += f"   ↳ Kỷ lục cá nhân : {max_score:>7.2f} điểm (Tại Episode {best_ep})\n"
            truck_report += f"   ↳ Điểm TB (50 Ep cuối): {avg_last_50:>7.2f} điểm\n"
            truck_report += "-"*70 + "\n"
            print(truck_report, flush=True)
            report += truck_report
        f.write(report)

if __name__ == "__main__":
    # Bắt đầu luyện AI với 500 vòng (Hoặc có thể chỉnh xuống 300 nếu muốn nhanh)
    train_v20_decentralized(n_episodes=500)