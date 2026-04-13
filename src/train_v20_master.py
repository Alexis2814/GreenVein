import os
import torch
import numpy as np
from environment import GreenVeinEnv
from agent import DQNAgent
from collections import deque
import matplotlib.pyplot as plt

def train_v20_master(n_episodes=500, max_t=2500):
    print("\n" + "🌟"*25)
    print(" BẮT ĐẦU HUẤN LUYỆN V20 (STATE=6) ".center(50))
    print("🌟"*25 + "\n")
    
    env = GreenVeinEnv()

    # 🌟 CHÚ Ý: state_size=6 do có thêm biến Fuel
    agents = {t: DQNAgent(state_size=6, action_size=3, seed=42) for t in env.truck_ids}

    eps = 1.0 
    eps_end = 0.01
    eps_decay = 0.98 # Học nhanh hơn một chút

    scores_window = {t: deque(maxlen=20) for t in env.truck_ids}
    scores_history = {t: [] for t in env.truck_ids}
    
    os.makedirs('models_v20', exist_ok=True)

    for i_episode in range(1, n_episodes + 1):
        print(f"\n🚀 --- VÒNG {i_episode}/{n_episodes} ---")
        states, _ = env.reset()
        scores = {t: 0.0 for t in env.truck_ids}
        dones = {t: False for t in env.truck_ids}

        for t_step in range(max_t):
            actions = {t: agents[t].act(states[t], eps) if not dones[t] else 1 for t in env.truck_ids}
            next_states, rewards, terminated, _, _ = env.step(actions)

            for t in env.truck_ids:
                if not dones[t]:
                    agents[t].step(states[t], actions[t], rewards[t], next_states[t], terminated[t])
                    scores[t] += rewards[t]
                    dones[t] = terminated[t]

            states = next_states
            if all(dones.values()): break

        eps = max(eps_end, eps_decay * eps)
        
        # ====================================================
        # 🌟 BẢNG TỔNG KẾT ĐÃ THÊM SỐ KM VÀ XĂNG
        # ====================================================
        print(f'\n📊 BẢNG TỔNG KẾT EPISODE {i_episode}:')
        for t in env.truck_ids:
            scores_window[t].append(scores[t])
            scores_history[t].append(scores[t])
            
            # Lấy thông tin quãng đường và xăng từ environment
            dist = env.trip_distance.get(t, 0.0)
            fuel = env.current_fuel.get(t, 0.0)
            
            print(f'🚛 [{t}] - Điểm: {scores[t]:>7.2f} | Rác: {env.total_collected.get(t,0):>5.1f}kg | Đi: {dist:>5.2f}km | Xăng: {fuel:>4.1f}%')

        # ====================================================
        # 🌟 LƯU BỘ NÃO VÀ VẼ BIỂU ĐỒ REAL-TIME MỖI 50 VÒNG
        # ====================================================
        if i_episode % 50 == 0:
            for t in env.truck_ids:
                torch.save(agents[t].qnetwork_local.state_dict(), f'models_v20/master_{t}_ep{i_episode}.pth')
            print(f"💾 Đã lưu mốc trọng số vòng {i_episode} vào thư mục 'models_v20/'")
            
            plt.figure(figsize=(12, 6))
            plt.style.use('ggplot')
            colors = {'XeRac_AI_1': '#ff7f0e', 'XeRac_AI_2': '#1f77b4', 'XeRac_AI_3': '#2ca02c'}
            for t in env.truck_ids:
                smoothed = [np.mean(scores_history[t][max(0, i-10):i+1]) for i in range(len(scores_history[t]))]
                plt.plot(smoothed, label=f"{t}", color=colors[t], linewidth=2)
            plt.axhline(0, color='red', linestyle='--')
            plt.title(f'TIẾN TRÌNH TU LUYỆN V20 (Cập nhật vòng {i_episode})', fontsize=14, fontweight='bold')
            plt.xlabel('Vòng (Episode)')
            plt.ylabel('Điểm số (Reward)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('models_v20/training_v20_realtime.png')
            plt.close() 

    env.close()

if __name__ == "__main__":
    train_v20_master()