import os
import torch
import numpy as np
from environment import GreenVeinEnv
from agent import DQNAgent
from collections import deque
import matplotlib.pyplot as plt

# Tối ưu hóa backend PyTorch
torch.set_num_threads(1) 
try:
    torch.set_float32_matmul_precision('high')
except:
    pass

def train_v20_decentralized(n_episodes=500, max_t=800, eps_start=1.0, eps_end=0.01, eps_decay=0.98):
    print("\n" + "🌟"*25, flush=True)
    print(" BẮT ĐẦU HUẤN LUYỆN V20 PRO (3 BỘ NÃO ĐỘC LẬP) ".center(50), flush=True)
    print("🌟"*25 + "\n", flush=True)
    
    env = GreenVeinEnv()

    env.sumo_cmd = [
        "sumo", "-c", env.sumo_cfg, 
        "--no-warnings", 
        "--no-step-log", 
        "--time-to-teleport", "-1",
        "--error-log", "sumo_error_train.log"
    ]

    # =========================================================================
    # 🌟 NÂNG CẤP KIẾN TRÚC: Khởi tạo 3 Bộ Não riêng biệt cho 3 cụm địa hình
    # =========================================================================
    agents = {
        truck_id: DQNAgent(state_size=6, action_size=3, seed=42)
        for truck_id in env.truck_ids
    }

    eps = eps_start 

    scores_window = {t: deque(maxlen=20) for t in env.truck_ids}
    scores_history = {t: [] for t in env.truck_ids}
    
    # 🌟 Lưu vào thư mục MỚI, không ghi đè thư mục models_v20 cũ
    os.makedirs('models_v20_pro', exist_ok=True)

    for i_episode in range(1, n_episodes + 1):
        print(f"\n🚀 --- VÒNG {i_episode}/{n_episodes} | Epsilon (Random): {eps:.3f} ---", flush=True)
        states, _ = env.reset()
        scores = {t: 0.0 for t in env.truck_ids}
        dones = {t: False for t in env.truck_ids}

        for t_step in range(max_t):
            if t_step > 0 and t_step % 200 == 0:
                alive = [t for t in env.truck_ids if not dones[t]]
                if alive: print(f"   ⏳ [Bước {t_step}/{max_t}] Xe còn hoạt động: {', '.join(alive)}...", flush=True)

            # 🌟 Đã sửa: Khóa an toàn không cho xe "chết" hành động ngẫu nhiên
            with torch.no_grad():
                actions = {}
                for t in env.truck_ids:
                    if not dones[t]:
                        actions[t] = agents[t].act(states[t], eps)
                    else:
                        actions[t] = 1 # Xe xong việc thì kéo phanh tay đỗ lại
            
            next_states, rewards, terminated, _, _ = env.step(actions)

            # Huấn luyện độc lập: Xe nào học kinh nghiệm của xe nấy
            for t in env.truck_ids:
                if not dones[t]:
                    agents[t].step(states[t], actions[t], rewards[t], next_states[t], terminated[t])
                    scores[t] += rewards[t]
                    dones[t] = terminated[t]

            states = next_states
            if all(dones.values()): break

        eps = max(eps_end, eps_decay * eps)
        
        print(f'\n📊 BẢNG TỔNG KẾT EPISODE {i_episode}:', flush=True)
        for t in env.truck_ids:
            scores_window[t].append(scores[t])
            scores_history[t].append(scores[t])
            
            dist = env.trip_distance.get(t, 0.0)
            fuel = env.current_fuel.get(t, 0.0)
            co2_kg = env.trip_co2.get(t, 0.0) / 1000.0 
            
            # Thẩm mỹ: In ra tên vùng cho dễ quan sát
            zone_name = env.zone_names.get(t, "Không rõ")
            print(f'🚛 [{t}] ({zone_name}) - Điểm: {scores[t]:>7.2f} | Rác: {env.total_collected.get(t,0):>5.1f}kg | Đi: {dist:>5.2f}km | Xăng: {fuel:>4.1f}% | CO2: {co2_kg:>5.2f}kg')

        if i_episode % 10 == 0:
            # Lưu riêng biệt 3 tệp trọng số cho 3 AI
            for t in env.truck_ids:
                torch.save(agents[t].qnetwork_local.state_dict(), f'models_v20_pro/brain_{t}_ep{i_episode}.pth')
            print(f"💾 Đã lưu 3 bộ Não Độc lập vòng {i_episode} vào thư mục 'models_v20_pro/'", flush=True)
            
            plt.figure(figsize=(12, 6))
            plt.style.use('ggplot')
            colors = {'XeRac_AI_1': '#ff7f0e', 'XeRac_AI_2': '#1f77b4', 'XeRac_AI_3': '#2ca02c'}
            for t in env.truck_ids:
                smoothed = [np.mean(scores_history[t][max(0, i-10):i+1]) for i in range(len(scores_history[t]))]
                plt.plot(smoothed, label=f"{t} (Brain {t[-1]})", color=colors[t], linewidth=2)
            plt.axhline(0, color='red', linestyle='--')
            plt.title(f'TIẾN TRÌNH TU LUYỆN V20 PRO (3 BỘ NÃO ĐỘC LẬP) - Vòng {i_episode}', fontsize=14, fontweight='bold')
            plt.xlabel('Vòng (Episode)')
            plt.ylabel('Điểm số (Reward)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('models_v20_pro/training_progress.png')
            plt.close() 

    env.close()
    
    # ====================================================
    # 🏆 TỔNG KẾT CHIẾN DỊCH SAU KHI KẾT THÚC
    # ====================================================
    print("\n" + "🏆"*25, flush=True)
    print(" TỔNG KẾT CHIẾN DỊCH SAU KHI KẾT THÚC ".center(50), flush=True)
    print("🏆"*25 + "\n", flush=True)

    with open('models_v20_pro/final_report.txt', 'w', encoding='utf-8') as f:
        report = f"BÁO CÁO TỔNG KẾT CHIẾN DỊCH GREENVEIN PRO ({n_episodes} EPISODES)\n"
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
        
    # 🌟 Thêm bước vẽ biểu đồ bản Final thật đẹp
    print("🎨 Đang vẽ biểu đồ tiến trình học tập cuối cùng...", flush=True)
    plt.figure(figsize=(12, 6))
    plt.style.use('ggplot')
    colors = {'XeRac_AI_1': '#ff7f0e', 'XeRac_AI_2': '#1f77b4', 'XeRac_AI_3': '#2ca02c'}
    
    for t in env.truck_ids:
        # Đường Smoothed bản cuối lấy trung bình 20 vòng cho thật mượt
        smoothed_scores = [np.mean(scores_history[t][max(0, i-20):i+1]) for i in range(len(scores_history[t]))]
        plt.plot(smoothed_scores, label=f"{t} (Brain {t[-1]})", color=colors[t], linewidth=2, alpha=0.9)
        plt.plot(scores_history[t], color=colors[t], linewidth=1, alpha=0.15)
        
    plt.title(f'TIẾN TRÌNH HUẤN LUYỆN V20 PRO {n_episodes} VÒNG (3 Não Độc Lập)', fontsize=16, fontweight='bold')
    plt.xlabel('Vòng (Episode)', fontsize=12)
    plt.ylabel('Điểm số (Reward)', fontsize=12)
    plt.axhline(0, color='red', linestyle='--', alpha=0.8) 
    plt.legend()
    plt.tight_layout()
    plt.savefig('models_v20_pro/training_progress_final.png', dpi=300)
    print("✅ Đã lưu biểu đồ thành công tại: models_v20_pro/training_progress_final.png", flush=True)

if __name__ == "__main__":
    train_v20_decentralized()