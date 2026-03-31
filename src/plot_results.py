import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import GreenVeinEnv
from agent import DQNAgent

def evaluate_and_plot():
    print("🌟 KHỞI ĐỘNG TASK 17: KẾT XUẤT BIỂU ĐỒ BÁO CÁO TỐT NGHIỆP...")
    print("=" * 60)
    
    env = GreenVeinEnv()

    # 🌟 CHẠY NGẦM ĐỂ LẤY SỐ LIỆU SIÊU TỐC (Không bật GUI)
    env.sumo_cmd[0] = "sumo" 
    
    # Xóa các lệnh liên quan đến GUI nếu có tồn tại
    if "--gui-settings-file" in env.sumo_cmd:
        idx = env.sumo_cmd.index("--gui-settings-file")
        del env.sumo_cmd[idx:idx+2]
    if "--delay" in env.sumo_cmd:
        idx = env.sumo_cmd.index("--delay")
        del env.sumo_cmd[idx:idx+2]
        
    env.frame_skip = 50 # Tăng tốc mô phỏng tối đa

    state_size = 4
    action_size = 3
    agents = {t: DQNAgent(state_size=state_size, action_size=action_size, seed=42) for t in env.truck_ids}

    # Tự động tìm Bằng lái xịn nhất
    if not os.path.exists('models') or not os.listdir('models'):
        print("❌ LỖI: Không tìm thấy thư mục 'models'.")
        return
        
    episodes = [int(re.search(r'ep(\d+)', f).group(1)) for f in os.listdir('models') if re.search(r'ep(\d+)', f)]
    latest_episode = max(episodes) if episodes else 0
    print(f"🔍 Đang dùng Bằng lái xuất sắc nhất: Epoch {latest_episode}")

    for truck_id in env.truck_ids:
        model_path = f'models/checkpoint_{truck_id}_ep{latest_episode}.pth'
        agents[truck_id].qnetwork_local.load_state_dict(torch.load(model_path))
        agents[truck_id].qnetwork_local.eval() 

    # ==================================================
    num_eval_episodes = 10     # Sát hạch 10 vòng để có biểu đồ uy tín
    eval_eps = 0.00            # Kỷ luật thép: KHÔNG ĐI BỪA (Epsilon = 0)
    max_test_steps = 300 

    total_scores = {t: [] for t in env.truck_ids}
    success_counts = {t: 0 for t in env.truck_ids}
    total_co2 = {t: [] for t in env.truck_ids}

    for ep in range(1, num_eval_episodes + 1):
        print(f"⏳ Đang chạy ngầm mô phỏng vòng {ep}/{num_eval_episodes}...")
        states, _ = env.reset()
        dones = {t: False for t in env.truck_ids}
        scores = {t: 0.0 for t in env.truck_ids}

        for step in range(max_test_steps):
            actions = {}
            for t in env.truck_ids:
                if not dones[t]:
                    actions[t] = agents[t].act(states[t], eps=eval_eps)
                else:
                    actions[t] = 1 

            next_states, rewards, terminated, _, _ = env.step(actions)

            for t in env.truck_ids:
                if not dones[t]:
                    scores[t] += rewards[t]
                    dones[t] = terminated[t]

            states = next_states
            if all(dones.values()): break

        # Thu thập số liệu sau mỗi chuyến đi
        for t in env.truck_ids:
            total_scores[t].append(scores[t])
            if env.is_done.get(t, False):
                success_counts[t] += 1
            
            final_dist = env.trip_distance.get(t, 0.0)
            final_co2_km = (env.trip_co2[t] / final_dist) if final_dist > 0.001 else 0.0
            total_co2[t].append(final_co2_km)

    env.close()

    # ==================================================
    # 🎨 VẼ BIỂU ĐỒ BÁO CÁO VỚI MATPLOTLIB
    # ==================================================
    print("\n🎨 ĐANG KẾT XUẤT BIỂU ĐỒ...")
    
    # Thiết lập phong cách đồ thị
    plt.style.use('ggplot')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('BÁO CÁO HIỆU NĂNG AI - DỰ ÁN GREENVEIN', fontsize=18, fontweight='bold', y=1.05)

    colors = {'XeRac_AI_1': '#ff7f0e', 'XeRac_AI_2': '#1f77b4', 'XeRac_AI_3': '#2ca02c'}
    labels = {'XeRac_AI_1': 'Cụm Tây', 'XeRac_AI_2': 'Cụm Trung Tâm', 'XeRac_AI_3': 'Cụm Đông'}

    # 1️⃣ BIỂU ĐỒ 1: Biến động Điểm số (Line Chart)
    x_axis = range(1, num_eval_episodes + 1)
    for t in env.truck_ids:
        ax1.plot(x_axis, total_scores[t], marker='o', linewidth=2.5, markersize=8,
                 color=colors[t], label=labels[t])
    
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5, label='Ranh giới thất bại')
    ax1.set_title('Điểm số qua 10 vòng sát hạch', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Vòng (Episode)', fontsize=12)
    ax1.set_ylabel('Điểm số (Reward)', fontsize=12)
    ax1.set_xticks(x_axis)
    ax1.legend()

    # 2️⃣ BIỂU ĐỒ 2: Tỷ lệ Hoàn thành (Bar Chart)
    rates = [(success_counts[t] / num_eval_episodes) * 100 for t in env.truck_ids]
    bars2 = ax2.bar([labels[t] for t in env.truck_ids], rates, color=[colors[t] for t in env.truck_ids], width=0.6)
    ax2.set_title('Tỷ lệ Hoàn thành Nhiệm vụ (%)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 115) 
    
    # Ghi số % lên đỉnh cột
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', 
                 ha='center', va='bottom', fontweight='bold', fontsize=12)

    # 3️⃣ BIỂU ĐỒ 3: Lượng CO2 Trung bình (Bar Chart)
    avg_co2_list = [np.mean(total_co2[t]) for t in env.truck_ids]
    bars3 = ax3.bar([labels[t] for t in env.truck_ids], avg_co2_list, color=[colors[t] for t in env.truck_ids], width=0.6)
    ax3.set_title('Xả thải CO2 Trung bình (g/km)', fontsize=14, fontweight='bold')
    
    # Tự động điều chỉnh trục Y theo lượng CO2 cao nhất
    max_co2 = max(avg_co2_list) if max(avg_co2_list) > 0 else 100
    ax3.set_ylim(0, max_co2 * 1.2) 

    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + (max_co2*0.02), f'{yval:.1f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    
    # Lưu ảnh tự động ra thư mục gốc
    output_img = 'BaoCao_AI_GreenVein.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu ảnh siêu nét: '{output_img}' tại thư mục gốc.")
    
    # Hiển thị lên màn hình
    plt.show()

if __name__ == "__main__":
    # Nhớ cài đặt thư viện trước nếu chưa có: pip install matplotlib
    evaluate_and_plot()