import os
import re
import torch
import numpy as np
import pandas as pd  # Import thêm thư viện pandas để xuất Excel/CSV
import traci  
import matplotlib.pyplot as plt
from environment import GreenVeinEnv
from agent import DQNAgent

def run_full_evaluation():
    print(" BẮT ĐẦU CHIẾN DỊCH TỔNG LỰC: MÔ PHỎNG TRỰC QUAN & XUẤT BIỂU ĐỒ")
    print("=" * 65)

    env = GreenVeinEnv()
    
    # 🌟 1. CẤU HÌNH SUMO-GUI (Có Delay và File Giao diện)
    env.sumo_cmd[0] = "sumo-gui"
    view_settings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'envs', 'viewsettings.xml'))

    if "--start" not in env.sumo_cmd:
        env.sumo_cmd.extend([
            "--start", 
            "--quit-on-end", 
            "--delay", "100", # Chạy mượt mà vừa mắt
            "--gui-settings-file", view_settings_path
        ])
        
    env.frame_skip = 10 

    state_size = 5
    action_size = 3
    agents = {t: DQNAgent(state_size=state_size, action_size=action_size, seed=42) for t in env.truck_ids}

    # Tự động load Bằng lái
    if not os.path.exists('models') or not os.listdir('models'):
        print(" LỖI: Không tìm thấy thư mục 'models'.")
        return
        
    episodes = [int(re.search(r'ep(\d+)', f).group(1)) for f in os.listdir('models') if re.search(r'ep(\d+)', f)]
    latest_episode = max(episodes) if episodes else 0
    print(f" Đang nạp Bằng lái xuất sắc nhất: Epoch {latest_episode}")

    for truck_id in env.truck_ids:
        model_path = f'models/checkpoint_{truck_id}_ep{latest_episode}.pth'
        agents[truck_id].qnetwork_local.load_state_dict(torch.load(model_path))
        agents[truck_id].qnetwork_local.eval() 

    # ==================================================
    num_eval_episodes = 5      # Chạy 5 vòng có GUI để lấy số liệu
    eval_eps = 0.00            # Kỷ luật thép: Epsilon = 0 (100% Kỹ năng thực)
    max_test_steps = 300 

    total_scores = {t: [] for t in env.truck_ids}
    success_counts = {t: 0 for t in env.truck_ids}
    total_co2 = {t: [] for t in env.truck_ids}

    #  2. CHẠY MÔ PHỎNG TRỰC QUAN
    for ep in range(1, num_eval_episodes + 1):
        print(f"\n ĐANG CHẠY BÀI TEST SỐ {ep}/{num_eval_episodes}...")
        states, _ = env.reset()
        
        try:
            traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
            traci.gui.setZoom(traci.gui.DEFAULT_VIEW, 500)
        except Exception:
            pass

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

        # Lưu trữ số liệu sau mỗi vòng
        for t in env.truck_ids:
            total_scores[t].append(scores[t])
            if env.is_done.get(t, False):
                success_counts[t] += 1
                
            final_dist = env.trip_distance.get(t, 0.0)
            final_co2_km = (env.trip_co2[t] / final_dist) if final_dist > 0.001 else 0.0
            total_co2[t].append(final_co2_km)
            
            status = "Hoàn thành " if env.is_done.get(t, False) else "Bỏ cuộc/Kẹt xe "
            print(f"   [{t}] Điểm: {scores[t]:>7.2f} | {status}")

    env.close()

    #  3. BÁO CÁO TERMINAL
    print("\n" + "=" * 18 + " BÁO CÁO ĐÁNH GIÁ CHUYÊN SÂU (5 VÒNG) " + "=" * 18)
    print(f"{'Tài xế AI':<15} | {'Điểm TB':<10} | {'Tỷ lệ Hoàn thành':<18} | {'CO2 TB (g/km)':<12}")
    print("-" * 70)
    for t in env.truck_ids:
        print(f"{t:<15} | {np.mean(total_scores[t]):>10.2f} | {(success_counts[t]/num_eval_episodes)*100:>17.1f}% | {np.mean(total_co2[t]):>12.1f}")
    print("=" * 70)

    #  4. VẼ BIỂU ĐỒ BÁO CÁO MATPLOTLIB
    print("\n ĐANG KẾT XUẤT BIỂU ĐỒ TỔNG HỢP...")
    plt.style.use('ggplot')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('BÁO CÁO HIỆU NĂNG AI - DỰ ÁN GREENVEIN', fontsize=18, fontweight='bold', y=1.05)

    colors = {'XeRac_AI_1': '#ff7f0e', 'XeRac_AI_2': '#1f77b4', 'XeRac_AI_3': '#2ca02c'}
    labels = {'XeRac_AI_1': 'Cụm Tây', 'XeRac_AI_2': 'Cụm Trung Tâm', 'XeRac_AI_3': 'Cụm Đông'}

    # 1️⃣ Biểu đồ Đường (Điểm số)
    x_axis = range(1, num_eval_episodes + 1)
    for t in env.truck_ids:
        ax1.plot(x_axis, total_scores[t], marker='o', linewidth=2.5, markersize=8, color=colors[t], label=labels[t])
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5, label='Ranh giới thất bại')
    ax1.set_title('Điểm số qua 5 vòng sát hạch', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Vòng (Episode)', fontsize=12)
    ax1.set_ylabel('Điểm số (Reward)', fontsize=12)
    ax1.set_xticks(x_axis)
    ax1.legend()

    # 2️⃣ Biểu đồ Cột (Tỷ lệ hoàn thành)
    rates = [(success_counts[t] / num_eval_episodes) * 100 for t in env.truck_ids]
    bars2 = ax2.bar([labels[t] for t in env.truck_ids], rates, color=[colors[t] for t in env.truck_ids], width=0.6)
    ax2.set_title('Tỷ lệ Hoàn thành Nhiệm vụ (%)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 115) 
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # 3️⃣ Biểu đồ Cột (Xả thải CO2)
    avg_co2_list = [np.mean(total_co2[t]) for t in env.truck_ids]
    bars3 = ax3.bar([labels[t] for t in env.truck_ids], avg_co2_list, color=[colors[t] for t in env.truck_ids], width=0.6)
    ax3.set_title('Xả thải CO2 Trung bình (g/km)', fontsize=14, fontweight='bold')
    max_co2 = max(avg_co2_list) if max(avg_co2_list) > 0 else 100
    ax3.set_ylim(0, max_co2 * 1.2) 
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + (max_co2*0.02), f'{yval:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    output_img = 'BaoCao_AI_GreenVein_Final.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f" Đã lưu ảnh siêu nét: '{output_img}' tại thư mục gốc.")

    # 🌟 5. XUẤT DỮ LIỆU THÔ (RAW DATA) RA FILE CSV
    print("\n ĐANG LƯU DỮ LIỆU THÔ (RAW DATA) RA EXCEL/CSV...")
    report_data = {
        "Khu Vực": [labels[t] for t in env.truck_ids],
        "Mã Xe AI": list(env.truck_ids),
        "Điểm TB": [np.mean(total_scores[t]) for t in env.truck_ids],
        "Tỷ lệ Hoàn thành (%)": [(success_counts[t] / num_eval_episodes) * 100 for t in env.truck_ids],
        "Xả thải CO2 TB (g/km)": [np.mean(total_co2[t]) for t in env.truck_ids]
    }
    df = pd.DataFrame(report_data)
    df = df.round({"Điểm TB": 2, "Xả thải CO2 TB (g/km)": 2})
    csv_filename = "GreenVein_RawData_Report.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f" Đã lưu Raw Data: '{csv_filename}' thành công!")
    plt.show()
if __name__ == "__main__":
    run_full_evaluation()