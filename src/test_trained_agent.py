import os
import re
import torch
import numpy as np
import traci  # Import thêm traci để can thiệp giao diện
from environment import GreenVeinEnv
from agent import DQNAgent

def evaluate_trained_model():
    print("🌟 BẮT ĐẦU KIỂM THỬ AI NÂNG CAO (STATISTICAL EVALUATION)")
    print("=" * 60)

    env = GreenVeinEnv()
    env.sumo_cmd[0] = "sumo-gui"
    view_settings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'envs', 'viewsettings.xml'))

    if "--start" not in env.sumo_cmd:
        env.sumo_cmd.extend([
            "--start", 
            "--quit-on-end", 
            "--delay", "100",
            "--gui-settings-file", view_settings_path  # 🌟 Lệnh nạp file giao diện
        ])
        
    env.frame_skip = 10 

    state_size = 5
    action_size = 3
    agents = {t: DQNAgent(state_size=state_size, action_size=action_size, seed=42) for t in env.truck_ids}

    if not os.path.exists('models') or not os.listdir('models'):
        print("❌ LỖI: Không tìm thấy thư mục 'models'.")
        return
        
    episodes = [int(re.search(r'ep(\d+)', f).group(1)) for f in os.listdir('models') if re.search(r'ep(\d+)', f)]
    latest_episode = max(episodes) if episodes else 0
    print(f"🔍 Tự động chọn Bằng lái kinh nghiệm nhất: Epoch {latest_episode}")

    for truck_id in env.truck_ids:
        model_path = f'models/checkpoint_{truck_id}_ep{latest_episode}.pth'
        agents[truck_id].qnetwork_local.load_state_dict(torch.load(model_path))
        agents[truck_id].qnetwork_local.eval() 

    # ==================================================
    num_eval_episodes = 5      
    eval_eps = 0.05            
    max_test_steps = 300 

    total_scores = {truck_id: [] for truck_id in env.truck_ids}
    success_counts = {truck_id: 0 for truck_id in env.truck_ids}
    # 🌟 Thêm dictionary để lưu trữ tổng lượng CO2 trung bình
    total_co2_emissions = {truck_id: [] for truck_id in env.truck_ids}

    for ep in range(1, num_eval_episodes + 1):
        print(f"\n🚀 ĐANG CHẠY BÀI TEST SỐ {ep}/{num_eval_episodes}...")
        states, _ = env.reset()
        
        # 🌟 CÀI ĐẶT GIAO DIỆN REAL WORLD & ZOOM VÀO BẢN ĐỒ
        try:
            traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
            traci.gui.setZoom(traci.gui.DEFAULT_VIEW, 500)
        except Exception:
            pass

        dones = {truck_id: False for truck_id in env.truck_ids}
        scores = {truck_id: 0.0 for truck_id in env.truck_ids}

        for step in range(max_test_steps):
            actions = {}
            for truck_id in env.truck_ids:
                if not dones[truck_id]:
                    actions[truck_id] = agents[truck_id].act(states[truck_id], eps=eval_eps)
                else:
                    actions[truck_id] = 1 

            next_states, rewards, terminated, _, _ = env.step(actions)

            for truck_id in env.truck_ids:
                if not dones[truck_id]:
                    scores[truck_id] += rewards[truck_id]
                    dones[truck_id] = terminated[truck_id]

            states = next_states

            if all(dones.values()):
                break

        for truck_id in env.truck_ids:
            total_scores[truck_id].append(scores[truck_id])
            if env.is_done.get(truck_id, False):
                success_counts[truck_id] += 1
                
            # 🌟 Tính toán CO2 trung bình cho vòng này và lưu lại
            final_dist = env.trip_distance.get(truck_id, 0.0)
            final_co2_km = (env.trip_co2[truck_id] / final_dist) if final_dist > 0.001 else 0.0
            total_co2_emissions[truck_id].append(final_co2_km)
            
            status = "Hoàn thành 🟢" if env.is_done.get(truck_id, False) else "Bỏ cuộc/Kẹt xe 🔴"
            print(f"   [{truck_id}] Điểm: {scores[truck_id]:>7.2f} | {status}")

    env.close()

    # 🌟 Cập nhật Bảng Báo cáo Đánh giá Chuyên sâu (Có CO2)
    print("\n" + "=" * 18 + " BÁO CÁO ĐÁNH GIÁ CHUYÊN SÂU (5 VÒNG) " + "=" * 18)
    print(f"{'Tài xế AI':<15} | {'Điểm TB':<10} | {'Tỷ lệ Hoàn thành':<18} | {'CO2 TB (g/km)':<12}")
    print("-" * 70)
    for truck_id in env.truck_ids:
        avg_score = np.mean(total_scores[truck_id])
        success_rate = (success_counts[truck_id] / num_eval_episodes) * 100
        avg_co2 = np.mean(total_co2_emissions[truck_id])
        print(f"{truck_id:<15} | {avg_score:>10.2f} | {success_rate:>17.1f}% | {avg_co2:>12.1f}")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_trained_model()