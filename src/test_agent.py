import os
import time
import torch
import numpy as np
from environment import GreenVeinEnv
from agent import DQNAgent

# Không cần GPU để chạy nghiệm thu, CPU là quá đủ sức mạnh rồi
device = torch.device("cpu")

def test_trained_agents(episodes=1, model_dir="models_v20_pro", ep_to_load=500):
    print("\n" + "🎬"*25, flush=True)
    print(" BẮT ĐẦU BUỔI LỄ NGHIỆM THU (TESTING) ".center(50), flush=True)
    print("🎬"*25 + "\n", flush=True)

    env = GreenVeinEnv()

    # 🌟 ÉP BẬT GIAO DIỆN ĐỒ HỌA (GUI) ĐỂ XEM PHIM
    if env.sumo_cmd[0] == "sumo":
        env.sumo_cmd[0] = "sumo-gui"
    
    # Khởi tạo vỏ não trống
    agents = {}
    for t in env.truck_ids:
        agents[t] = DQNAgent(state_size=6, action_size=3, seed=42)
        
        # 🌟 LẮP NÃO ĐÃ ĐƯỢC HUẤN LUYỆN VÀO (Lấy mốc vòng 500)
        model_path = os.path.join(model_dir, f'brain_{t}_ep{ep_to_load}.pth')
        if os.path.exists(model_path):
            agents[t].qnetwork_local.load_state_dict(torch.load(model_path, map_location=device))
            agents[t].qnetwork_local.eval() # Khóa bộ nhớ, chỉ thực thi xuất chiêu
            print(f"✅ Đã lắp màng não {model_path} cho {t}")
        else:
            print(f"❌ KHÔNG TÌM THẤY NÃO: {model_path}! Kỹ sư trưởng kiểm tra lại nhé.")
            return

    print("\n🚀 Bắt đầu mô phỏng...")
    print("💡 MẸO: Trên cửa sổ SUMO GUI, hãy chỉnh thanh 'Delay' (Trễ) lên khoảng 100ms để xe không chạy lướt qua quá nhanh nhé!")
    time.sleep(2)

    for i_episode in range(1, episodes + 1):
        states, _ = env.reset()
        scores = {t: 0.0 for t in env.truck_ids}
        dones = {t: False for t in env.truck_ids}

        for t_step in range(800): # 800 bước ~ hết Ca Đêm
            
            # 🌟 TẮT HOÀN TOÀN SỰ NGẪU NHIÊN (Epsilon = 0)
            eps = 0.0 

            with torch.no_grad():
                actions = {t: agents[t].act(states[t], eps) if not dones[t] else 1 for t in env.truck_ids}
            
            next_states, rewards, terminated, _, _ = env.step(actions)

            for t in env.truck_ids:
                if not dones[t]:
                    scores[t] += rewards[t]
                    dones[t] = terminated[t]

            states = next_states
            
            if all(dones.values()):
                break

        print(f'\n📊 TỔNG KẾT BUỔI NGHIỆM THU:')
        print("=" * 70)
        for t in env.truck_ids:
            dist = env.trip_distance.get(t, 0.0)
            fuel = env.current_fuel.get(t, 0.0)
            co2_kg = env.trip_co2.get(t, 0.0) / 1000.0 
            zone = env.zone_names.get(t, "Không rõ")
            print(f'🚛 [{t}] ({zone}) - Điểm AI: {scores[t]:>7.2f} | Rác thu được: {env.total_collected.get(t,0):>5.1f}kg')
            print(f'   ↳ Quãng đường: {dist:>5.2f}km | Xăng còn: {fuel:>4.1f}% | Xả CO2: {co2_kg:>5.2f}kg')
            print("-" * 70)

    env.close()
    print("\n🎉 BUỔI NGHIỆM THU KẾT THÚC THÀNH CÔNG TỐT ĐẸP!")

if __name__ == "__main__":
    test_trained_agents()