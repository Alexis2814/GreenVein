import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import GreenVeinEnv
from agent import DQNAgent

def test_v20_master():
    print("🎬 ĐANG KHỞI ĐỘNG CHẾ ĐỘ NGHIỆM THU V20 (CÓ GUI)...")
    
    env = GreenVeinEnv()
    
    # 🌟 CẤU HÌNH BẬT GUI VÀ TẮT CẢNH BÁO
    env.sumo_cmd = [
        "sumo-gui", "-c", env.sumo_cfg, 
        "--start", 
        "--no-warnings", 
        "--time-to-teleport", "-1",
        "--error-log", "sumo_error_gui.log"
    ]
    
    # 🌟 CHÚ Ý: state_size=6 cho hệ thống có kim xăng
    agents = {t: DQNAgent(state_size=6, action_size=3, seed=42) for t in env.truck_ids}

    # Nạp bộ não Master xịn nhất (vòng 500)
    for t in env.truck_ids:
        path = f'models_v20/master_{t}_ep500.pth'
        if os.path.exists(path):
            agents[t].qnetwork_local.load_state_dict(torch.load(path))
            print(f"✅ Đã lắp bộ não Master (Ep 500) cho {t}")
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy file {path}. Xe {t} sẽ chạy random!")

    states, _ = env.reset()
    
    # 🌟 Theo dõi lịch sử xăng để vẽ biểu đồ báo cáo
    fuel_history = {t: [] for t in env.truck_ids}
    time_steps = []
    
    dones = {t: False for t in env.truck_ids}
    
    print("\n🚀 BẮT ĐẦU CHẠY THỰC TẾ (100% TRÍ KHÔN, KHÔNG RANDOM)...")
    for t_step in range(2500):
        # eps = 0.0 -> AI sẽ đưa ra quyết định tốt nhất dựa trên những gì đã học
        actions = {t: agents[t].act(states[t], eps=0.0) if not dones[t] else 1 for t in env.truck_ids}
        
        next_states, rewards, terminated, _, _ = env.step(actions)
        
        # Ghi nhận biến động xăng
        for t in env.truck_ids:
            if not dones[t]:
                fuel_history[t].append(env.current_fuel[t])
            else:
                if len(fuel_history[t]) > 0:
                    fuel_history[t].append(fuel_history[t][-1]) # Giữ nguyên vạch xăng nếu xe đã nghỉ
                
            dones[t] = terminated[t]
        
        time_steps.append(t_step)
        states = next_states
        if all(dones.values()): break

    print("\n🏁 KẾT THÚC BUỔI NGHIỆM THU V20!")
    
    # 🎨 VẼ BIỂU ĐỒ CHỨNG MINH HÀNH VI ĐỔ XĂNG
    plt.figure(figsize=(12, 5))
    colors = {'XeRac_AI_1': '#ff7f0e', 'XeRac_AI_2': '#1f77b4', 'XeRac_AI_3': '#2ca02c'}
    
    for t in env.truck_ids:
        plt.plot(time_steps, fuel_history[t], label=f"Mức xăng {t}", color=colors[t], linewidth=2)
        
    plt.title("BIỂU ĐỒ TIÊU HAO & BƠM XĂNG THỰC TẾ (V20)", fontsize=14, fontweight='bold')
    plt.xlabel("Thời gian (bước mô phỏng)", fontsize=12)
    plt.ylabel("Nhiên liệu (%)", fontsize=12)
    plt.axhline(30, color='orange', linestyle='--', linewidth=1.5, label="Ngưỡng tìm trạm (<30%)")
    plt.axhline(15, color='red', linestyle='--', linewidth=2, label="Ngưỡng khẩn cấp (<15%)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('models_v20/v20_fuel_behavior.png', dpi=300)
    print("📊 Đã lưu biểu đồ chứng minh hành vi đổ xăng tại: models_v20/v20_fuel_behavior.png")
    
    env.close()

if __name__ == "__main__":
    test_v20_master()