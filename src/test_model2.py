import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import GreenVeinEnv
from agent import DQNAgent

def test_final_model():
    print("🎬 ĐANG KHỞI ĐỘNG CHẾ ĐỘ NGHIỆM THU TUẦN 2 (CÓ GUI)...")
    
    env = GreenVeinEnv()
    
    # 🌟 CẤU HÌNH BẬT GUI VÀ TẮT CẢNH BÁO RÁC (Làm sạch Terminal)
    env.sumo_cmd = [
        "sumo-gui", "-c", env.sumo_cfg, 
        "--start", 
        "--no-warnings", 
        "--time-to-teleport", "-1",
        "--error-log", "sumo_error_gui.log"
    ]
    
    agents = {t: DQNAgent(state_size=5, action_size=3, seed=42) for t in env.truck_ids}

    # Nạp bộ não tốt nhất đã luyện được (vòng 1000)
    for t in env.truck_ids:
        path = f'models_v2/checkpoint_{t}_ep1000.pth'
        if os.path.exists(path):
            agents[t].qnetwork_local.load_state_dict(torch.load(path))
            print(f"✅ Đã lắp bộ não Expert cho {t}")
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy file {path}. Xe {t} sẽ chạy random!")

    states, _ = env.reset()
    load_history = {t: [] for t in env.truck_ids}
    time_steps = []
    
    dones = {t: False for t in env.truck_ids}
    
    for t_step in range(2500):
        # 🌟 TẮT KHÁM PHÁ (eps = 0.0): Chỉ dùng trí khôn thực tế để chạy nghiệm thu
        actions = {t: agents[t].act(states[t], eps=0.0) if not dones[t] else 1 for t in env.truck_ids}
        
        next_states, rewards, terminated, _, _ = env.step(actions)
        
        # Ghi lại dữ liệu tải trọng để vẽ biểu đồ sawtooth (Răng cưa)
        for t in env.truck_ids:
            # Lấy % tải trọng từ state (vị trí cuối cùng trong array state)
            load_history[t].append(states[t][4])
            dones[t] = terminated[t]
        
        time_steps.append(t_step)
        states = next_states
        if all(dones.values()): break

    print("\n🏁 KẾT THÚC BUỔI NGHIỆM THU!")
    
    # 🎨 VẼ BIỂU ĐỒ CHỨNG MINH RÀNG BUỘC VẬT LÝ (TASK 2.2)
    plt.figure(figsize=(12, 5))
    for t in env.truck_ids:
        plt.plot(load_history[t], label=f"Tải trọng {t}", linewidth=2)
    
    plt.title("BIỂU ĐỒ TẢI TRỌNG XE RÁC - CHỨNG MINH RÀNG BUỘC VẬT LÝ (TUẦN 2)", fontsize=14, fontweight='bold')
    plt.xlabel("Thời gian (bước mô phỏng)", fontsize=12)
    plt.ylabel("Tải trọng (%)", fontsize=12)
    plt.axhline(95, color='r', linestyle='--', linewidth=2, label="Ngưỡng 95% (Bắt buộc quay về Trạm)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Lưu file ảnh chất lượng cao để chèn vào báo cáo
    plt.tight_layout()
    plt.savefig('models_v2/week2_physical_constraint_proof.png', dpi=300)
    print("📊 Đã lưu biểu đồ chứng minh tại: models_v2/week2_physical_constraint_proof.png")
    
    env.close()

if __name__ == "__main__":
    test_final_model()