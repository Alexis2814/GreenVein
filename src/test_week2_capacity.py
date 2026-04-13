import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import GreenVeinEnv
from agent import DQNAgent

def run_acceptance_test():
    print("🎬 ĐANG KHỞI ĐỘNG CHẾ ĐỘ NGHIỆM THU TUẦN 2 (RÀNG BUỘC VẬT LÝ)...")
    
    env = GreenVeinEnv()
    
    # Bật SUMO GUI để xem tận mắt
    env.sumo_cmd = [
        "sumo-gui", "-c", env.sumo_cfg, 
        "--start", "--no-warnings", "--time-to-teleport", "-1"
    ]
    
    # Khởi tạo Agent (Chú ý state_size=6 như đã nâng cấp)
    agents = {t: DQNAgent(state_size=6, action_size=3, seed=42) for t in env.truck_ids}

    # Tải trọng số tốt nhất mà fen vừa huấn luyện xong (thay đổi Ep cho đúng)
    for t in env.truck_ids:
        path = f'models_v20/master_{t}_ep500.pth'
        if os.path.exists(path):
            agents[t].qnetwork_local.load_state_dict(torch.load(path))
            print(f"✅ Đã nạp não bộ thành công cho {t}")
        else:
            print(f"⚠️ Cảnh báo: Chưa có file weights cho {t}. Xe sẽ chạy random.")

    states, _ = env.reset()
    
    # 🌟 BIẾN THEO DÕI TẢI TRỌNG (Vũ khí nghiệm thu)
    load_history = {t: [] for t in env.truck_ids}
    time_steps = []
    dones = {t: False for t in env.truck_ids}
    
    print("\n🚀 BẮT ĐẦU CHẠY THỰC TẾ (100% Khai thác, Không Random)...")
    for t_step in range(2500):
        # eps=0.0 bắt AI dùng 100% kinh nghiệm đã học
        actions = {t: agents[t].act(states[t], eps=0.0) if not dones[t] else 1 for t in env.truck_ids}
        
        next_states, rewards, terminated, _, _ = env.step(actions)
        
        # Ghi chép lại tải trọng từng bước của từng xe
        for t in env.truck_ids:
            if not dones[t]:
                # Truy xuất trực tiếp biến current_load từ môi trường
                load_history[t].append(env.current_load[t])
            else:
                if len(load_history[t]) > 0:
                    load_history[t].append(load_history[t][-1])
                
            dones[t] = terminated[t]
        
        time_steps.append(t_step)
        states = next_states
        if all(dones.values()): break

    print("\n🏁 KẾT THÚC NGHIỆM THU!")
    
    # ==========================================
    # 🎨 VẼ BIỂU ĐỒ BÁO CÁO NGHIỆM THU
    # ==========================================
    plt.figure(figsize=(12, 6))
    colors = {'XeRac_AI_1': '#ff7f0e', 'XeRac_AI_2': '#1f77b4', 'XeRac_AI_3': '#2ca02c'}
    
    for t in env.truck_ids:
        plt.plot(time_steps, load_history[t], label=f"Tải trọng {t}", color=colors[t], linewidth=2, alpha=0.8)
        
    plt.title("BIỂU ĐỒ CHỨNG MINH RÀNG BUỘC VẬT LÝ (TUẦN 2)", fontsize=14, fontweight='bold')
    plt.xlabel("Thời gian (Bước mô phỏng)", fontsize=12)
    plt.ylabel("Tải trọng rác trên xe (kg)", fontsize=12)
    
    # Vẽ đường giới hạn đỏ
    plt.axhline(env.MAX_CAPACITY_KG, color='red', linestyle='--', linewidth=2, label=f"Giới hạn thùng ({env.MAX_CAPACITY_KG} kg)")
    
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    # Lưu file nộp báo cáo
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/week2_capacity_acceptance.png'
    plt.savefig(report_path, dpi=300)
    print(f"📊 Đã lưu Biểu đồ Nghiệm thu tại: {report_path}")
    
    env.close()

if __name__ == "__main__":
    run_acceptance_test()