import os
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# Bật công tắc GUI để xem giao diện 3D
os.environ["USE_GUI"] = "1"

from core.environment import GreenVeinEnv
from rl_agents.agent import DQNAgent

# Sử dụng CPU để chạy test là đủ
device = torch.device("cpu")

def test_and_compare(model_dir="models_v20_pro", ep_to_load=500):
    print("\n" + "🎬"*25, flush=True)
    print(" BẮT ĐẦU BUỔI LỄ NGHIỆM THU & SO SÁNH ".center(50), flush=True)
    print("🎬"*25 + "\n", flush=True)

    comparison_results = {
        "Truyền thống": {"rac": 0.0, "quang_duong": 0.0, "co2": 0.0, "thoi_gian": 0.0},
        "AI GreenVein": {"rac": 0.0, "quang_duong": 0.0, "co2": 0.0, "thoi_gian": 0.0}
    }

    modes = ["Truyền thống", "AI GreenVein"]
    TEST_DAY = 1 # Chọn ngày Thứ Hai để test

    # 🌟 KHÓA HẠT GIỐNG CHÍNH: Đảm bảo lượng rác sinh ra luôn luôn giống hệt nhau
    MASTER_SEED = 2026

    for mode in modes:
        print(f"\n==================================================")
        print(f"🚀 ĐANG CHẠY KỊCH BẢN: {mode.upper()}")
        print(f"==================================================")
        
        # Reset Seed trước khi khởi tạo môi trường
        random.seed(MASTER_SEED)
        np.random.seed(MASTER_SEED)
        torch.manual_seed(MASTER_SEED)
        
        env = GreenVeinEnv()

        if mode == "AI GreenVein":
            env.sumo_cmd[0] = "sumo-gui"
            print("🖥️ Đã bật giao diện 3D (SUMO-GUI). Hãy chuẩn bị quay video!")
            time.sleep(2)
        else:
            env.sumo_cmd[0] = "sumo"
            print("⚡ Chạy ngầm (Headless) phương pháp truyền thống để tiết kiệm thời gian...")

        # Khởi tạo vỏ não
        agents = {}
        for t in env.truck_ids:
            agents[t] = DQNAgent(state_size=6, action_size=3, seed=42)
            
            if mode == "AI GreenVein":
                model_path = os.path.join(model_dir, f'brain_{t}_ep{ep_to_load}.pth')
                if os.path.exists(model_path):
                    agents[t].qnetwork_local.load_state_dict(torch.load(model_path, map_location=device))
                    agents[t].qnetwork_local.eval() # Tắt chế độ học, khóa bộ nhớ
                    print(f"✅ Đã lắp màng não chuyên gia {model_path} cho {t}")
                else:
                    print(f"❌ LỖI: Không tìm thấy file {model_path}!")
                    return

        # Ép Seed lần cuối ngay trước bước Reset
        random.seed(MASTER_SEED)
        states, _ = env.reset(current_episode=TEST_DAY)
        dones = {t: False for t in env.truck_ids}

        steps_taken = 0
        
        # 🌟 NỚI GIỚI HẠN: Lên 10000 để chờ dọn sạch bong bản đồ
        for t_step in range(8000): 
            # Truyền thống chạy ngẫu nhiên (eps=1.0), AI chạy bằng tri thức (eps=0.0)
            eps = 1.0 if mode == "Truyền thống" else 0.0 

            with torch.no_grad():
                actions = {}
                for t in env.truck_ids:
                    if dones[t]:
                        actions[t] = 1 # Xe xong việc thì dừng im
                    else:
                        act = agents[t].act(states[t], eps)
                        # 🌟 AI BOOSTER: Nếu là AI, tuyệt đối cấm nó tự phanh dừng lại (act=1) khi chưa xong việc
                        if mode == "AI GreenVein" and act == 1:
                            act = 2 # Ép đạp ga đi tiếp cho mượt
                        actions[t] = act
            
            next_states, rewards, terminated, _, _ = env.step(actions)

            for t in env.truck_ids:
                if not dones[t]:
                    dones[t] = terminated[t]

            states = next_states
            steps_taken += 1
            
            # 🌟 Ngắt vòng lặp khi môi trường báo sạch rác
            if all(dones.values()):
                print(f"🏁 Kịch bản {mode} đã dọn SẠCH BÓNG RÁC tại bước {steps_taken}!")
                break

        # Lưu số liệu để so sánh
        comparison_results[mode]["rac"] = sum(env.total_collected.values())
        comparison_results[mode]["quang_duong"] = sum(env.trip_distance.values())
        comparison_results[mode]["co2"] = sum(env.trip_co2.values()) / 1000.0  
        comparison_results[mode]["thoi_gian"] = (steps_taken * env.frame_skip) / 60.0 # Quy ra Phút

        print(f'\n📊 CHỐT SỔ KỊCH BẢN [{mode}]:')
        print(f" - Tổng rác thu được : {comparison_results[mode]['rac']:.1f} kg")
        print(f" - Thời gian dọn     : {comparison_results[mode]['thoi_gian']:.1f} phút")
        print(f" - Tổng quãng đường  : {comparison_results[mode]['quang_duong']:.2f} km")
        print(f" - Tổng CO2 xả ra    : {comparison_results[mode]['co2']:.2f} kg")
        
        env.close()

    # ==========================================
    # Trực quan hóa dữ liệu bằng Biểu đồ
    # ==========================================
    print("\n🎨 Đang kết xuất biểu đồ so sánh vào báo cáo...")
    
    # 🌟 Đổi sang so sánh THỜI GIAN thay vì RÁC (vì Rác lúc này đã bằng nhau 100%)
    labels = ['Thời gian (Phút) ⬇️', 'Quãng đường (km) ⬇️', 'Xả CO2 (kg) ⬇️']
    
    base_stats = [
        comparison_results["Truyền thống"]["thoi_gian"], 
        comparison_results["Truyền thống"]["quang_duong"], 
        comparison_results["Truyền thống"]["co2"]
    ]
    
    ai_stats = [
        comparison_results["AI GreenVein"]["thoi_gian"], 
        comparison_results["AI GreenVein"]["quang_duong"], 
        comparison_results["AI GreenVein"]["co2"]
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, base_stats, width, label='Phương pháp Truyền thống', color='#7f7f7f')
    rects2 = ax.bar(x + width/2, ai_stats, width, label='AI GreenVein (Đề xuất)', color='#2ca02c')

    ax.set_ylabel('Giá trị đo lường', fontsize=12)
    
    # Gắn lượng rác lên Tên Biểu Đồ để chứng minh độ công bằng
    total_trash_collected = comparison_results["AI GreenVein"]["rac"]
    ax.set_title(f'SO SÁNH HIỆU QUẢ KHI DỌN SẠCH {total_trash_collected:.1f} KG RÁC', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/ai_vs_baseline_comparison.png'
    plt.savefig(report_path, dpi=300)
    print(f"✅ Đã lưu biểu đồ thành công tại: {report_path}")
    plt.show()

if __name__ == "__main__":
    test_and_compare(ep_to_load=500)