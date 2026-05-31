import os
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

# Bật công tắc GUI để xem giao diện 3D
os.environ["USE_GUI"] = "1"

from core.environment import GreenVeinEnv
from rl_agents.agent import DQNAgent

device = torch.device("cpu")

def test_and_compare(model_dir="models_v20_pro", ep_to_load=500):
    # =================================================================
    # 🔥 BẢN VÁ V101: "TẨY NÃO" TOÀN TẬP THƯ VIỆN TRACI
    # Diệt tận gốc lỗi "Connection 'default' is already active"
    # =================================================================
    import traci
    try:
        # Không chờ đợi SUMO phản hồi để tránh lỗi đứt mạng
        traci.close(wait=False) 
    except:
        pass
        
    # Hack lõi: Lùng sục và xóa sạch mọi bóng ma 'default' trong RAM Python
    for mod_name, mod in list(sys.modules.items()):
        if 'traci' in mod_name and hasattr(mod, '_connections'):
            try:
                getattr(mod, '_connections').clear()
            except:
                pass
    # =================================================================

    print("\n" + "🎬"*25, flush=True)
    print(" BẮT ĐẦU BUỔI LỄ NGHIỆM THU & SO SÁNH ".center(50), flush=True)
    print("🎬"*25 + "\n", flush=True)

    comparison_results = {
        "Truyền thống": {"rac": 0.0, "quang_duong": 0.0, "co2": 0.0, "thoi_gian": 0.0},
        "AI GreenVein": {"rac": 0.0, "quang_duong": 0.0, "co2": 0.0, "thoi_gian": 0.0}
    }

    modes = ["Truyền thống", "AI GreenVein"]
    
    # 🔥 BƯỚC 1: KÍCH HOẠT LSTM BẰNG CÁCH CHỌN NGÀY NGẪU NHIÊN
    TEST_DAY = random.randint(0, 6)
    day_names = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    day_type = "Ngày nghỉ (Khu văn phòng ít rác)" if TEST_DAY >= 5 else "Ngày làm việc (Khu văn phòng nhiều rác)"
    
    print(f"📅 HỆ THỐNG LSTM ĐANG DỰ ĐOÁN RÁC CHO: {day_names[TEST_DAY]} - {day_type}")

    CURRENT_RUN_SEED = random.randint(1, 99999)

    for mode in modes:
        print(f"\n==================================================")
        print(f"🚀 ĐANG CHẠY KỊCH BẢN: {mode.upper()}")
        print(f"==================================================")
        
        # Ép hạt giống đồng bộ cho Truyền thống và AI
        random.seed(CURRENT_RUN_SEED)
        np.random.seed(CURRENT_RUN_SEED)
        torch.manual_seed(CURRENT_RUN_SEED)

        # 🌟 Dọn dẹp lại lần nữa ngay trước khi khởi tạo môi trường cho chắc cốp
        try: 
            traci.close(wait=False)
        except: 
            pass
        for mod_name, mod in list(sys.modules.items()):
            if 'traci' in mod_name and hasattr(mod, '_connections'):
                try: 
                    getattr(mod, '_connections').clear()
                except: 
                    pass

        # Khởi tạo môi trường an toàn
        env = GreenVeinEnv()

        if mode == "AI GreenVein":
            env.sumo_cmd[0] = "sumo-gui"
            env.sumo_cmd.extend(["--window-size", "1000,700"])
            print("🖥️ Đang triệu hồi giao diện 3D (SUMO-GUI)...")
            time.sleep(2)
        else:
            env.sumo_cmd[0] = "sumo"
            print("⚡ Chạy ngầm (Headless) phương pháp truyền thống để tiết kiệm thời gian...")

        agents = {}
        for t in env.truck_ids:
            agents[t] = DQNAgent(state_size=6, action_size=3, seed=42)
            
            if mode == "AI GreenVein":
                model_path = os.path.join(model_dir, f'brain_{t}_ep{ep_to_load}.pth')
                if os.path.exists(model_path):
                    agents[t].qnetwork_local.load_state_dict(torch.load(model_path, map_location=device))
                    agents[t].qnetwork_local.eval() 
                    print(f"✅ Đã lắp màng não DQN chuyên gia {model_path} cho {t}")
                else:
                    print(f"❌ LỖI: Không tìm thấy file {model_path}!")
                    return

        random.seed(CURRENT_RUN_SEED)
        states, _ = env.reset(current_episode=TEST_DAY)
        dones = {t: False for t in env.truck_ids}
        steps_taken = 0
        
        for t_step in range(8000): 
            eps = 1.0 if mode == "Truyền thống" else 0.0 
            with torch.no_grad():
                actions = {}
                for t in env.truck_ids:
                    if dones[t]:
                        actions[t] = 1 
                    else:
                        act = agents[t].act(states[t], eps)
                        if mode == "AI GreenVein" and act == 1: act = 2 
                        actions[t] = act
            
            next_states, rewards, terminated, _, _ = env.step(actions)
            for t in env.truck_ids:
                if not dones[t]: dones[t] = terminated[t]
            states = next_states
            steps_taken += 1
            
            if all(dones.values()):
                print(f"🏁 Kịch bản {mode} đã dọn SẠCH BÓNG RÁC tại bước {steps_taken}!")
                break

        comparison_results[mode]["rac"] = sum(env.total_collected.values())
        comparison_results[mode]["quang_duong"] = sum(env.trip_distance.values())
        comparison_results[mode]["co2"] = sum(env.trip_co2.values()) / 1000.0  
        comparison_results[mode]["thoi_gian"] = (steps_taken * env.frame_skip) / 60.0 

        print(f'\n📊 CHỐT SỔ KỊCH BẢN [{mode}]:')
        print(f" - Tổng rác thu được : {comparison_results[mode]['rac']:.1f} kg")
        print(f" - Thời gian dọn     : {comparison_results[mode]['thoi_gian']:.1f} phút")
        print(f" - Tổng quãng đường  : {comparison_results[mode]['quang_duong']:.2f} km")
        print(f" - Tổng CO2 xả ra    : {comparison_results[mode]['co2']:.2f} kg")
        
        # Chỉ đóng Truyền thống, AI giữ lại xem kết quả trên giao diện
        if mode == "Truyền thống":
            env.close()

    print("\n🎨 Đang kết xuất biểu đồ so sánh vào báo cáo...")
    labels = ['Thời gian (Phút) ⬇️', 'Quãng đường (km) ⬇️', 'Xả CO2 (kg) ⬇️']
    base_stats = [comparison_results["Truyền thống"]["thoi_gian"], comparison_results["Truyền thống"]["quang_duong"], comparison_results["Truyền thống"]["co2"]]
    ai_stats = [comparison_results["AI GreenVein"]["thoi_gian"], comparison_results["AI GreenVein"]["quang_duong"], comparison_results["AI GreenVein"]["co2"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, base_stats, width, label='Phương pháp Truyền thống', color='#7f7f7f')
    rects2 = ax.bar(x + width/2, ai_stats, width, label='AI GreenVein (Đề xuất)', color='#2ca02c')
    ax.set_ylabel('Giá trị đo lường', fontsize=12)
    
    total_trash_collected = comparison_results["AI GreenVein"]["rac"]
    ax.set_title(f'SO SÁNH HIỆU QUẢ DỌN {total_trash_collected:.1f} KG RÁC ({day_names[TEST_DAY].upper()})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/ai_vs_baseline_comparison.png'
    plt.savefig(report_path, dpi=300)
    print(f"✅ Đã lưu biểu đồ thành công tại: {report_path}")
    plt.close(fig)

if __name__ == "__main__":
    test_and_compare(ep_to_load=500)