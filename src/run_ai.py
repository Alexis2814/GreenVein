import os
import sys
import time
import random
import traci

# 1. KẾT NỐI SUMO
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("❌ Lỗi: Máy tính chưa có biến môi trường SUMO_HOME.")

sumo_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'envs', 'greenvein.sumocfg'))
sumoCmd = ["sumo-gui", "-c", sumo_cfg_path]

def run_simulation():
    print("🚀 Đang khởi động AI...")
    traci.start(sumoCmd)
    
    start_edge = "684766065#0" 
    traci.route.add("depot_route", [start_edge])
    
    # Lấy danh sách toàn bộ con đường trên bản đồ
    all_edges = traci.edge.getIDList()
    
    for i in range(1, 4):
        truck_id = f"XeRac_AI_{i}"
        traci.vehicle.add(truck_id, "depot_route")
        
        # Sơn xe và chỉnh kích thước
        traci.vehicle.setShapeClass(truck_id, "truck")
        traci.vehicle.setLength(truck_id, 8.0)
        traci.vehicle.setColor(truck_id, (34, 139, 34)) # Màu xanh lá
        
        # 🧠 CẤP NÃO CHO AI: Bắt xe chọn 1 đích đến ngẫu nhiên thật xa để không bị biến mất
        random_destination = random.choice(all_edges)
        try:
            # Ép xe tìm đường từ Vườn hoa Thủy Lợi đến điểm ngẫu nhiên
            traci.vehicle.changeTarget(truck_id, random_destination)
        except:
            pass # Lỡ random trúng đường người đi bộ thì bỏ qua
            
        print(f"🚛 {truck_id} đã sẵn sàng xuất kích!")

    print("🛑 Dừng 5 giây để fen kịp Zoom vào Vườn hoa Thủy lợi...")
    time.sleep(5) # Cho fen 5 giây để thao tác tìm xe

    print("⏳ BẮT ĐẦU CHẠY! (Đã làm chậm tốc độ để dễ nhìn)")
    step = 0
    
    # --- ĐOẠN CODE ĐÃ ĐƯỢC NÂNG CẤP CHỐNG LỖI ---
    try:
        # Thêm hàm kiểm tra: Chỉ chạy khi SUMO còn mở và còn xe trên đường
        while step < 1000 and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            time.sleep(0.05)  # Làm chậm thời gian lại để fen xem phim
            step += 1
    except traci.exceptions.FatalTraCIError:
        print("⚠️ SUMO đã đóng cửa sổ (hoặc mô phỏng kết thúc). Dừng Python an toàn!")
    finally:
        # Đảm bảo lệnh close() luôn được gọi dù có lỗi hay không để giải phóng cổng mạng
        try:
            traci.close()
            print("🛑 Đã đóng kết nối TraCI.")
        except:
            pass
    # -------------------------------------------

if __name__ == "__main__":
    run_simulation()