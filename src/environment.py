import os
import sys
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# =====================================================================
# 🌟 BẬT CHẾ ĐỘ GUI ĐỂ QUAY PHIM: Ép dùng traci thay vì libsumo
# =====================================================================
# try:
#     import libsumo as traci
#     print("🚀 [HỆ THỐNG] Đã nạp thành công LIBSUMO - Chế độ Siêu Tốc được kích hoạt!")
# except ImportError:
import traci
print("🖥️ [HỆ THỐNG] Ép dùng TraCI - Chế độ giao diện đồ họa (GUI) đã mở!")
# =====================================================================

from waste_generator import RealWasteGenerator

class GreenVeinEnv(gym.Env):
    def __init__(self):
        super(GreenVeinEnv, self).__init__()
        
        self.sumo_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'envs', 'greenvein.sumocfg'))
        
        self.sumo_cmd = [
            "sumo", "-c", self.sumo_cfg, 
            "--no-warnings", 
            "--time-to-teleport", "-1", 
            "--error-log", "sumo_error.log",
            "--no-step-log",
        ]
        
        self.truck_ids = ["XeRac_AI_1", "XeRac_AI_2", "XeRac_AI_3"]
        self.depot_edge = "684766065#0"
        self.depot_names = {"684766065#0": "Trạm Điều hành GreenVein"}
        
        self.zone_names = {
            "XeRac_AI_1": "Cụm Tây",
            "XeRac_AI_2": "Cụm Trung Tâm",
            "XeRac_AI_3": "Cụm Đông"
        }

        self.frame_skip = 10 
        self.action_space = spaces.Discrete(3)
        # 🌟 NÂNG CẤP STATE SIZE LÊN 6: Thêm thông số Nhiên liệu (Fuel)
        self.observation_space = spaces.Box(low=0.0, high=10000.0, shape=(6,), dtype=np.float32)
        
        self.MAX_CAPACITY_KG = 5000.0 
        self.BIN_MAX_WEIGHT_KG = 200.0 
        self.MAX_FUEL = 100.0 # 🌟 Dung tích bình xăng 100%

    def assign_urgent_target(self, truck_id):
        try:
            current_edge = traci.vehicle.getRoadID(truck_id)
        except:
            current_edge = self.depot_edge
            
        if not current_edge or current_edge == "":
            current_edge = self.depot_edge
            
        # =========================================================================
        # 🌟 THUẬT TOÁN IOT TỐI ƯU V20: VRP & CROSS-ZONE
        # =========================================================================
        red_bins = [b for b in self.zone_bins[truck_id] if b != current_edge and self.bin_levels[b] >= 70.0]
        yellow_bins = [b for b in self.zone_bins[truck_id] if b != current_edge and 40.0 <= self.bin_levels[b] < 70.0]
        
        # 🌟 NÂNG CẤP 1: HỢP TÁC LIÊN KHU VỰC (CỨU BỒ)
        if not red_bins:
            for other_truck in self.truck_ids:
                if other_truck != truck_id:
                    # Nếu thấy cụm khác có thùng sắp nổ (>85%), nhảy sang cứu ngay
                    help_bins = [b for b in self.zone_bins[other_truck] if b != current_edge and self.bin_levels[b] >= 85.0]
                    red_bins.extend(help_bins)

        candidate_bins = []
        if len(red_bins) > 0:
            candidate_bins = sorted(red_bins, key=lambda b: self.bin_levels[b], reverse=True)[:5]
        elif len(yellow_bins) > 0 and self.current_load[truck_id] < self.MAX_CAPACITY_KG * 0.5:
            candidate_bins = sorted(yellow_bins, key=lambda b: self.bin_levels[b], reverse=True)[:3]
        else:
            try:
                traci.vehicle.changeTarget(truck_id, self.depot_edge)
                self.is_heading_depot[truck_id] = True
                return True
            except:
                pass
        
        # 🌟 NÂNG CẤP 2: ĐỊNH TUYẾN VRP (CHỌN ĐIỂM GẦN NHẤT ĐỂ ĐI)
        valid_targets = []
        for target_bin in candidate_bins:
            try:
                route = traci.simulation.findRoute(current_edge, target_bin, vType="garbage_truck")
                if route and len(route.edges) > 0:
                    valid_targets.append((target_bin, len(route.edges))) # Lưu lại số cạnh (edges) phải đi
            except:
                continue

        if valid_targets:
            # Sắp xếp để ưu tiên thùng rác tốn ít đường đi nhất
            valid_targets.sort(key=lambda x: x[1])
            best_bin = valid_targets[0][0]
            
            try:
                route_to_bin = traci.simulation.findRoute(current_edge, best_bin, vType="garbage_truck")
                route_to_depot = traci.simulation.findRoute(best_bin, self.depot_edge, vType="garbage_truck")
                
                if route_to_depot and len(route_to_depot.edges) > 0:
                    full_route = list(route_to_bin.edges) + list(route_to_depot.edges)[1:]
                    traci.vehicle.setRoute(truck_id, full_route)
                else:
                    traci.vehicle.changeTarget(truck_id, best_bin)
                return True 
            except:
                pass

        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try: traci.close()
        except: pass
            
        traci.start(self.sumo_cmd)
        
        start_hour = random.choice([5, 10, 14, 18])
        self.virtual_time_seconds = start_hour * 3600 
        self.current_hour_int = start_hour
        try: traci.simulation.setScale(0.1)
        except: pass

        self.current_step = 0
        all_edges = traci.edge.getIDList()
        valid_edges = []
        
        for edge_id in all_edges:
            if edge_id.startswith(":"): continue
            try:
                allowed = traci.lane.getAllowed(edge_id + "_0")
                disallowed = traci.lane.getDisallowed(edge_id + "_0")
                if len(allowed) == 0:
                    if "delivery" not in disallowed: valid_edges.append(edge_id)
                elif "delivery" in allowed:
                    valid_edges.append(edge_id)
            except: continue

        edge_x_coords = {}
        for edge_id in valid_edges:
            try:
                shape = traci.lane.getShape(edge_id + "_0") 
                avg_x = sum([point[0] for point in shape]) / len(shape)
                edge_x_coords[edge_id] = avg_x
            except:
                edge_x_coords[edge_id] = 0.0
                
        valid_edges_sorted = sorted(valid_edges, key=lambda e: edge_x_coords[e])
        chunk = len(valid_edges_sorted) // 3
        
        target_zones = {
            "XeRac_AI_1": valid_edges_sorted[:chunk],
            "XeRac_AI_2": valid_edges_sorted[chunk:2*chunk],
            "XeRac_AI_3": valid_edges_sorted[2*chunk:]
        }

        self.bin_levels = {}
        self.generators = {}
        self.zone_bins = {truck_id: [] for truck_id in self.truck_ids}

        for truck_id, edges in target_zones.items():
            sampled_bins = random.sample(edges, min(20, len(edges)))
            self.zone_bins[truck_id] = sampled_bins
            zone_type = "commercial" if truck_id == "XeRac_AI_2" else "residential"
            
            for b in sampled_bins:
                self.bin_levels[b] = random.uniform(10.0, 60.0) 
                self.generators[b] = RealWasteGenerator(zone_type=zone_type)

        traci.route.add("depot_route", [self.depot_edge])
        
        if "garbage_truck" not in traci.vehicletype.getIDList():
            traci.vehicletype.copy("DEFAULT_VEHTYPE", "garbage_truck")
            traci.vehicletype.setVehicleClass("garbage_truck", "delivery")
            traci.vehicletype.setShapeClass("garbage_truck", "truck")

        self.current_load = {truck_id: 0.0 for truck_id in self.truck_ids}
        self.current_fuel = {truck_id: self.MAX_FUEL for truck_id in self.truck_ids} 
        self.is_heading_depot = {truck_id: False for truck_id in self.truck_ids}
        self.is_done = {truck_id: False for truck_id in self.truck_ids} 
        self.aborted = {truck_id: False for truck_id in self.truck_ids} 
        self.stuck_time = {truck_id: 0 for truck_id in self.truck_ids}
        self.trip_co2 = {truck_id: 0.0 for truck_id in self.truck_ids}
        self.trip_distance = {truck_id: 0.0 for truck_id in self.truck_ids}
        self.total_collected = {truck_id: 0.0 for truck_id in self.truck_ids} 
        self.last_visited_bin = {truck_id: "" for truck_id in self.truck_ids} 

        for truck_id in self.truck_ids:
            traci.vehicle.add(truck_id, "depot_route", typeID="garbage_truck")
            if truck_id == "XeRac_AI_1": traci.vehicle.setColor(truck_id, (255, 69, 0))   
            elif truck_id == "XeRac_AI_2": traci.vehicle.setColor(truck_id, (30, 144, 255)) 
            else: traci.vehicle.setColor(truck_id, (50, 205, 50))                        

        traci.simulationStep()

        for truck_id in self.truck_ids:
            if self.zone_bins[truck_id]:
                self.assign_urgent_target(truck_id)
        
        print("\n" + "=".center(75, "="))
        print(" GREENVEIN V20 - ĐẲNG CẤP THỰC TẾ (VRP, FUEL & CO-OP) ".center(75, "="))
        print("=".center(75, "="))
        return {truck_id: np.zeros(6, dtype=np.float32) for truck_id in self.truck_ids}, {}

    def step(self, action_dict):
        step_rewards = {truck_id: 0.0 for truck_id in self.truck_ids}

        self.virtual_time_seconds = (self.virtual_time_seconds + self.frame_skip) % 86400
        hour = self.virtual_time_seconds / 3600.0
        
        new_hour_int = int(hour)
        if new_hour_int != self.current_hour_int:
            self.current_hour_int = new_hour_int
            if 7 <= hour < 9: scale = 1.2 
            elif 17 <= hour < 19.5: scale = 1.5 
            elif 9 <= hour < 17: scale = 0.5 
            else: scale = 0.1 
            try: traci.simulation.setScale(scale)
            except: pass

        hour_display = int(hour)
        minute_display = int((hour - hour_display) * 60)
        time_str = f"{hour_display:02d}:{minute_display:02d}"
        
        # 🌟 Xác định Giờ cao điểm / Cấm tải (7h-8h sáng, 17h-18h chiều)
        is_rush_hour = (7 <= hour < 8) or (17 <= hour < 18)
        
        if 7 <= hour < 9: traffic_status = "🟠 ÙN Ứ"
        elif 17 <= hour < 19.5: traffic_status = "🔴 TẮC CỨNG"
        elif 9 <= hour < 17: traffic_status = "🟡 TRUNG BÌNH"
        else: traffic_status = "🟢 VẮNG VẺ"

        # Hiển thị log Radar IoT
        if self.current_step > 0 and self.current_step % (50 * self.frame_skip) == 0:
            print(f"\n📡 --- [RADAR IoT GREENVEIN - {time_str} | {traffic_status}] ---")
            for t in self.truck_ids:
                if not self.is_done.get(t, True):
                    red = sum(1 for b in self.zone_bins[t] if self.bin_levels[b] >= 70.0)
                    yellow = sum(1 for b in self.zone_bins[t] if 40.0 <= self.bin_levels[b] < 70.0)
                    fuel = self.current_fuel[t]
                    
                    if fuel < 20.0:
                        print(f"   ⚠️ {t}: XĂNG SẮP HẾT ({fuel:.1f}%). Đang tìm đường về Trạm!")
                    elif red > 0:
                        print(f"   🚨 {t}: Có {red} thùng ĐỎ! Đang phi tốc hành tới thu dọn. (Xăng: {fuel:.1f}%)")
                    elif yellow > 0:
                        print(f"   🟡 {t}: {yellow} thùng VÀNG. Đang đi vét phòng ngừa. (Xăng: {fuel:.1f}%)")
                    else:
                        print(f"   🟢 {t}: Cụm an toàn. Chờ lệnh tại Trạm. (Xăng: {fuel:.1f}%)")
            print("-" * 70)

        active_vehicles = traci.vehicle.getIDList()
        
        for truck_id in self.truck_ids:
            if truck_id in active_vehicles and truck_id in action_dict:
                act = action_dict[truck_id]
                try:
                    if act == 0: traci.vehicle.setSpeed(truck_id, 3.0)      
                    elif act == 1: traci.vehicle.setSpeed(truck_id, -1.0)   
                    elif act == 2: traci.vehicle.setSpeed(truck_id, 15.0)   
                except: pass

        current_sumo_time = traci.simulation.getTime()
        traci.simulationStep(current_sumo_time + self.frame_skip)
        
        self.current_step += self.frame_skip 
        
        for b in self.bin_levels.keys():
            growth_per_sec = self.generators[b].get_fill_rate(self.current_step)
            total_growth = growth_per_sec * self.frame_skip
            self.bin_levels[b] = min(100.0, self.bin_levels[b] + total_growth)

        active_vehicles = traci.vehicle.getIDList()
        next_states = {}
        rewards = {}
        terminated = {t: False for t in self.truck_ids}
        truncated = {t: False for t in self.truck_ids}

        # 🌟 NÂNG CẤP 3: GLOBAL REWARD (Thưởng tập thể)
        global_collected_this_step = 0.0

        for truck_id in self.truck_ids:
            if truck_id not in active_vehicles:
                if self.aborted.get(truck_id, False):
                    rewards[truck_id] = 0.0
                    next_states[truck_id] = np.zeros(6, dtype=np.float32) # State 6
                    terminated[truck_id] = True
                elif not self.is_done[truck_id]:
                    self.is_done[truck_id] = True
                    if self.is_heading_depot.get(truck_id, False):
                        if self.total_collected.get(truck_id, 0.0) > 0:
                            rewards[truck_id] = 200.0  
                        else:
                            rewards[truck_id] = -20.0  
                    else:
                        rewards[truck_id] = -50.0 
                        print(f"⏰ [{time_str} | {traffic_status}] 🚨 [{truck_id}] Gặp sự cố chết máy dọc đường! (-50đ)")
                    
                    next_states[truck_id] = np.zeros(6, dtype=np.float32) # State 6
                    terminated[truck_id] = True 
                else:
                    rewards[truck_id] = 0.0
                    next_states[truck_id] = np.zeros(6, dtype=np.float32) # State 6
                    terminated[truck_id] = True

            else:
                current_edge = traci.vehicle.getRoadID(truck_id)
                current_speed_kmh = traci.vehicle.getSpeed(truck_id) * 3.6 
                
                # 🌟 NÂNG CẤP TỤT XĂNG (FUEL CONSUMPTION)
                if current_speed_kmh > 1.0:
                    self.current_fuel[truck_id] -= (0.015 * self.frame_skip) # Đang chạy tốn xăng
                else:
                    self.current_fuel[truck_id] -= (0.005 * self.frame_skip) # Đứng im kẹt xe vẫn tốn nhẹ
                
                # 🌟 NÂNG CẤP PHẠT GIỜ CẤM TẢI
                if is_rush_hour and current_edge != self.depot_edge:
                    step_rewards[truck_id] -= 1.0 # Trừ điểm liên tục nếu ngoan cố chạy ra đường giờ tắc
                
                try:
                    street_name = traci.edge.getStreetName(current_edge)
                    display_loc = f"phố {street_name}" if street_name else f"ngõ {current_edge}"
                except:
                    display_loc = f"ngõ {current_edge}"

                if current_edge == self.depot_edge:
                    # 🌟 FIX LOGIC XĂNG VÀ XẢ RÁC TÁCH BIỆT
                    is_fuel_refilled = False
                    is_trash_dumped = False
                    
                    # 1. Hành vi đổ xăng (Chỉ đổ khi < 30%)
                    if self.current_fuel[truck_id] < 30.0:
                        self.current_fuel[truck_id] = self.MAX_FUEL
                        is_fuel_refilled = True
                        
                    # 2. Hành vi xả rác
                    if self.current_load[truck_id] > 0 and self.last_visited_bin[truck_id] != current_edge:
                        dumped_amount = self.current_load[truck_id]
                        self.current_load[truck_id] = 0.0
                        self.is_heading_depot[truck_id] = False
                        self.last_visited_bin[truck_id] = current_edge
                        is_trash_dumped = True
                        
                        step_rewards[truck_id] += 150.0 
                        success = self.assign_urgent_target(truck_id)
                        if not success: self.stuck_time[truck_id] = 9999
                        
                    # 3. Thông báo gộp thông minh
                    if is_fuel_refilled and is_trash_dumped:
                        print(f"⏰ [{time_str} | {traffic_status}] ♻️⛽ [{truck_id}] Xả {dumped_amount:.1f} kg rác VÀ đổ đầy bình xăng. Sẵn sàng đi tiếp!")
                    elif is_trash_dumped:
                        print(f"⏰ [{time_str} | {traffic_status}] ♻️ [{truck_id}] Xả {dumped_amount:.1f} kg rác. Bụng rỗng, sẵn sàng nhận lệnh!")
                    elif is_fuel_refilled:
                        print(f"⏰ [{time_str} | {traffic_status}] ⛽ [{truck_id}] Ghé trạm đổ đầy bình xăng.")

                elif current_edge in self.zone_bins[truck_id] or current_edge in self.bin_levels:
                    if self.last_visited_bin[truck_id] != current_edge:
                        self.last_visited_bin[truck_id] = current_edge
                        bin_level = self.bin_levels[current_edge]
                        
                        if bin_level >= 30.0 and self.current_load[truck_id] < self.MAX_CAPACITY_KG:
                            kg_in_bin = (bin_level / 100.0) * self.BIN_MAX_WEIGHT_KG
                            available_space = self.MAX_CAPACITY_KG - self.current_load[truck_id]
                            
                            amount_collected = min(kg_in_bin, available_space)
                            self.current_load[truck_id] += amount_collected
                            self.total_collected[truck_id] += amount_collected 
                            global_collected_this_step += amount_collected # Góp vào quỹ thưởng chung
                            
                            remaining_kg = kg_in_bin - amount_collected
                            self.bin_levels[current_edge] = (remaining_kg / self.BIN_MAX_WEIGHT_KG) * 100.0
                            
                            if bin_level >= 70.0:
                                print(f"⏰ [{time_str} | {traffic_status}] 🚨 [{truck_id}] Xử lý điểm ĐỎ tại {display_loc}. Thu {amount_collected:.1f} kg.")
                            else:
                                print(f"⏰ [{time_str} | {traffic_status}] 🟡 [{truck_id}] Tiện đường hốt thùng Vàng tại {display_loc} ({amount_collected:.1f} kg).")
                            
                            step_rewards[truck_id] += (amount_collected / 2.0) 

                        if self.current_load[truck_id] < self.MAX_CAPACITY_KG:
                            success = self.assign_urgent_target(truck_id)
                            if not success: self.stuck_time[truck_id] = 9999

                load_percent = (self.current_load[truck_id] / self.MAX_CAPACITY_KG) * 100.0
                
                # 🌟 ÉP VỀ TRẠM KHI ĐẦY RÁC **HOẶC SẮP HẾT XĂNG (<15%)**
                if (load_percent >= 95.0 or self.current_fuel[truck_id] < 15.0) and not self.is_heading_depot[truck_id]:
                    try:
                        traci.vehicle.changeTarget(truck_id, self.depot_edge)
                        self.is_heading_depot[truck_id] = True
                    except: pass

                co2_emission_g_s = traci.vehicle.getCO2Emission(truck_id) / 1000.0
                
                self.trip_co2[truck_id] += (co2_emission_g_s * self.frame_skip)
                current_distance_km = traci.vehicle.getDistance(truck_id) / 1000.0
                self.trip_distance[truck_id] = current_distance_km
                avg_co2_per_km = (self.trip_co2[truck_id] / current_distance_km) if current_distance_km > 0.001 else 0.0

                if current_speed_kmh < 1.0: self.stuck_time[truck_id] += 1
                else: self.stuck_time[truck_id] = 0

                reward = step_rewards[truck_id]
                overflow_count = sum(1 for b in self.zone_bins[truck_id] if self.bin_levels[b] >= 99.0)
                if overflow_count > 0: reward -= (overflow_count * 0.5)

                # 🌟 STATE MỚI CHUẨN 6 CHIỀU (Thêm Fuel ở cuối)
                next_states[truck_id] = np.array([
                    current_speed_kmh, avg_co2_per_km, float(self.stuck_time[truck_id]), 
                    current_distance_km, load_percent, float(self.current_fuel[truck_id])
                ], dtype=np.float32)
                
                if current_speed_kmh > 15.0: reward += 3.0  
                elif current_speed_kmh < 1.0: reward -= (1.0 + self.stuck_time[truck_id] * 0.5)
                reward -= (avg_co2_per_km / 500.0)

                # 🌟 CHẾT VÌ KẸT CỨNG HOẶC HẾT SẠCH XĂNG
                if self.stuck_time[truck_id] > 30 or avg_co2_per_km > 3000.0 or self.stuck_time[truck_id] == 9999 or self.current_fuel[truck_id] <= 0:
                    reward -= 100.0
                    reason = "Hết xăng giữa đường" if self.current_fuel[truck_id] <= 0 else "Kẹt cứng quá lâu"
                    print(f"⏰ [{time_str} | {traffic_status}] 💀 [{truck_id}] Báo tử do: {reason}! Đội cứu hộ đang tới...")
                    try: traci.vehicle.remove(truck_id) 
                    except: pass

                rewards[truck_id] = reward

        # 🌟 CỘNG ĐIỂM GLOBAL (TINH THẦN ĐỒNG ĐỘI)
        if global_collected_this_step > 0:
            for t in self.truck_ids:
                if not terminated[t]:
                    # Mỗi xe được chia một chút điểm nhỏ từ quỹ rác chung toàn bản đồ
                    rewards[t] += (global_collected_this_step * 0.05) 

        return next_states, rewards, terminated, truncated, {}

    def close(self):
        try: traci.close()
        except: pass