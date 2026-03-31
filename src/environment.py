import os
import sys
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci

class GreenVeinEnv(gym.Env):
    def __init__(self):
        super(GreenVeinEnv, self).__init__()
        
        self.sumo_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'envs', 'greenvein.sumocfg'))
        self.sumo_cmd = ["sumo", "-c", self.sumo_cfg, "--no-warnings", "--time-to-teleport", "-1"]
        
        self.truck_ids = ["XeRac_AI_1", "XeRac_AI_2", "XeRac_AI_3"]
        self.depot_edge = "684766065#0"
        
        self.depot_names = {"684766065#0": "Trạm Điều hành GreenVein"}
        
        self.zone_names = {
            "XeRac_AI_1": "Cụm Tây (Láng Thượng, Láng Hạ, Ngã Tư Sở, Thịnh Quang)",
            "XeRac_AI_2": "Cụm Trung Tâm (Trung Liệt, Quang Trung, Ô Chợ Dừa, Nam Đồng)",
            "XeRac_AI_3": "Cụm Đông (Kim Liên, Phương Mai, Khâm Thiên, Trung Phụng)"
        }

        self.frame_skip = 50 
        self.action_space = spaces.Discrete(3)
        # Không gian trạng thái 4 chiều: [Tốc độ, Phát thải TB, Thời gian kẹt, Quãng đường]
        self.observation_space = spaces.Box(low=0.0, high=10000.0, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except:
            pass
            
        traci.start(self.sumo_cmd)
        
        all_edges = traci.edge.getIDList()
        valid_edges = []
        for edge_id in all_edges:
            if edge_id.startswith(":"): continue
            try:
                allowed = traci.lane.getAllowed(edge_id + "_0")
                disallowed = traci.lane.getDisallowed(edge_id + "_0")
                if len(allowed) == 0:
                    if "truck" not in disallowed: valid_edges.append(edge_id)
                elif "truck" in allowed:
                    valid_edges.append(edge_id)
            except:
                continue

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

        traci.route.add("depot_route", [self.depot_edge])
        
        for truck_id in self.truck_ids:
            traci.vehicle.add(truck_id, "depot_route")
            traci.vehicle.setShapeClass(truck_id, "truck")
            
            if truck_id == "XeRac_AI_1": traci.vehicle.setColor(truck_id, (255, 69, 0))   
            elif truck_id == "XeRac_AI_2": traci.vehicle.setColor(truck_id, (30, 144, 255)) 
            else: traci.vehicle.setColor(truck_id, (50, 205, 50))                        
            
            if target_zones[truck_id]:
                for _ in range(10): 
                    random_dest = random.choice(target_zones[truck_id])
                    try:
                        traci.vehicle.changeTarget(truck_id, random_dest)
                        break 
                    except:
                        pass 

        traci.simulationStep()

        self.trip_co2 = {truck_id: 0.0 for truck_id in self.truck_ids}
        self.trip_distance = {truck_id: 0.0 for truck_id in self.truck_ids}
        
        self.is_done = {truck_id: False for truck_id in self.truck_ids} 
        self.aborted = {truck_id: False for truck_id in self.truck_ids} 
        self.stuck_time = {truck_id: 0 for truck_id in self.truck_ids}

        # 🌟 TỐI ƯU STATE: Khởi tạo mảng [0.0, 0.0, 0.0, 0.0]
        initial_states = {truck_id: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32) for truck_id in self.truck_ids}
        
        print("\n" + "=".center(60, "="))
        print(" PHÂN CÔNG KHU VỰC THU GOM RÁC ".center(60, "="))
        print("=".center(60, "="))
        for truck_id in self.truck_ids:
            print(f"📍 [{truck_id}] Xuất phát: {self.depot_names.get(self.depot_edge, 'Trạm Điều hành')}")
            print(f"   🗺️  Khu vực: {self.zone_names[truck_id]}")
            print("-" * 60)

        return initial_states, {}

    def step(self, action_dict):
        for _ in range(self.frame_skip): 
            traci.simulationStep()

        next_states = {}
        rewards = {}
        terminated = {}
        truncated = {truck_id: False for truck_id in self.truck_ids}
        active_vehicles = traci.vehicle.getIDList()

        for truck_id in self.truck_ids:
            if truck_id in active_vehicles:
                current_speed_kmh = traci.vehicle.getSpeed(truck_id) * 3.6 
                co2_emission_g_s = traci.vehicle.getCO2Emission(truck_id) / 1000.0
                
                self.trip_co2[truck_id] += (co2_emission_g_s * self.frame_skip)
                current_distance_km = traci.vehicle.getDistance(truck_id) / 1000.0
                self.trip_distance[truck_id] = current_distance_km
                
                avg_co2_per_km = (self.trip_co2[truck_id] / current_distance_km) if current_distance_km > 0.001 else 0.0

                if current_speed_kmh < 1.0:
                    self.stuck_time[truck_id] += 1
                else:
                    self.stuck_time[truck_id] = 0

                # 🌟 TỐI ƯU STATE: Cập nhật Giác quan mới
                next_states[truck_id] = np.array([
                    current_speed_kmh, 
                    avg_co2_per_km, 
                    float(self.stuck_time[truck_id]), 
                    current_distance_km
                ], dtype=np.float32)
                
                # 🌟 TỐI ƯU REWARD: Hàm phần thưởng/Trừng phạt lũy tiến
                reward = 0.0
                if current_speed_kmh > 18.0: 
                    reward += 2.0  
                elif current_speed_kmh < 1.0: 
                    reward -= (1.0 + self.stuck_time[truck_id] * 0.5)
                reward -= (avg_co2_per_km / 500.0)

                if self.stuck_time[truck_id] > 30 or avg_co2_per_km > 3000.0:
                    reward -= 100.0
                    terminated[truck_id] = True
                    self.aborted[truck_id] = True
                    reason = "Đứng im > 30 bước (Kẹt cứng)" if self.stuck_time[truck_id] > 30 else "Phát thải > 3000g"
                    print(f"💀 [{truck_id}] ĐÃ BỎ CUỘC VÌ KẸT XE! ({reason}). Gọi cứu hộ cẩu khỏi map!")
                    try:
                        traci.vehicle.remove(truck_id) 
                    except:
                        pass
                else:
                    terminated[truck_id] = False

                rewards[truck_id] = reward

            else:
                if self.aborted.get(truck_id, False):
                    rewards[truck_id] = 0.0
                    next_states[truck_id] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    terminated[truck_id] = True
                elif not self.is_done[truck_id]:
                    final_dist = self.trip_distance.get(truck_id, 0.0)
                    final_co2_km = (self.trip_co2[truck_id] / final_dist) if final_dist > 0.001 else 0.0
                    print(f"🏁 [{truck_id}] ĐÃ HOÀN THÀNH TỐT ĐẸP! Quãng đường: {final_dist:.2f} km | CO2: {final_co2_km:.1f} g/km")
                    self.is_done[truck_id] = True
                    rewards[truck_id] = 100.0
                    next_states[truck_id] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    terminated[truck_id] = True
                else:
                    rewards[truck_id] = 0.0
                    next_states[truck_id] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    terminated[truck_id] = True

        return next_states, rewards, terminated, truncated, {}

    def close(self):
        try:
            traci.close()
        except:
            pass