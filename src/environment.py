import os
import sys
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import libsumo as traci
    print("🚀 [HỆ THỐNG] Đã nạp thành công LIBSUMO - Chế độ Siêu Tốc được kích hoạt!")
except ImportError:
    import traci
    print("🖥️ [HỆ THỐNG] Không tìm thấy libsumo, dùng TraCI (chạy chậm)...")

from waste_generator import RealWasteGenerator

class GreenVeinEnv(gym.Env):
    def __init__(self):
        super(GreenVeinEnv, self).__init__()
        
        self.sumo_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'envs', 'greenvein.sumocfg'))
        self.sumo_cmd = [
            "sumo", "-c", self.sumo_cfg, 
            "--no-warnings", "--time-to-teleport", "-1", 
            "--error-log", "sumo_error.log", "--no-step-log", "--mesosim", "true"
        ]
        
        self.truck_ids = ["XeRac_AI_1", "XeRac_AI_2", "XeRac_AI_3"]
        self.zone_names = {"XeRac_AI_1": "Cụm Tây", "XeRac_AI_2": "Cụm Trung Tâm", "XeRac_AI_3": "Cụm Đông"}
        self.color_map = {"XeRac_AI_1": (255, 100, 0), "XeRac_AI_2": (50, 150, 255), "XeRac_AI_3": (0, 255, 0)}

        # =====================================================================
        # 🌟 BẢNG QUY HOẠCH THEO TUYẾN - ĐÃ CHUẨN HÓA
        # =====================================================================
        self.CONFIG_QUY_HOACH = {
            "depots": {
                "XeRac_AI_1": "946030657",
                "XeRac_AI_2": "946030657",
                "XeRac_AI_3": "946030657"
            },
            "zones": {
                "XeRac_AI_1": [
                    "707072725#2", "707066366#7", "707066366#11", "709017803#1", "179998311#2", 
                    "-180001033#9", "-198407217#3", "136524198#2", "1215063383", "707066366#9", 
                    "1012665674", "-219978979#1", "1208997907", "708576350#0", "178091734#1", 
                    "-1262082048", "-179995750#3", "-477417897#1"
                ],
                "XeRac_AI_2": [
                    "180082698#1", "1215943717#0", "1215943717#2", "711031662#4", "-180082702#0", 
                    "-601455486#1", "29313248#0", "1420319339", "25953535#0", "890573930#0", 
                    "-459315213#1", "597126919#1", "597113041", "-28958235#2", "-597111783#3", 
                    "601535720#1", "218427624#1", "1034359440#1"
                ],
                "XeRac_AI_3": [
                    "-675484248#3", "180001031#1", "-707366491#1", "1412423844#0", "38028986#4", 
                    "707087632#5", "-148202928#3", "-219863682", "-1461754606#4", "-11838452#1", 
                    "196054187#0", "194581852#1", "560585021#3", "-835632103#0", "1155941851#5", 
                    "-890573859", "-180082714#1", "-1276658079#1"
                ]
            }
        }

        self.depot_edges = {} 
        self.frame_skip = 10 
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=10000.0, shape=(6,), dtype=np.float32)
        
        self.MAX_CAPACITY_KG = 5000.0 
        self.BIN_MAX_WEIGHT_KG = 200.0 
        self.MAX_FUEL = 100.0 
        
        self.edge_centers = {}
        self.valid_edges_list = [] 
        self.valid_passenger_edges = [] 
        self.valid_motorcycle_edges = []
        
        self.route_cache_car = []
        self.route_cache_moto = []
        self.blacklist = {t: {} for t in self.truck_ids}
        self.street_map = {}
        self.hanoi_streets = ["Tôn Đức Thắng", "Tây Sơn", "Chùa Bộc", "Thái Hà", "Thái Thịnh", "Đường Láng", "Nguyễn Trãi", "Xã Đàn", "Khâm Thiên", "Đê La Thành"]

    def get_real_street_name(self, edge_id):
        try:
            name = traci.edge.getStreetName(edge_id)
            if name and not name.replace('-', '').replace('#', '').isdigit(): return f"phố {name}"
        except: pass
        if edge_id not in self.street_map:
            random.seed(hash(edge_id)) 
            self.street_map[edge_id] = f"ngõ {random.randint(1, 200)} {random.choice(self.hanoi_streets)}"
            random.seed() 
        return self.street_map[edge_id]

    def _route_to_target(self, truck_id, current_edge, target_edge, depot_edge, is_depot=False):
        if is_depot:
            try:
                r1 = traci.simulation.findRoute(current_edge, depot_edge, vType="garbage_truck")
                if r1 and len(r1.edges) > 0:
                    traci.vehicle.setRoute(truck_id, r1.edges)
                    self.is_heading_depot[truck_id] = True
                    return True
            except: pass
            return False
            
        try:
            r1 = traci.simulation.findRoute(current_edge, target_edge, vType="garbage_truck")
            r2 = traci.simulation.findRoute(target_edge, depot_edge, vType="garbage_truck")
            if r1 and r2 and len(r1.edges) > 0 and len(r2.edges) > 0:
                full_route = list(r1.edges) + list(r2.edges)[1:]
                traci.vehicle.setRoute(truck_id, full_route)
                self.is_heading_depot[truck_id] = False
                return True
        except: pass
        return False

    def assign_urgent_target(self, truck_id):
        depot = self.depot_edges.get(truck_id, "684766065#0")
        try: current_edge = traci.vehicle.getRoadID(truck_id)
        except: current_edge = depot
        if not current_edge or current_edge.startswith(":"): current_edge = depot
        
        try: traci.vehicle.resume(truck_id)
        except: pass

        # Xóa án phạt Sổ Đen rất nhanh để xe quay lại ăn rác vàng
        for b in list(self.blacklist[truck_id].keys()):
            self.blacklist[truck_id][b] -= 10
            if self.blacklist[truck_id][b] <= 0: del self.blacklist[truck_id][b]

        if (self.current_load[truck_id] / self.MAX_CAPACITY_KG) >= 0.95:
            if current_edge == depot: 
                self.is_heading_depot[truck_id] = True
                return True
            if self._route_to_target(truck_id, current_edge, depot, depot, is_depot=True): return True
            return False

        curr_x, curr_y = self.edge_centers.get(current_edge, (0.0, 0.0))
        candidates = []
        AVERAGE_SPEED_MPS = 5.0 
        
        # 1. Quét rác Phường mình (Chỉ cần > 5.0% là đưa vào tầm ngắm)
        for b in self.zone_bins[truck_id]:
            if b == current_edge or b in self.blacklist[truck_id]: continue
            
            dist = np.hypot(curr_x - self.edge_centers.get(b, (0.0, 0.0))[0], curr_y - self.edge_centers.get(b, (0.0, 0.0))[1])
            estimated_time = dist / AVERAGE_SPEED_MPS
            
            current_lvl = self.bin_levels.get(b, 0)
            fill_rate = self.generators[b].get_fill_rate(self.current_step)
            predicted_lvl = current_lvl + (fill_rate * estimated_time)
            
            if predicted_lvl > 5.0: # Hạ chuẩn để không bỏ sót thùng Vàng/Xanh nhạt
                score = predicted_lvl / (dist + 50.0) 
                candidates.append((b, score, predicted_lvl))
        
        # 2. Cứu trợ Hàng xóm (Nếu phường mình thật sự sạch bách)
        if not candidates:
            for b, lvl in self.bin_levels.items():
                if b in self.zone_bins[truck_id] or b == current_edge or b in self.blacklist[truck_id]: continue
                dist = np.hypot(curr_x - self.edge_centers.get(b, (0.0, 0.0))[0], curr_y - self.edge_centers.get(b, (0.0, 0.0))[1])
                estimated_time = dist / AVERAGE_SPEED_MPS
                predicted_lvl = lvl + (self.generators[b].get_fill_rate(self.current_step) * estimated_time)
                
                if predicted_lvl > 15.0: # Hạ chuẩn cứu trợ
                    score = predicted_lvl / (dist + 200.0) 
                    candidates.append((b, score, predicted_lvl))

        if not candidates:
            if current_edge == depot:
                self.is_heading_depot[truck_id] = True
                try: traci.vehicle.setSpeed(truck_id, 0.0)
                except: pass
                return True 
            if self._route_to_target(truck_id, current_edge, depot, depot, is_depot=True): return True
            return False
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        for best_bin, score, p_lvl in candidates[:15]: # Quét rộng ra 15 ứng viên
            if self._route_to_target(truck_id, current_edge, best_bin, depot, is_depot=False):
                try: traci.vehicle.setSpeed(truck_id, -1) 
                except: pass
                return True
            else:
                # Phạt nhẹ để tí nữa xe chạy ra góc khác có thể tìm được đường vào
                self.blacklist[truck_id][best_bin] = 30 
        
        if current_edge == depot: return True 
        if self._route_to_target(truck_id, current_edge, depot, depot, is_depot=True): return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try: traci.close()
        except: pass
            
        traci.start(self.sumo_cmd)
        self.virtual_time_seconds = 20 * 3600  
        self.current_hour_int = 20
        try: traci.simulation.setScale(0.1)
        except: pass

        self.current_step = 0
        all_edges = traci.edge.getIDList()
        self.valid_edges_list = []
        self.valid_passenger_edges = []
        self.valid_motorcycle_edges = []
        self.street_map.clear() 
        self.blacklist = {t: {} for t in self.truck_ids}
        
        vehicle_types = [
            ("garbage_truck", "delivery", "truck", (255,255,255), 10.0, 3.5, 15.0),
            ("passenger_car", "passenger", "passenger", (210,210,210), 4.5, 1.8, 12.0),
            ("motorcycle_lead", "motorcycle", "motorcycle", (255,105,180), 2.0, 0.8, 16.0),
            ("motorcycle_norm", "motorcycle", "motorcycle", (80,80,80), 2.0, 0.8, 14.0)
        ]
        existing_types = traci.vehicletype.getIDList()
        for v_id, v_class, v_shape, color, length, width, speed in vehicle_types:
            if v_id not in existing_types:
                traci.vehicletype.copy("DEFAULT_VEHTYPE", v_id)
                traci.vehicletype.setVehicleClass(v_id, v_class)
                traci.vehicletype.setShapeClass(v_id, v_shape)
                traci.vehicletype.setColor(v_id, color)
                traci.vehicletype.setLength(v_id, length)
                traci.vehicletype.setWidth(v_id, width)
                traci.vehicletype.setMaxSpeed(v_id, speed)
        
        for edge_id in all_edges:
            if edge_id.startswith(":"): continue
            try:
                allowed = traci.lane.getAllowed(edge_id + "_0")
                disallowed = traci.lane.getDisallowed(edge_id + "_0")
                if len(allowed) == 0:
                    if "delivery" not in disallowed: self.valid_edges_list.append(edge_id)
                    if "passenger" not in disallowed: self.valid_passenger_edges.append(edge_id)
                    if "motorcycle" not in disallowed: self.valid_motorcycle_edges.append(edge_id)
                else:
                    if "delivery" in allowed: self.valid_edges_list.append(edge_id)
                    if "passenger" in allowed: self.valid_passenger_edges.append(edge_id)
                    if "motorcycle" in allowed: self.valid_motorcycle_edges.append(edge_id)
            except: continue
        
        self.edge_centers = {}
        for edge_id in self.valid_edges_list:
            try:
                shape = traci.lane.getShape(edge_id + "_0") 
                self.edge_centers[edge_id] = (sum([p[0] for p in shape])/len(shape), sum([p[1] for p in shape])/len(shape))
            except: self.edge_centers[edge_id] = (0.0, 0.0)

        self.bin_levels = {}
        self.generators = {}
        self.zone_bins = {truck_id: [] for truck_id in self.truck_ids}
        self.depot_edges = {}

        for truck_id in self.truck_ids:
            user_depot = self.CONFIG_QUY_HOACH["depots"].get(truck_id, "")
            if user_depot:
                self.depot_edges[truck_id] = user_depot
                if user_depot not in self.valid_edges_list: self.valid_edges_list.append(user_depot)
            else:
                self.depot_edges[truck_id] = random.choice(self.valid_edges_list)

        for truck_id in self.truck_ids:
            for b in self.CONFIG_QUY_HOACH["zones"].get(truck_id, []):
                self.zone_bins[truck_id].append(b)
                if b not in self.valid_edges_list: self.valid_edges_list.append(b)

        for edge_id in self.valid_edges_list:
            if edge_id not in self.edge_centers:
                try:
                    shape = traci.lane.getShape(edge_id + "_0") 
                    self.edge_centers[edge_id] = (sum([p[0] for p in shape])/len(shape), sum([p[1] for p in shape])/len(shape))
                except: self.edge_centers[edge_id] = (0.0, 0.0)

        for truck_id in self.truck_ids:
            for b in self.zone_bins[truck_id]:
                self.bin_levels[b] = random.uniform(10.0, 60.0) 
                self.generators[b] = RealWasteGenerator(zone_type="commercial" if truck_id == "XeRac_AI_2" else "residential")

        print("⏳ Đang thiết lập ma trận kẹt xe (Pre-computing routes)... Xin chờ 2 giây...")
        self.route_cache_car = []
        self.route_cache_moto = []
        for _ in range(1000):
            if len(self.route_cache_car) >= 300: break
            s = random.choice(self.valid_passenger_edges)
            e = random.choice(self.valid_passenger_edges)
            try:
                r = traci.simulation.findRoute(s, e, vType="passenger_car")
                if r and len(r.edges) >= 4: self.route_cache_car.append(r.edges)
            except: pass
            
        for _ in range(1000):
            if len(self.route_cache_moto) >= 300: break
            s = random.choice(self.valid_motorcycle_edges)
            e = random.choice(self.valid_motorcycle_edges)
            try:
                r = traci.simulation.findRoute(s, e, vType="motorcycle_lead")
                if r and len(r.edges) >= 4: self.route_cache_moto.append(r.edges)
            except: pass

        print("🚗 Đang rải 400 xe dân sự phủ kín mặt đường...")
        for i in range(400):
            v_type = random.choice(["passenger_car", "passenger_car", "motorcycle_lead", "motorcycle_norm"])
            cache = self.route_cache_moto if "motorcycle" in v_type else self.route_cache_car
            if cache:
                vid = f"xe_dan_init_{i}_{random.randint(100, 9999)}"
                try:
                    traci.route.add(f"route_{vid}", random.choice(cache))
                    traci.vehicle.add(vid, f"route_{vid}", typeID=v_type, departPos="random")
                except: pass

        for b in self.bin_levels.keys():
            tx, ty = self.edge_centers.get(b, (0.0, 0.0))
            try: traci.poi.add(f"BIN_{b}", tx, ty, (0, 255, 0, 255), poiType="Thùng Rác", layer=100, width=15.0, height=15.0)
            except: pass

        # Đã XÓA MÃ TẠO ĐỒNG HỒ Ở ĐÂY THEO YÊU CẦU

        self.current_load = {t: 0.0 for t in self.truck_ids}
        self.current_fuel = {t: self.MAX_FUEL for t in self.truck_ids} 
        self.is_heading_depot = {t: False for t in self.truck_ids}
        self.is_done = {t: False for t in self.truck_ids} 
        self.stuck_time = {t: 0 for t in self.truck_ids}
        self.trip_co2 = {t: 0.0 for t in self.truck_ids}
        self.trip_distance = {t: 0.0 for t in self.truck_ids}
        self.total_collected = {t: 0.0 for t in self.truck_ids} 
        self.last_visited_bin = {t: "" for t in self.truck_ids} 
        self.has_departed = {t: False for t in self.truck_ids}

        for truck_id in self.truck_ids:
            try:
                traci.route.add(f"route_{truck_id}_init", [self.depot_edges[truck_id]])
                traci.vehicle.add(truck_id, f"route_{truck_id}_init", typeID="garbage_truck")
                traci.vehicle.setColor(truck_id, self.color_map[truck_id])
            except: pass                       

        traci.simulationStep()
        for truck_id in self.truck_ids:
            self.assign_urgent_target(truck_id)
        
        return {truck_id: np.zeros(6, dtype=np.float32) for truck_id in self.truck_ids}, {}

    def step(self, action_dict):
        next_states = {t: np.zeros(6, dtype=np.float32) for t in self.truck_ids}
        rewards = {t: 0.0 for t in self.truck_ids}
        step_rewards = {t: 0.0 for t in self.truck_ids}
        terminated = {t: False for t in self.truck_ids}

        self.virtual_time_seconds = (self.virtual_time_seconds + self.frame_skip) % 86400
        hour = self.virtual_time_seconds / 3600.0
        time_str = f"{int(hour):02d}:{int((hour % 1) * 60):02d}"
        is_night_shift = (hour >= 20.0) or (hour < 5.0)

        # Đã XÓA MÃ CẬP NHẬT ĐỒNG HỒ Ở ĐÂY

        active_vehicles = traci.vehicle.getIDList()
        
        for truck_id in self.truck_ids:
            if truck_id in active_vehicles:
                try:
                    x, y = traci.vehicle.getPosition(truck_id)
                    poi_id = f"TRACKER_{truck_id}"
                    if poi_id in traci.poi.getIDList():
                        traci.poi.setPosition(poi_id, x, y)
                    else:
                        traci.poi.add(poi_id, x, y, self.color_map[truck_id], poiType=f"🚛 {self.zone_names[truck_id]}", layer=300, width=25.0, height=25.0)
                except: pass

        bg_cars = [v for v in active_vehicles if v.startswith("xe_dan_")]
        if len(bg_cars) < 400 and self.current_step % (5 * self.frame_skip) == 0:
            for i in range(20): 
                v_type = random.choice(["passenger_car", "passenger_car", "motorcycle_lead", "motorcycle_norm"])
                cache = self.route_cache_moto if "motorcycle" in v_type else self.route_cache_car
                if cache:
                    vid = f"xe_dan_{self.current_step}_{i}_{random.randint(100, 9999)}"
                    try:
                        traci.route.add(f"route_{vid}", random.choice(cache))
                        traci.vehicle.add(vid, f"route_{vid}", typeID=v_type)
                    except: pass

        for t in self.truck_ids:
            if t in active_vehicles: self.has_departed[t] = True
        
        total_trash_on_map = sum(self.bin_levels.values())
        
        for truck_id in self.truck_ids:
            depot = self.depot_edges.get(truck_id, "684766065#0")
            if truck_id not in active_vehicles:
                if not self.has_departed[truck_id]: continue 
                
                if not self.is_done[truck_id]:
                    if total_trash_on_map > 50.0:
                        # Chỉ in log cẩu xe khi bị lỗi thật sự mất tích khỏi bản đồ
                        try:
                            route_id = f"route_respawn_{truck_id}_{self.current_step}"
                            traci.route.add(route_id, [depot])
                            traci.vehicle.add(truck_id, route_id, typeID="garbage_truck")
                            traci.vehicle.setColor(truck_id, self.color_map[truck_id])
                            self.stuck_time[truck_id] = 0
                            self.assign_urgent_target(truck_id)
                        except: pass
                        rewards[truck_id] = -20.0 # Giảm nhẹ điểm trừ
                        terminated[truck_id] = False
                    else:
                        self.is_done[truck_id] = True
                        print(f"⏰ [{time_str}] 🏁 [{truck_id}] HOÀN THÀNH NHIỆM VỤ! RÁC ĐÃ SẠCH.")
                        rewards[truck_id] = 200.0  
                        terminated[truck_id] = True 
            else:
                act = action_dict.get(truck_id, 1)
                try: edge = traci.vehicle.getRoadID(truck_id)
                except: edge = ""
                
                if not is_night_shift: act = 1 if (edge == depot or edge == "") else 2

                if edge == depot:
                    act = 0

                try:
                    if act == 0: traci.vehicle.setSpeed(truck_id, 3.0)      
                    elif act == 1: traci.vehicle.setSpeed(truck_id, 0.0)   
                    elif act == 2: traci.vehicle.setSpeed(truck_id, 15.0)   
                except: pass

                current_speed_kmh = traci.vehicle.getSpeed(truck_id) * 3.6 
                fuel_consumed = (0.015 if current_speed_kmh > 1.0 else 0.005) * self.frame_skip
                
                self.current_fuel[truck_id] -= fuel_consumed
                step_rewards[truck_id] -= (fuel_consumed * 2.0) 
                
                if not is_night_shift and edge != depot:
                    step_rewards[truck_id] -= 5.0 

                if edge == depot:
                    if self.current_fuel[truck_id] < 30.0: 
                        print(f"⛽ [{time_str}] [{truck_id}] Đang nạp đầy nhiên liệu tại Trạm...")
                        self.current_fuel[truck_id] = self.MAX_FUEL
                        
                    if self.current_load[truck_id] > 0 and self.last_visited_bin[truck_id] != edge:
                        dumped_amount = self.current_load[truck_id]
                        self.current_load[truck_id] = 0.0
                        self.last_visited_bin[truck_id] = edge
                        step_rewards[truck_id] += dumped_amount * 0.1 
                        print(f"⏰ [{time_str}] ♻️ [{truck_id}] Đổ {dumped_amount:.1f}kg rác tại Trạm.")
                        self.assign_urgent_target(truck_id)

                elif edge in self.bin_levels:
                    if self.last_visited_bin[truck_id] != edge:
                        self.last_visited_bin[truck_id] = edge
                        bin_level = self.bin_levels[edge]
                        
                        if bin_level >= 5.0 and self.current_load[truck_id] < self.MAX_CAPACITY_KG:
                            kg_in_bin = (bin_level / 100.0) * self.BIN_MAX_WEIGHT_KG
                            amount_collected = min(kg_in_bin, self.MAX_CAPACITY_KG - self.current_load[truck_id])
                            
                            self.current_load[truck_id] += amount_collected
                            self.total_collected[truck_id] += amount_collected 
                            self.bin_levels[edge] = ((kg_in_bin - amount_collected) / self.BIN_MAX_WEIGHT_KG) * 100.0
                            
                            try:
                                traci.poi.setColor(f"BIN_{edge}", (0, 255, 0, 255))
                                traci.poi.setType(f"BIN_{edge}", "Sạch ✅")
                            except: pass
                            
                            step_rewards[truck_id] += (amount_collected / 2.0) 
                            street_name = self.get_real_street_name(edge)
                            
                            if bin_level >= 70.0:
                                print(f"⏰ [{time_str}] 🚨 [{truck_id}] Thu {amount_collected:.1f}kg rác ĐỎ tại {street_name}.")
                            else:
                                print(f"⏰ [{time_str}] 🟡 [{truck_id}] Thu {amount_collected:.1f}kg rác tại {street_name}.")
                            
                            if self.current_load[truck_id] < self.MAX_CAPACITY_KG:
                                success = self.assign_urgent_target(truck_id)

                if edge != depot: step_rewards[truck_id] -= 0.1

                co2_emission_g_s = traci.vehicle.getCO2Emission(truck_id) / 1000.0
                self.trip_co2[truck_id] += (co2_emission_g_s * self.frame_skip)
                
                current_distance_km = traci.vehicle.getDistance(truck_id) / 1000.0
                self.trip_distance[truck_id] = current_distance_km
                avg_co2_per_km = (self.trip_co2[truck_id] / current_distance_km) if current_distance_km > 0.001 else 0.0

                # 🌟 BẢN VÁ: Tăng sức chịu đựng và chỉ tính kẹt khi xe THỰC SỰ ĐỨNG IM (< 0.5 km/h)
                if current_speed_kmh < 0.5: 
                    if not is_night_shift and (edge == depot or edge == ""): self.stuck_time[truck_id] = 0 
                    elif edge != depot: self.stuck_time[truck_id] += 1
                    else: self.stuck_time[truck_id] = 0
                else: 
                    self.stuck_time[truck_id] = 0

                next_states[truck_id] = np.array([
                    current_speed_kmh, avg_co2_per_km, float(self.stuck_time[truck_id]), 
                    current_distance_km, (self.current_load[truck_id] / self.MAX_CAPACITY_KG) * 100.0, float(self.current_fuel[truck_id])
                ], dtype=np.float32)
                
                if current_speed_kmh > 15.0: step_rewards[truck_id] += 3.0  
                elif current_speed_kmh < 1.0: step_rewards[truck_id] -= 2.0 
                step_rewards[truck_id] -= min((avg_co2_per_km / 500.0), 5.0) 

                # Nâng giới hạn chịu đựng lên 300 nhịp (Khoảng 5 phút kẹt xe thực tế) mới gọi cẩu
                if self.stuck_time[truck_id] > 300 or self.current_fuel[truck_id] <= 0:
                    print(f"⏰ [{time_str}] ⚠️ [{truck_id}] Kẹt cứng quá lâu ở {self.get_real_street_name(edge)}! Đang gọi cẩu...")
                    step_rewards[truck_id] -= 50.0
                    self.blacklist[truck_id][edge] = 500
                    try: traci.vehicle.remove(truck_id) 
                    except: pass

                rewards[truck_id] = step_rewards[truck_id]

        traci.simulationStep(traci.simulation.getTime() + self.frame_skip)
        self.current_step += self.frame_skip 
        
        for b, level in self.bin_levels.items():
            if level < 100.0:
                growth_per_sec = self.generators[b].get_fill_rate(self.current_step)
                self.bin_levels[b] = min(100.0, level + growth_per_sec * self.frame_skip)
                
            if self.current_step % (5 * self.frame_skip) == 0:
                try:
                    if self.bin_levels[b] >= 70.0:
                        traci.poi.setColor(f"BIN_{b}", (255, 0, 0, 255)) 
                        traci.poi.setType(f"BIN_{b}", f"RÁC ĐỎ ({int(self.bin_levels[b])}%)")
                    elif self.bin_levels[b] >= 40.0:
                        traci.poi.setColor(f"BIN_{b}", (255, 200, 0, 255)) 
                        traci.poi.setType(f"BIN_{b}", f"Vàng ({int(self.bin_levels[b])}%)")
                    elif self.bin_levels[b] > 5.0:
                        traci.poi.setColor(f"BIN_{b}", (0, 255, 0, 255)) 
                        traci.poi.setType(f"BIN_{b}", f"Xanh ({int(self.bin_levels[b])}%)")
                except: pass

        return next_states, rewards, terminated, {t: False for t in self.truck_ids}, {}

    def close(self):
        try: traci.close()
        except: pass