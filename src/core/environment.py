import os
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import torch
import datetime

# 🌟 CÔNG TẮC CHUYỂN ĐỔI
if os.environ.get("USE_GUI") == "1":
    import traci
    print("🖥️ [HỆ THỐNG] Đã ép dùng TraCI để hỗ trợ mở giao diện 3D (GUI).")
else:
    try:
        import libsumo as traci
        print("🚀 [HỆ THỐNG] Đã nạp thành công LIBSUMO - Chế độ Siêu Tốc được kích hoạt!")
    except ImportError:
        import traci
        print("🖥️ [HỆ THỐNG] Không tìm thấy libsumo, dùng TraCI (chạy chậm)...")

from core.waste_generator import RealWasteGenerator
from core.config import Config

import torch.nn as nn
class WasteForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(WasteForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class GreenVeinEnv(gym.Env):
    def __init__(self):
        super(GreenVeinEnv, self).__init__()
        
        self.sumo_cfg = os.path.join(Config.BASE_DIR, 'envs', 'greenvein.sumocfg')
        self.sumo_cmd = [
            "sumo", "-c", self.sumo_cfg, 
            "--no-warnings", 
            "--time-to-teleport", "-1",  # 🌟 TẮT XÓA XE: XE BẤT TỬ
            "--error-log", "sumo_error.log", "--no-step-log", "--mesosim", "true"
        ]
        
        self.truck_ids = ["XeRac_AI_1", "XeRac_AI_2", "XeRac_AI_3"]
        self.zone_names = {"XeRac_AI_1": "Cụm Tây", "XeRac_AI_2": "Cụm Trung Tâm", "XeRac_AI_3": "Cụm Đông"}
        self.color_map = {"XeRac_AI_1": (255, 100, 0), "XeRac_AI_2": (50, 150, 255), "XeRac_AI_3": (0, 255, 0)}

        self.CONFIG_QUY_HOACH = {
            "depots": {"XeRac_AI_1": "946030657", "XeRac_AI_2": "946030657", "XeRac_AI_3": "946030657"},
            "zones": {
                "XeRac_AI_1": ["707072725#2", "707066366#7", "707066366#11", "709017803#1", "179998311#2", "-180001033#9", "-198407217#3", "136524198#2", "1215063383", "707066366#9", "1012665674", "-219978979#1", "1208997907", "708576350#0", "178091734#1", "-1262082048", "-179995750#3", "-477417897#1"],
                "XeRac_AI_2": ["180082698#1", "1215943717#0", "1215943717#2", "711031662#4", "-180082702#0", "-601455486#1", "29313248#0", "1420319339", "25953535#0", "890573930#0", "-459315213#1", "597126919#1", "597113041", "-28958235#2", "-597111783#3", "601535720#1", "218427624#1", "1034359440#1"],
                "XeRac_AI_3": ["-675484248#3", "180001031#1", "-707366491#1", "1412423844#0", "38028986#4", "707087632#5", "-148202928#3", "-219863682", "-1461754606#4", "-11838452#1", "196054187#0", "194581852#1", "560585021#3", "-835632103#0", "1155941851#5", "-890573859", "-180082714#1", "-1276658079#1"]
            }
        }

        self.depot_edges = {} 
        self.frame_skip = 10 
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=10000.0, shape=(6,), dtype=np.float32)
        
        self.MAX_CAPACITY_KG = 8000.0 
        self.BIN_MAX_WEIGHT_KG = 1200.0 
        self.MAX_FUEL = 100.0 
        
        self.edge_centers = {}
        self.valid_edges_list = [] 
        self.passenger_edges = [] 
        self.zone_bins = {t: [] for t in self.truck_ids}
        self.bin_levels = {}
        self.route_cache_car = []
        self.blacklist = {t: {} for t in self.truck_ids}
        self.street_map = {}
        self.hanoi_streets = ["Tôn Đức Thắng", "Tây Sơn", "Chùa Bộc", "Thái Hà", "Thái Thịnh", "Đường Láng", "Nguyễn Trãi", "Xã Đàn", "Khâm Thiên", "Đê La Thành"]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_model = WasteForecaster().to(self.device)
        
        lstm_path = os.path.join(Config.BASE_DIR, 'src', 'lstm_module', 'waste_forecaster.pth')
        if os.path.exists(lstm_path):
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device, weights_only=True))
            self.lstm_model.eval()
        
        self.seq_length = 12
        self.history_buffer = {}
        self.last_record_time = 0
        self.target_bins = {t: "" for t in self.truck_ids}
        self.episode_completed = False
        self.cached_predictions = {}
        self.working_time = {t: 0.0 for t in self.truck_ids}

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

    def _get_predicted_trash_level(self, b):
        if b not in self.cached_predictions:
            history = list(self.history_buffer[b])
            while len(history) < self.seq_length: history.insert(0, history[0] if history else 0.0)
            seq_tensor = torch.tensor(history, dtype=torch.float32).view(1, self.seq_length, 1) / 100.0
            with torch.no_grad(): 
                self.cached_predictions[b] = self.lstm_model(seq_tensor.to(self.device)).item() * 100.0
        return self.cached_predictions[b]

    def assign_urgent_target(self, truck_id):
        depot = self.depot_edges.get(truck_id, "684766065#0")
        try: current_edge = traci.vehicle.getRoadID(truck_id)
        except: current_edge = depot
        if not current_edge or current_edge.startswith(":"): current_edge = depot

        total_trash_kg = sum((lvl / 100.0) * self.BIN_MAX_WEIGHT_KG for lvl in self.bin_levels.values())
        load_percent = self.current_load[truck_id] / self.MAX_CAPACITY_KG
        
        # 🌟 BÀN TAY SẮT: CHỈ VỀ TRẠM KHI ĐẦY > 90% HOẶC ĐÃ SẠCH BẢN ĐỒ
        if load_percent >= 0.90 or (total_trash_kg < 50.0 and self.current_load[truck_id] > 0):
            try:
                traci.vehicle.setRoute(truck_id, [current_edge, depot])
                self.target_bins[truck_id] = depot 
                return True
            except:
                # Nếu lỗi đường về trạm, bay thẳng về trạm!
                traci.vehicle.moveTo(truck_id, depot + "_0", 0)
                traci.vehicle.setRoute(truck_id, [depot])
                self.target_bins[truck_id] = depot
                return True

        curr_x, curr_y = self.edge_centers.get(current_edge, (0.0, 0.0))
        candidates = []
        active_targets = [tgt for tid, tgt in self.target_bins.items() if tid != truck_id]
        
        # Ưu tiên nhặt rác trong khu
        for b in self.zone_bins[truck_id]:
            lvl = self.bin_levels[b]
            if lvl >= 5.0 and b not in active_targets:
                dist = np.hypot(curr_x - self.edge_centers.get(b, (0.0, 0.0))[0], curr_y - self.edge_centers.get(b, (0.0, 0.0))[1])
                score = lvl / (dist + 5.0) 
                candidates.append((b, score))
                
        # Nếu khu mình sạch, đi nhặt phụ
        if not candidates:
            for b, lvl in self.bin_levels.items():
                if lvl >= 5.0 and b not in active_targets:
                    dist = np.hypot(curr_x - self.edge_centers.get(b, (0.0, 0.0))[0], curr_y - self.edge_centers.get(b, (0.0, 0.0))[1])
                    score = lvl / (dist + 50.0)  
                    candidates.append((b, score))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_bin = candidates[0][0]
            
            try:
                r1 = traci.simulation.findRoute(current_edge, best_bin, vType="garbage_truck")
                if r1 and len(r1.edges) > 0:
                    traci.vehicle.setRoute(truck_id, r1.edges)
                    self.target_bins[truck_id] = best_bin
                    return True
            except: pass
            
            # 🌟 NẾU KHÔNG CÓ ĐƯỜNG ĐI, BAY THẲNG ĐẾN ĐIỂM RÁC ĐÓ NHẶT (Không bao giờ bỏ cuộc)
            try:
                traci.vehicle.moveTo(truck_id, best_bin + "_0", 0)
                traci.vehicle.setRoute(truck_id, [best_bin])
                self.target_bins[truck_id] = best_bin
                return True
            except: pass

        # Nếu thực sự không còn rác nào và bụng chưa đầy thì đi lang thang
        try:
            next_edge = random.choice(self.valid_edges_list)
            traci.vehicle.changeTarget(truck_id, next_edge)
            self.target_bins[truck_id] = next_edge
        except: pass
        return False

    def get_target_traffic(self, hour, day_of_week):
        is_weekend = day_of_week >= 5 
        if is_weekend: return 150 if hour >= 22.5 or hour < 5 else 350
        else: return 50 if hour >= 22.5 or hour < 5 else 450

    def reset(self, seed=None, options=None, current_episode=1):
        super().reset(seed=seed)
        try: traci.close()
        except: pass
            
        traci.start(self.sumo_cmd)
        self.is_gui = (self.sumo_cmd[0] == "sumo-gui")
        self.day_of_week = (current_episode - 1) % 7 
        print(f"\n📅 [LỊCH TRÌNH] Episode {current_episode} | Khởi tạo lưới mô phỏng...")
        
        self.virtual_time_seconds = 20 * 3600 
        try: traci.simulation.setScale(0.1)
        except: pass

        self.current_step = 0
        
        vehicle_types = [
            ("garbage_truck", "ignoring", "truck", (255,255,255), 10.0, 3.5, 15.0),
            ("passenger_car", "passenger", "passenger", (210,210,210), 4.5, 1.8, 12.0)
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

        all_edges = traci.edge.getIDList()
        self.valid_edges_list = []
        self.passenger_edges = []
        
        for edge_id in all_edges:
            if edge_id.startswith(":"): continue
            try:
                allowed = traci.lane.getAllowed(edge_id + "_0")
                disallowed = traci.lane.getDisallowed(edge_id + "_0")
                is_passenger_ok = ("passenger" in allowed) or (len(allowed) == 0 and "passenger" not in disallowed)
                if is_passenger_ok: self.passenger_edges.append(edge_id)
                self.valid_edges_list.append(edge_id)
            except: pass

        self.edge_centers = {}
        for edge_id in self.valid_edges_list:
            try:
                shape = traci.lane.getShape(edge_id + "_0") 
                self.edge_centers[edge_id] = (sum([p[0] for p in shape])/len(shape), sum([p[1] for p in shape])/len(shape))
            except: self.edge_centers[edge_id] = (0.0, 0.0)

        self.zone_bins = {t: [] for t in self.truck_ids}
        self.bin_levels = {}
        self.depot_edges = {}

        for truck_id in self.truck_ids:
            user_depot = self.CONFIG_QUY_HOACH["depots"].get(truck_id, "")
            if user_depot and user_depot in self.valid_edges_list:
                self.depot_edges[truck_id] = user_depot
            else:
                self.depot_edges[truck_id] = random.choice(self.passenger_edges if self.passenger_edges else self.valid_edges_list)

        for truck_id in self.truck_ids:
            depot = self.depot_edges[truck_id]
            for b in self.CONFIG_QUY_HOACH["zones"].get(truck_id, []):
                if b not in self.valid_edges_list: continue
                
                is_reachable = False
                try:
                    r_test = traci.simulation.findRoute(depot, b, vType="garbage_truck")
                    if r_test and len(r_test.edges) > 0:
                        is_reachable = True
                except: pass
                
                if is_reachable:
                    self.zone_bins[truck_id].append(b)
                    lvl = random.uniform(60.0, 100.0) 
                    self.bin_levels[b] = lvl
                else:
                    bx, by = self.edge_centers.get(b, (0.0, 0.0))
                    if bx == 0.0 and by == 0.0: continue
                    best_fallback = None
                    min_dist = float('inf')
                    search_pool = self.passenger_edges if self.passenger_edges else self.valid_edges_list
                    for valid_e in search_pool:
                        vx, vy = self.edge_centers[valid_e]
                        d = np.hypot(bx - vx, by - vy)
                        if d < min_dist:
                            try:
                                r_check = traci.simulation.findRoute(depot, valid_e, vType="garbage_truck")
                                if r_check and len(r_check.edges) > 0:
                                    min_dist = d
                                    best_fallback = valid_e
                            except: pass
                    
                    if best_fallback:
                        self.zone_bins[truck_id].append(best_fallback)
                        lvl = random.uniform(60.0, 100.0)
                        self.bin_levels[best_fallback] = max(lvl, self.bin_levels.get(best_fallback, 0.0))

        self.street_map.clear() 
        self.blacklist = {t: {} for t in self.truck_ids}
        self.target_bins = {t: "" for t in self.truck_ids} 
        self.episode_completed = False
        self.cached_predictions = {}

        for b in self.bin_levels.keys():
            self.history_buffer[b] = deque([self.bin_levels[b]]*self.seq_length, maxlen=self.seq_length)

        total_trash_kg = sum((lvl / 100.0) * self.BIN_MAX_WEIGHT_KG for lvl in self.bin_levels.values())
        print(f"📦 [CHỐT HỒ SƠ QUY HOẠCH] Tổng lượng rác thực tế trên bản đồ: {total_trash_kg:.1f} kg.")

        self.route_cache_car = []
        target_pool = self.passenger_edges if self.passenger_edges else self.valid_edges_list
        for _ in range(1000):
            if len(self.route_cache_car) >= 300: break
            s = random.choice(target_pool)
            e = random.choice(target_pool)
            try:
                r = traci.simulation.findRoute(s, e, vType="passenger_car")
                if r and len(r.edges) >= 4: self.route_cache_car.append(r.edges)
            except: pass

        initial_traffic = self.get_target_traffic(20.0, self.day_of_week)
        day_type = "Cuối tuần" if self.day_of_week >= 5 else "Ngày thường"
        print(f"🚗 Khởi tạo giao thông ({day_type} - 20:00) với {initial_traffic} xe nền...")
        
        for i in range(initial_traffic):
            v_type = random.choice(["passenger_car", "passenger_car"])
            if self.route_cache_car:
                vid = f"xe_dan_init_{i}_{random.randint(100, 9999)}"
                try:
                    traci.route.add(f"route_{vid}", random.choice(self.route_cache_car))
                    traci.vehicle.add(vid, f"route_{vid}", typeID=v_type, depart=str(i))
                except: pass

        if self.is_gui:
            for b, level in self.bin_levels.items():
                tx, ty = self.edge_centers.get(b, (0.0, 0.0))
                try: 
                    color = (255, 0, 0, 255) if level >= 70.0 else (255, 200, 0, 255) 
                    width = 25.0 if level >= 70.0 else 20.0
                    traci.poi.add(f"BIN_{b}", tx, ty, color, poiType=f"Rác: {int(level)}%", layer=100, width=width, height=width)
                except: pass

        self.current_load = {t: 0.0 for t in self.truck_ids}
        self.current_fuel = {t: self.MAX_FUEL for t in self.truck_ids} 
        self.is_done = {t: False for t in self.truck_ids} 
        self.stuck_time = {t: 0 for t in self.truck_ids}
        self.trip_co2 = {t: 0.0 for t in self.truck_ids}
        self.trip_distance = {t: 0.0 for t in self.truck_ids}
        self.total_collected = {t: 0.0 for t in self.truck_ids} 
        self.has_departed = {t: False for t in self.truck_ids}
        self.working_time = {t: 0.0 for t in self.truck_ids}

        for truck_id in self.truck_ids:
            try:
                depot_edge = self.depot_edges[truck_id]
                route_id = f"route_init_{truck_id}"
                traci.route.add(route_id, [depot_edge])
                traci.vehicle.add(truck_id, route_id, typeID="garbage_truck", depart="now")
                traci.vehicle.setColor(truck_id, self.color_map[truck_id])
            except: pass                       

        for _ in range(5): traci.simulationStep()
        for truck_id in self.truck_ids: self.assign_urgent_target(truck_id)
        return {truck_id: np.zeros(6, dtype=np.float32) for truck_id in self.truck_ids}, {}

    def step(self, action_dict):
        next_states = {t: np.zeros(6, dtype=np.float32) for t in self.truck_ids}
        rewards = {t: 0.0 for t in self.truck_ids}
        terminated = {t: False for t in self.truck_ids}

        self.virtual_time_seconds += self.frame_skip
        hour = (self.virtual_time_seconds % 86400) / 3600.0
        time_str = f"{int(hour):02d}:{int((hour % 1) * 60):02d}"
        
        self.cached_predictions.clear()

        try: active_vehicles = traci.vehicle.getIDList()
        except:
            for t in self.truck_ids: terminated[t] = True
            return next_states, rewards, terminated, {t: False for t in self.truck_ids}, {}
        
        if self.is_gui:
            for truck_id in self.truck_ids:
                if truck_id in active_vehicles:
                    try:
                        x, y = traci.vehicle.getPosition(truck_id)
                        poi_id = f"TRACKER_{truck_id}"
                        if poi_id in traci.poi.getIDList(): traci.poi.setPosition(poi_id, x, y)
                        else: traci.poi.add(poi_id, x, y, self.color_map[truck_id], poiType=f"🚛 {self.zone_names[truck_id]}", layer=300, width=30.0, height=30.0)
                    except: pass

        for t in self.truck_ids:
            if t in active_vehicles: self.has_departed[t] = True
        
        target_traffic = self.get_target_traffic(hour, self.day_of_week)
        active_truck_count = sum(1 for t in self.truck_ids if t in active_vehicles)
        num_bg_cars = len(active_vehicles) - active_truck_count 
        
        if num_bg_cars < target_traffic and self.current_step % (5 * self.frame_skip) == 0:
            spawn_amount = min(20, target_traffic - num_bg_cars) 
            for i in range(spawn_amount): 
                v_type = random.choice(["passenger_car", "passenger_car"])
                if self.route_cache_car:
                    vid = f"xe_dan_{self.current_step}_{i}_{random.randint(100, 9999)}"
                    try:
                        depart_time = str(int(self.current_step) + random.randint(1, 15))
                        traci.route.add(f"route_{vid}", random.choice(self.route_cache_car))
                        traci.vehicle.add(vid, f"route_{vid}", typeID=v_type, depart=depart_time)
                    except: pass

        dt = 1.0
        connection_closed = False
        active_bins = {b: lvl for b, lvl in self.bin_levels.items() if lvl >= 5.0}

        for _ in range(self.frame_skip):
            try:
                traci.simulationStep()
                active_trucks = traci.vehicle.getIDList()
                for t_id in self.truck_ids:
                    if t_id in active_trucks and not self.is_done[t_id]:
                        
                        if self.working_time[t_id] > 0:
                            self.working_time[t_id] -= dt
                            try: traci.vehicle.setSpeed(t_id, 0.0)
                            except: pass
                            co2 = max(0.0, traci.vehicle.getCO2Emission(t_id))
                            self.trip_co2[t_id] += (co2 * dt) / 1000.0
                            continue
                        
                        speed_m_s = max(0.0, traci.vehicle.getSpeed(t_id))
                        self.trip_distance[t_id] += (speed_m_s * dt) / 1000.0
                        co2 = max(0.0, traci.vehicle.getCO2Emission(t_id))
                        self.trip_co2[t_id] += (co2 * dt) / 1000.0
                        
                        curr_e = traci.vehicle.getRoadID(t_id)
                        if not curr_e or curr_e.startswith(":"): continue
                        
                        vx, vy = traci.vehicle.getPosition(t_id)
                        depot_e = self.depot_edges[t_id]
                        
                        is_heading_depot = (self.target_bins.get(t_id) == depot_e)
                        if is_heading_depot and (curr_e == depot_e or curr_e == "-" + depot_e):
                            if self.current_load[t_id] > 0:
                                print(f"⏰ [{time_str}] ♻️ [{t_id}] Đổ {self.current_load[t_id]:.1f}kg rác tại Trạm.")
                                self.current_load[t_id] = 0.0
                                self.working_time[t_id] = 120.0 # 2 phút đổ rác để đẩy nhanh tiến độ
                                self.target_bins[t_id] = "" 
                                self.assign_urgent_target(t_id) 
                        
                        # 🌟 LOGIC THU GOM RÁC CỰC MẠNH: CHỈ CẦN CÙNG TÊN ĐƯỜNG LÀ THU NGAY
                        for b_e in list(active_bins.keys()):
                            if self.current_load[t_id] >= self.MAX_CAPACITY_KG: break
                            
                            is_near = False
                            # Sai số mạnh: Chung ID đường hoặc cách nhau 80m đều tính là tới nơi
                            if curr_e == b_e or curr_e == "-" + b_e or curr_e.replace("-", "") == b_e.replace("-", ""):
                                is_near = True
                            else:
                                bx, by = self.edge_centers[b_e]
                                if np.hypot(vx-bx, vy-by) < 80.0:
                                    is_near = True

                            if is_near:
                                lvl = self.bin_levels[b_e]
                                kg_in_bin = (lvl / 100.0) * self.BIN_MAX_WEIGHT_KG
                                amt = min(kg_in_bin, self.MAX_CAPACITY_KG - self.current_load[t_id])
                                self.current_load[t_id] += amt
                                self.total_collected[t_id] += amt
                                
                                new_level = ((kg_in_bin - amt) / self.BIN_MAX_WEIGHT_KG) * 100.0
                                self.bin_levels[b_e] = 0.0 if new_level < 1.0 else new_level
                                
                                if self.bin_levels[b_e] < 5.0:
                                    del active_bins[b_e]
                                    if self.is_gui:
                                        try:
                                            traci.poi.setColor(f"BIN_{b_e}", (0, 255, 0, 255))
                                            traci.poi.setType(f"BIN_{b_e}", "Sạch ✅")
                                        except: pass
                                
                                print(f"⏰ [{time_str}] 🟡 [{t_id}] Thu {amt:.1f}kg tại {self.get_real_street_name(b_e)}.")
                                self.working_time[t_id] = 60.0 # 1 phút cẩu rác
                                self.target_bins[t_id] = ""
                                self.assign_urgent_target(t_id)
                                break 
            except: 
                connection_closed = True
                break

        if connection_closed:
            for t in self.truck_ids: terminated[t] = True
            return next_states, rewards, terminated, {t: False for t in self.truck_ids}, {}

        self.current_step += self.frame_skip 
        try: active_after = traci.vehicle.getIDList()
        except:
            for t in self.truck_ids: terminated[t] = True
            return next_states, rewards, terminated, {t: False for t in self.truck_ids}, {}

        for t in self.truck_ids:
            if t not in active_after:
                if self.has_departed[t] and not self.is_done[t]:
                    try:
                        depot_edge = self.depot_edges[t]
                        new_r = f"r_respawn_{t}_{self.current_step}"
                        traci.route.add(new_r, [depot_edge])
                        traci.vehicle.add(t, new_r, typeID="garbage_truck", depart="now")
                        traci.vehicle.setColor(t, self.color_map[t])
                        self.target_bins[t] = ""
                        self.assign_urgent_target(t)
                    except: pass
                continue

            act = action_dict.get(t, 1)
            
            if self.working_time[t] > 0:
                try: traci.vehicle.setSpeed(t, 0.0)
                except: pass
            else:
                try: traci.vehicle.setSpeed(t, [3.0, 0.0, 15.0][act])
                except: pass

            try: current_speed_kmh = traci.vehicle.getSpeed(t) * 3.6
            except: current_speed_kmh = 0.0

            self.current_fuel[t] -= (0.015 if current_speed_kmh > 1.0 else 0.005) * self.frame_skip
            current_distance_km = self.trip_distance.get(t, 0.0)
            avg_co2_per_km = (self.trip_co2[t] / current_distance_km) if current_distance_km > 0.001 else 0.0

            if current_speed_kmh < 0.5 and self.working_time[t] <= 0: 
                self.stuck_time[t] += 1
            else: 
                self.stuck_time[t] = 0

            # NẾU KẸT QUÁ 150 GIÂY THÌ NHẢY THẲNG TỚI ĐIỂM RÁC LUÔN
            if self.stuck_time[t] > 15:
                tgt = self.target_bins.get(t)
                if tgt and tgt not in ["", "WANDERING"]:
                    try:
                        traci.vehicle.moveTo(t, tgt + "_0", 0)
                        traci.vehicle.setRoute(t, [tgt])
                        self.stuck_time[t] = 0
                    except: pass
                else:
                    self.assign_urgent_target(t)

            if self.target_bins.get(t) == "": self.assign_urgent_target(t)
            
            next_states[t] = np.array([current_speed_kmh, avg_co2_per_km, float(self.stuck_time[t]), current_distance_km, (self.current_load[t]/self.MAX_CAPACITY_KG)*100.0, 100.0], dtype=np.float32)

        total_trash_kg = sum((lvl / 100.0) * self.BIN_MAX_WEIGHT_KG for lvl in self.bin_levels.values())
        if total_trash_kg < 50.0:
            if not self.episode_completed:
                self.episode_completed = True
                print(f"\n🎉 [{time_str}] NHIỆM VỤ HOÀN THÀNH: Toàn bộ rác đã được dọn sạch!")
            for tid in self.truck_ids:
                self.is_done[tid] = True
                terminated[tid] = True

        return next_states, {t: 0.0 for t in self.truck_ids}, terminated, {t: False for t in self.truck_ids}, {}

    def close(self):
        try: traci.close()
        except: pass