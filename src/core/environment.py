import os
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import torch
import datetime

# 🌟 CÔNG TẮC CHUYỂN ĐỔI: Dùng TraCI nếu đang mở GUI, dùng libsumo nếu chạy ngầm
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
            "--no-warnings", "--time-to-teleport", "-1", 
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_model = WasteForecaster().to(self.device)
        
        lstm_path = os.path.join(Config.BASE_DIR, 'src', 'lstm_module', 'waste_forecaster.pth')
        if os.path.exists(lstm_path):
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device, weights_only=True))
            self.lstm_model.eval()
        
        self.seq_length = 12
        self.history_buffer = {}
        self.last_record_time = 0
        self.target_bins = {}

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
        try:
            r = traci.simulation.findRoute(current_edge, target_edge, vType="garbage_truck")
            if r and len(r.edges) > 0:
                traci.vehicle.changeTarget(truck_id, target_edge)
                self.is_heading_depot[truck_id] = is_depot
                return True
            return False
        except:
            return False

    def _wander(self, truck_id):
        try:
            curr_edge = traci.vehicle.getRoadID(truck_id)
            if not curr_edge or curr_edge.startswith(":"): return
            
            for _ in range(10):
                if self.valid_edges_list:
                    next_edge = random.choice(self.valid_edges_list)
                    r = traci.simulation.findRoute(curr_edge, next_edge, vType="garbage_truck")
                    if r and len(r.edges) > 0:
                        traci.vehicle.changeTarget(truck_id, next_edge)
                        self.target_bins[truck_id] = next_edge
                        return
        except: pass

    def assign_urgent_target(self, truck_id):
        depot = self.depot_edges.get(truck_id, "684766065#0")
        try: current_edge = traci.vehicle.getRoadID(truck_id)
        except: current_edge = depot
        if not current_edge or current_edge.startswith(":"): current_edge = depot
        
        try: traci.vehicle.setSpeed(truck_id, -1.0) 
        except: pass

        for b in list(self.blacklist[truck_id].keys()):
            self.blacklist[truck_id][b] -= 10
            if self.blacklist[truck_id][b] <= 0: del self.blacklist[truck_id][b]

        total_trash_on_map = sum(self.bin_levels.values())
        if (self.current_load[truck_id] / self.MAX_CAPACITY_KG) >= 0.80 or total_trash_on_map < 10.0:
            if current_edge == depot: 
                self.is_heading_depot[truck_id] = True
                self.target_bins[truck_id] = depot 
                return True
            if self._route_to_target(truck_id, current_edge, depot, depot, is_depot=True): 
                self.target_bins[truck_id] = depot
                return True
            self._wander(truck_id)
            return False

        curr_x, curr_y = self.edge_centers.get(current_edge, (0.0, 0.0))
        candidates = []
        
        for b in self.zone_bins[truck_id]:
            actual_lvl = self.bin_levels[b]
            if b == current_edge or b in self.blacklist[truck_id] or actual_lvl < 5.0: continue
            
            history = list(self.history_buffer[b])
            while len(history) < self.seq_length: history.insert(0, history[0] if history else 0.0)
            seq_tensor = torch.tensor(history, dtype=torch.float32).view(1, self.seq_length, 1) / 100.0
            with torch.no_grad(): 
                predicted_lvl = self.lstm_model(seq_tensor.to(self.device)).item() * 100.0
            
            dist = np.hypot(curr_x - self.edge_centers.get(b, (0.0, 0.0))[0], curr_y - self.edge_centers.get(b, (0.0, 0.0))[1])
            combined_urgency = (actual_lvl * 0.7) + (max(0, predicted_lvl) * 0.3)
            score = combined_urgency / (dist + 50.0) 
            candidates.append((b, score, actual_lvl))
            
        if not candidates:
            for b, lvl in self.bin_levels.items():
                if b in self.zone_bins[truck_id] or b == current_edge or b in self.blacklist[truck_id] or lvl < 5.0: continue
                
                history = list(self.history_buffer[b])
                while len(history) < self.seq_length: history.insert(0, history[0] if history else 0.0)
                seq_tensor = torch.tensor(history, dtype=torch.float32).view(1, self.seq_length, 1) / 100.0
                with torch.no_grad(): 
                    predicted_lvl = self.lstm_model(seq_tensor.to(self.device)).item() * 100.0

                dist = np.hypot(curr_x - self.edge_centers.get(b, (0.0, 0.0))[0], curr_y - self.edge_centers.get(b, (0.0, 0.0))[1])
                combined_urgency = (lvl * 0.7) + (max(0, predicted_lvl) * 0.3)
                score = combined_urgency / (dist + 200.0)  
                candidates.append((b, score, lvl))

        if not candidates:
            if current_edge == depot:
                self.is_heading_depot[truck_id] = True
                try: traci.vehicle.setSpeed(truck_id, 0.0)
                except: pass
                self.target_bins[truck_id] = depot
                return True 
            if self._route_to_target(truck_id, current_edge, depot, depot, is_depot=True): 
                self.target_bins[truck_id] = depot
                return True
            self._wander(truck_id)
            return False
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for best_bin, score, a_lvl in candidates[:15]:
            if self._route_to_target(truck_id, current_edge, best_bin, depot, is_depot=False):
                try: traci.vehicle.setSpeed(truck_id, -1) 
                except: pass
                self.target_bins[truck_id] = best_bin 
                return True
            else:
                self.blacklist[truck_id][best_bin] = 100 
        
        if current_edge == depot: 
            self.target_bins[truck_id] = depot
            return True 
        if self._route_to_target(truck_id, current_edge, depot, depot, is_depot=True): 
            self.target_bins[truck_id] = depot
            return True
        self._wander(truck_id)
        return False

    def get_target_traffic(self, hour, day_of_week):
        is_weekend = day_of_week >= 5 
        if is_weekend:
            if 19 <= hour <= 22.5: return 500      
            elif hour >= 22.5 or hour < 5: return 100 
            else: return 300                       
        else:
            if 7 <= hour <= 9.5 or 17 <= hour <= 19.5: return 800 
            elif 19.5 < hour <= 22.5: return 300                  
            elif hour >= 22.5 or hour < 5: return 50              
            else: return 400                                      

    def reset(self, seed=None, options=None, current_episode=1):
        super().reset(seed=seed)
        try: traci.close()
        except: pass
            
        traci.start(self.sumo_cmd)
        
        self.is_gui = (self.sumo_cmd[0] == "sumo-gui")
        
        self.day_of_week = (current_episode - 1) % 7 
        day_names = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
        print(f"\n📅 [LỊCH TRÌNH] Episode {current_episode}: Hôm nay là {day_names[self.day_of_week]}")
        
        self.virtual_time_seconds = 20 * 3600 
        
        try: traci.simulation.setScale(0.1)
        except: pass

        self.current_step = 0
        all_edges = traci.edge.getIDList()
        self.valid_edges_list = []
        self.valid_passenger_edges = []
        self.valid_motorcycle_edges = []
        self.street_map.clear() 
        self.blacklist = {t: {} for t in self.truck_ids}
        self.target_bins = {t: "" for t in self.truck_ids} 
        
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

        self.history_buffer = {}
        self.last_record_time = 0

        for truck_id in self.truck_ids:
            zone_type = "commercial" if truck_id == "XeRac_AI_2" else "residential"
            
            for b in self.zone_bins[truck_id]:
                if self.day_of_week >= 5: 
                    if zone_type == "commercial":
                        initial_trash = random.uniform(70.0, 100.0) 
                    else:
                        initial_trash = random.uniform(50.0, 95.0)  
                else: 
                    if zone_type == "commercial":
                        initial_trash = random.uniform(60.0, 90.0)  
                    else:
                        initial_trash = random.uniform(40.0, 75.0)  
                    
                self.bin_levels[b] = initial_trash
                self.generators[b] = RealWasteGenerator(zone_type=zone_type)
                
                self.history_buffer[b] = deque(maxlen=self.seq_length)
                for _ in range(self.seq_length):
                    self.history_buffer[b].append(initial_trash)

        print("⏳ Đang thiết lập ma trận định tuyến...")
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

        initial_traffic = self.get_target_traffic(20.0, self.day_of_week)
        day_type = "Cuối tuần" if self.day_of_week >= 5 else "Ngày thường"
        print(f"🚗 Khởi tạo giao thông ({day_type} - 20:00) với {initial_traffic} xe nền...")
        
        for i in range(initial_traffic):
            v_type = random.choice(["passenger_car", "passenger_car", "motorcycle_lead", "motorcycle_norm"])
            cache = self.route_cache_moto if "motorcycle" in v_type else self.route_cache_car
            if cache:
                vid = f"xe_dan_init_{i}_{random.randint(100, 9999)}"
                try:
                    traci.route.add(f"route_{vid}", random.choice(cache))
                    traci.vehicle.add(vid, f"route_{vid}", typeID=v_type, departPos="random")
                except: pass

        if self.is_gui:
            for b, level in self.bin_levels.items():
                tx, ty = self.edge_centers.get(b, (0.0, 0.0))
                try: 
                    if level >= 70.0: 
                        color = (255, 0, 0, 255)
                        width = 25.0
                    else: 
                        color = (255, 200, 0, 255) 
                        width = 20.0
                    
                    traci.poi.add(f"BIN_{b}", tx, ty, color, poiType=f"Rác: {int(level)}%", layer=100, width=width, height=width)
                except: pass

        self.current_load = {t: 0.0 for t in self.truck_ids}
        self.current_fuel = {t: self.MAX_FUEL for t in self.truck_ids} 
        self.is_heading_depot = {t: False for t in self.truck_ids}
        self.is_done = {t: False for t in self.truck_ids} 
        self.stuck_time = {t: 0 for t in self.truck_ids}
        self.trip_co2 = {t: 0.0 for t in self.truck_ids}
        self.trip_distance = {t: 0.0 for t in self.truck_ids}
        self.total_collected = {t: 0.0 for t in self.truck_ids} 
        self.has_departed = {t: False for t in self.truck_ids}

        for truck_id in self.truck_ids:
            try:
                safe_route_id = f"route_init_{truck_id}_{random.randint(1000, 9999)}"
                traci.route.add(safe_route_id, [self.depot_edges[truck_id]])
                traci.vehicle.add(truck_id, safe_route_id, typeID="garbage_truck")
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

        self.virtual_time_seconds += self.frame_skip
        hour = (self.virtual_time_seconds % 86400) / 3600.0
        time_str = f"{int(hour):02d}:{int((hour % 1) * 60):02d}"
        is_night_shift = (hour >= 20.0) or (hour < 5.0)

        active_vehicles = traci.vehicle.getIDList()
        
        if self.is_gui:
            for truck_id in self.truck_ids:
                if truck_id in active_vehicles:
                    try:
                        x, y = traci.vehicle.getPosition(truck_id)
                        poi_id = f"TRACKER_{truck_id}"
                        if poi_id in traci.poi.getIDList():
                            traci.poi.setPosition(poi_id, x, y)
                        else:
                            traci.poi.add(poi_id, x, y, self.color_map[truck_id], poiType=f"🚛 {self.zone_names[truck_id]}", layer=300, width=30.0, height=30.0)
                    except: pass

        target_traffic = self.get_target_traffic(hour, self.day_of_week)
        
        active_truck_count = sum(1 for t in self.truck_ids if t in active_vehicles)
        num_bg_cars = len(active_vehicles) - active_truck_count 
        
        if num_bg_cars < target_traffic and self.current_step % (5 * self.frame_skip) == 0:
            spawn_amount = min(20, target_traffic - num_bg_cars) 
            for i in range(spawn_amount): 
                v_type = random.choice(["passenger_car", "passenger_car", "motorcycle_lead", "motorcycle_norm"])
                cache = self.route_cache_moto if "motorcycle" in v_type else self.route_cache_car
                if cache:
                    vid = f"xe_dan_{self.current_step}_{i}_{random.randint(100, 9999)}"
                    try:
                        traci.route.add(f"route_{vid}", random.choice(cache))
                        traci.vehicle.add(vid, f"route_{vid}", typeID=v_type, departPos="random")
                    except: pass

        for t in self.truck_ids:
            if t in active_vehicles: self.has_departed[t] = True
        
        total_trash_on_map = sum(self.bin_levels.values())
        
        for truck_id in self.truck_ids:
            depot = self.depot_edges.get(truck_id, "684766065#0")
            
            if truck_id not in active_vehicles:
                if not self.has_departed[truck_id]: continue 
                
                if not self.is_done[truck_id]:
                    if total_trash_on_map >= 10.0:
                        try:
                            safe_route_id = f"route_respawn_{truck_id}_{self.current_step}_{random.randint(1000, 9999)}"
                            traci.route.add(safe_route_id, [depot])
                            traci.vehicle.add(truck_id, safe_route_id, typeID="garbage_truck", departPos="random")
                            traci.vehicle.changeTarget(truck_id, depot)
                            traci.vehicle.setColor(truck_id, self.color_map[truck_id])
                            self.stuck_time[truck_id] = 0
                            self.assign_urgent_target(truck_id)
                        except: pass
                        rewards[truck_id] = -20.0
                        terminated[truck_id] = False
                    else:
                        self.is_done[truck_id] = True
                        print(f"⏰ [{time_str}] 🏁 [{truck_id}] HOÀN THÀNH SỚM: Đã dọn sạch rác.")
                        rewards[truck_id] = 200.0  
                        terminated[truck_id] = True 
            else:
                # 🌟 KHỞI TẠO BIẾN MẶC ĐỊNH CHỐNG LỖI PYLANCE
                current_speed_kmh = 0.0
                current_distance_km = self.trip_distance.get(truck_id, 0.0)
                avg_co2_per_km = 0.0
                
                act = action_dict.get(truck_id, 1)
                try: edge = traci.vehicle.getRoadID(truck_id)
                except: edge = ""
                
                if not is_night_shift: act = 1 if (edge == depot or edge == "") else 2
                if edge == depot: act = 0

                try:
                    if act == 0: traci.vehicle.setSpeed(truck_id, 3.0)      
                    elif act == 1: traci.vehicle.setSpeed(truck_id, 0.0)   
                    elif act == 2: traci.vehicle.setSpeed(truck_id, 15.0)   
                except: pass

                try:
                    route = traci.vehicle.getRoute(truck_id)
                    route_index = traci.vehicle.getRouteIndex(truck_id)
                    if route_index >= len(route) - 2:
                        self._wander(truck_id)
                except: pass

                try:
                    current_speed_m_s = traci.vehicle.getSpeed(truck_id)
                except:
                    current_speed_m_s = 0.0

                current_speed_kmh = current_speed_m_s * 3.6 
                
                step_distance_km = (max(0.0, current_speed_m_s) * self.frame_skip) / 1000.0
                self.trip_distance[truck_id] += step_distance_km
                current_distance_km = self.trip_distance[truck_id]

                step_rewards[truck_id] -= 0.1 
                fuel_consumed = (0.015 if current_speed_kmh > 1.0 else 0.005) * self.frame_skip
                self.current_fuel[truck_id] -= fuel_consumed
                step_rewards[truck_id] -= (fuel_consumed * 1.0) 

                try:
                    vx, vy = traci.vehicle.getPosition(truck_id)
                except:
                    vx, vy = 0.0, 0.0

                is_at_depot = False
                if edge == depot:
                    is_at_depot = True
                elif vx != 0.0:
                    dx, dy = self.edge_centers.get(depot, (0.0, 0.0))
                    if np.hypot(vx - dx, vy - dy) < 80.0: 
                        is_at_depot = True

                if is_at_depot:
                    if self.current_fuel[truck_id] < 30.0: 
                        self.current_fuel[truck_id] = self.MAX_FUEL
                        
                    if self.current_load[truck_id] > 0:
                        is_full = self.current_load[truck_id] >= (self.MAX_CAPACITY_KG * 0.8)
                        is_map_clean = total_trash_on_map < 10.0
                        
                        if is_full or is_map_clean:
                            dumped_amount = self.current_load[truck_id]
                            self.current_load[truck_id] = 0.0
                            step_rewards[truck_id] += dumped_amount * 0.5 
                            print(f"⏰ [{time_str}] ♻️ [{truck_id}] Đổ {dumped_amount:.1f}kg rác tại Trạm.")
                            
                            self.target_bins[truck_id] = "" 
                            self.assign_urgent_target(truck_id)
                else:
                    collected_any = False
                    for b_edge, bin_level in self.bin_levels.items():
                        if bin_level >= 5.0 and self.current_load[truck_id] < self.MAX_CAPACITY_KG:
                            is_near = False
                            if edge == b_edge:
                                is_near = True
                            elif vx != 0.0:
                                tx, ty = self.edge_centers.get(b_edge, (0.0, 0.0))
                                if np.hypot(vx - tx, vy - ty) < 150.0:
                                    is_near = True
                            
                            if is_near:
                                kg_in_bin = (bin_level / 100.0) * self.BIN_MAX_WEIGHT_KG
                                amount_collected = min(kg_in_bin, self.MAX_CAPACITY_KG - self.current_load[truck_id])
                                
                                self.current_load[truck_id] += amount_collected
                                self.total_collected[truck_id] += amount_collected 
                                
                                new_level = ((kg_in_bin - amount_collected) / self.BIN_MAX_WEIGHT_KG) * 100.0
                                self.bin_levels[b_edge] = 0.0 if new_level < 1.0 else new_level
                                
                                if self.is_gui:
                                    try:
                                        traci.poi.setColor(f"BIN_{b_edge}", (0, 255, 0, 255))
                                        traci.poi.setType(f"BIN_{b_edge}", "Sạch ✅")
                                        traci.poi.setWidth(f"BIN_{b_edge}", 10.0)
                                        traci.poi.setHeight(f"BIN_{b_edge}", 10.0)
                                    except: pass
                                
                                step_rewards[truck_id] += (amount_collected * 2.0) 
                                street_name = self.get_real_street_name(b_edge)
                                print(f"⏰ [{time_str}] 🟡 [{truck_id}] Thu {amount_collected:.1f}kg rác tại {street_name} (Tiện đường).")
                                collected_any = True

                    if collected_any:
                        self.target_bins[truck_id] = "" 
                        if self.current_load[truck_id] < self.MAX_CAPACITY_KG:
                            self.assign_urgent_target(truck_id)

                try:
                    co2_emission_g_s = traci.vehicle.getCO2Emission(truck_id) / 1000.0
                except:
                    co2_emission_g_s = 0.0
                    
                self.trip_co2[truck_id] += (co2_emission_g_s * self.frame_skip)
                avg_co2_per_km = (self.trip_co2[truck_id] / current_distance_km) if current_distance_km > 0.001 else 0.0

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

                rewards[truck_id] = step_rewards[truck_id]

        for _ in range(self.frame_skip):
            try:
                traci.simulationStep()
            except:
                break
                
        self.current_step += self.frame_skip 

        if self.current_step - self.last_record_time >= 900:
            for b in self.bin_levels.keys():
                self.history_buffer[b].append(self.bin_levels[b])
            self.last_record_time = self.current_step

        return next_states, rewards, terminated, {t: False for t in self.truck_ids}, {}

    def close(self):
        try: traci.close()
        except: pass