import os
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import torch

# 🌟 CÔNG TẮC CHUYỂN ĐỔI HỆ THỐNG TRACI/LIBSUMO
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
            "--no-warnings", "true",
            "--time-to-teleport", "-1", 
            "--error-log", os.devnull, "--no-step-log", "--mesosim", "true"
        ]
        
        self.truck_ids = ["XeRac_AI_1", "XeRac_AI_2", "XeRac_AI_3"]
        self.zone_names = {"XeRac_AI_1": "Cụm Tây", "XeRac_AI_2": "Cụm Trung Tâm", "XeRac_AI_3": "Cụm Đông"}
        self.color_map = {"XeRac_AI_1": (255, 100, 0), "XeRac_AI_2": (50, 150, 255), "XeRac_AI_3": (0, 255, 0)}

        # =====================================================================
        # 🌟 SỔ ĐỎ QUY HOẠCH PHÂN TUYẾN 54 ĐIỂM CỐ ĐỊNH
        # =====================================================================
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
        self.bin_collected = {} 
        self.generators = {} 
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

    def assign_urgent_target(self, truck_id):
        depot = self.depot_edges.get(truck_id, "946030657")
        try: current_edge = traci.vehicle.getRoadID(truck_id)
        except: current_edge = depot
        if not current_edge or current_edge.startswith(":"): current_edge = depot

        for b in list(self.blacklist[truck_id].keys()):
            self.blacklist[truck_id][b] -= 1
            if self.blacklist[truck_id][b] <= 0: del self.blacklist[truck_id][b]

        load_percent = self.current_load[truck_id] / self.MAX_CAPACITY_KG
        
        if load_percent >= 0.90:
            try:
                r = traci.simulation.findRoute(current_edge, depot, vType="garbage_truck")
                if r and len(r.edges) > 0:
                    traci.vehicle.setRoute(truck_id, r.edges)
                    self.target_bins[truck_id] = depot 
                    return True
            except: pass

        curr_x, curr_y = self.edge_centers.get(current_edge, (0.0, 0.0))
        active_targets = [tgt for tid, tgt in self.target_bins.items() if tid != truck_id and tgt != depot]
        
        def get_best_optimal_route(candidates_list, penalty_factor=1.0):
            dists = []
            for b in candidates_list:
                tx, ty = self.edge_centers.get(b, (0.0, 0.0))
                dists.append((b, np.hypot(curr_x - tx, curr_y - ty)))
            dists.sort(key=lambda x: x[1])
            
            best_bin, best_route, best_score = None, None, -float('inf')
            # Quét rộng ra 15 thùng gần nhất để không bị Tunnel Vision
            for b, d in dists[:15]:
                try:
                    r1 = traci.simulation.findRoute(current_edge, b, vType="garbage_truck")
                    r2 = traci.simulation.findRoute(b, depot, vType="garbage_truck")
                    if r1 and r2 and len(r1.edges) > 0 and len(r2.edges) > 0:
                        real_travel_time = r1.travelTime if r1.travelTime > 0 else 1.0
                        current_lvl = self.bin_levels.get(b, 0)
                        fill_rate = self.generators[b].get_fill_rate(self.current_step)
                        predicted_lvl = current_lvl + (fill_rate * real_travel_time)
                        
                        if predicted_lvl >= 95.0:
                            score = 999999.0 / (real_travel_time + 1.0)
                        else:
                            score = predicted_lvl / ((real_travel_time * penalty_factor) + 10.0)
                            
                        if score > best_score:
                            best_score = score
                            best_bin = b
                            best_route = list(r1.edges) + list(r2.edges)[1:]
                except: continue
            return best_bin, best_route

        # LỚP 1: TÌM RÁC KHU VỰC NHÀ (CHƯA THU GOM)
        my_pending_bins = [b for b in self.zone_bins[truck_id] if not self.bin_collected.get(b, False) and b not in active_targets and b not in self.blacklist[truck_id]]
        best_b, best_r = get_best_optimal_route(my_pending_bins, penalty_factor=1.0)
        
        if best_b and best_r:
            traci.vehicle.setRoute(truck_id, best_r)
            self.target_bins[truck_id] = best_b
            return True
            
        # LỚP 2: CỨU TRỢ HÀNG XÓM
        if len(my_pending_bins) == 0:
            help_bins = [b for b, is_collected in self.bin_collected.items() if not is_collected and b not in self.zone_bins[truck_id] and b not in active_targets and b not in self.blacklist[truck_id]]
            if help_bins:
                best_b, best_r = get_best_optimal_route(help_bins, penalty_factor=1.8) 
                if best_b and best_r:
                    traci.vehicle.setRoute(truck_id, best_r)
                    self.target_bins[truck_id] = best_b
                    return True

        # LỚP 3: NẾU TOÀN THÀNH PHỐ ĐÃ THU GOM XONG 100% HOẶC CÒN RÁC TRONG BỤNG
        if self.current_load[truck_id] > 0 or all(self.bin_collected.values()):
            try:
                r = traci.simulation.findRoute(current_edge, depot, vType="garbage_truck")
                if r and len(r.edges) > 0:
                    traci.vehicle.setRoute(truck_id, r.edges)
                    self.target_bins[truck_id] = depot
                    return True
            except: pass

        # LỚP 4: TUẦN TRA QUANH PHƯỜNG ĐỂ ĐỢI ĐƯỜNG THÔNG
        if not all(self.bin_collected.values()):
            try:
                patrol_pool = [b for b in self.zone_bins[truck_id] if b != current_edge]
                random.shuffle(patrol_pool)
                for random_patrol in patrol_pool:
                    r1 = traci.simulation.findRoute(current_edge, random_patrol, vType="garbage_truck")
                    r2 = traci.simulation.findRoute(random_patrol, depot, vType="garbage_truck")
                    if r1 and r2 and len(r1.edges) > 0 and len(r2.edges) > 0:
                        full_r = list(r1.edges) + list(r2.edges)[1:]
                        traci.vehicle.setRoute(truck_id, full_r)
                        self.target_bins[truck_id] = "WANDERING"
                        return True
            except: pass

        # NẾU CẢ TUẦN TRA CŨNG KHÔNG TÌM ĐƯỢC ĐƯỜNG, ĐI TẠM VỀ TRẠM
        try:
            r = traci.simulation.findRoute(current_edge, depot, vType="garbage_truck")
            if r and len(r.edges) > 0:
                traci.vehicle.setRoute(truck_id, r.edges)
                self.target_bins[truck_id] = depot
                return True
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
        self.bin_collected = {} 
        self.generators = {}
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
                    r_test1 = traci.simulation.findRoute(depot, b, vType="garbage_truck")
                    r_test2 = traci.simulation.findRoute(b, depot, vType="garbage_truck")
                    if r_test1 and r_test2 and len(r_test1.edges) > 0 and len(r_test2.edges) > 0:
                        is_reachable = True
                except: pass
                
                if is_reachable:
                    self.zone_bins[truck_id].append(b)
                    self.bin_levels[b] = random.uniform(20.0, 80.0) 
                    self.bin_collected[b] = False
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
                                r1 = traci.simulation.findRoute(depot, valid_e, vType="garbage_truck")
                                r2 = traci.simulation.findRoute(valid_e, depot, vType="garbage_truck")
                                if r1 and r2 and len(r1.edges) > 0 and len(r2.edges) > 0:
                                    min_dist = d
                                    best_fallback = valid_e
                            except: pass
                    
                    if best_fallback:
                        self.zone_bins[truck_id].append(best_fallback)
                        self.bin_levels[best_fallback] = max(random.uniform(20.0, 80.0), self.bin_levels.get(best_fallback, 0.0))
                        self.bin_collected[best_fallback] = False

        # =====================================================================
        # 🌟 VÁ LỖI CỐT LÕI: TÍCH HỢP BỘ NÃO LSTM VÀO LƯỚI KHỞI TẠO RÁC
        # =====================================================================
        self.street_map.clear() 
        self.blacklist = {t: {} for t in self.truck_ids}
        self.target_bins = {t: "" for t in self.truck_ids} 
        self.episode_completed = False

        print(f"🧠 [HỆ THỐNG ĐẠI NÃO] LSTM đang phân tích chuỗi thời gian để nội suy % rác cho Ngày thứ {self.day_of_week}...")

        for b in self.bin_levels.keys():
            zone_type = "commercial" if random.random() > 0.5 else "residential"
            self.generators[b] = RealWasteGenerator(zone_type=zone_type)
            
            # --- 🌟 TIẾN TRÌNH LSTM DỰ BÁO ---
            # Tạo 1 chuỗi giả lập độ dài seq_length (12) quá khứ để mồi cho LSTM
            seq_data = []
            for d in range(self.seq_length):
                past_day = (self.day_of_week - self.seq_length + d) % 7
                past_we = 1 if past_day >= 5 else 0
                val = 0.2 if past_we else 0.8 if zone_type == "commercial" else 0.9 if past_we else 0.4
                seq_data.append([val])
                
            input_tensor = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred_norm = self.lstm_model(input_tensor).item()
                # Cắt xén (Clip) không cho rác âm hoặc quá 100%
                real_percent = max(0.0, min(100.0, pred_norm * 100.0))
            
            # Ghi đè hàm random cũ bằng chỉ số thực tế do LSTM tính toán
            self.bin_levels[b] = real_percent
            # --------------------------------
            
            self.history_buffer[b] = deque([self.bin_levels[b]]*self.seq_length, maxlen=self.seq_length)

        total_trash_kg = sum((lvl / 100.0) * self.BIN_MAX_WEIGHT_KG for lvl in self.bin_levels.values())
        print(f"📦 [CHỐT HỒ SƠ QUY HOẠCH] Tổng lượng rác khởi điểm: {total_trash_kg:.1f} kg.")

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
        print(f"🚗 Khởi tạo giao thông ({initial_traffic} xe nền)...")
        
        for i in range(initial_traffic):
            v_type = random.choice(["passenger_car", "passenger_car"])
            if self.route_cache_car:
                vid = f"xe_dan_init_{i}_{random.randint(100, 9999)}"
                try:
                    depart_time = str(random.randint(0, 15))
                    traci.route.add(f"route_{vid}", random.choice(self.route_cache_car))
                    traci.vehicle.add(vid, f"route_{vid}", typeID=v_type, depart=depart_time)
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

        # 🌟 VÁ LỖI XUẤT PHÁT: Cho 3 xe xuất phát cách nhau 10 giây để không giẫm lên nhau chết chùm tại Trạm!
        spawn_delay = 0
        for truck_id in self.truck_ids:
            try:
                depot_edge = self.depot_edges[truck_id]
                route_id = f"route_init_{truck_id}"
                traci.route.add(route_id, [depot_edge])
                traci.vehicle.add(truck_id, route_id, typeID="garbage_truck", depart=str(spawn_delay))
                traci.vehicle.setColor(truck_id, self.color_map[truck_id])
                spawn_delay += 10
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

        if self.current_step >= 20000:
            if not self.episode_completed:
                self.episode_completed = True
                print(f"\n⏰ [{time_str}] HẾT THỜI GIAN ĐỒ ÁN (Đạt mốc 20,000 nhịp). KẾT THÚC EPISODE!")
            for tid in self.truck_ids:
                self.is_done[tid] = True
                terminated[tid] = True
            return next_states, rewards, terminated, {t: False for t in self.truck_ids}, {}

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
                        depart_time = str(int(traci.simulation.getTime()) + random.randint(1, 15))
                        traci.route.add(f"route_{vid}", random.choice(self.route_cache_car))
                        traci.vehicle.add(vid, f"route_{vid}", typeID=v_type, depart=depart_time)
                    except: pass

        dt = 1.0
        connection_closed = False
        active_bins = {b: lvl for b, lvl in self.bin_levels.items() if not self.bin_collected.get(b, False)}

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
                            
                            if self.working_time[t_id] <= 0:
                                try: traci.vehicle.setSpeed(t_id, -1.0)
                                except: pass
                                self.target_bins[t_id] = ""
                                self.assign_urgent_target(t_id)
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
                            load_pct = self.current_load[t_id] / self.MAX_CAPACITY_KG
                            if load_pct >= 0.90 or all(self.bin_collected.values()):
                                if self.current_load[t_id] > 0:
                                    print(f"⏰ [{time_str}] ♻️ [{t_id}] Đã về Trạm. Bắt đầu xả {self.current_load[t_id]:.1f}kg rác (Mất 3 phút)...")
                                    self.current_load[t_id] = 0.0
                                    self.working_time[t_id] = 180.0 
                                    self.target_bins[t_id] = "WORKING"
                            else:
                                self.target_bins[t_id] = ""
                                self.assign_urgent_target(t_id)
                        
                        if self.target_bins.get(t_id) != "WORKING" and self.target_bins.get(t_id) != depot_e:
                            for b_e in list(active_bins.keys()):
                                if self.current_load[t_id] >= self.MAX_CAPACITY_KG: break
                                
                                is_near = False
                                # 🌟 TĂNG BÁN KÍNH RADAR LÊN 80M ĐỂ XE KHÔNG BAO GIỜ BỎ LỠ RÁC KHI PHÓNG QUA
                                if curr_e == b_e or curr_e == "-" + b_e or curr_e.replace("-", "") == b_e.replace("-", ""):
                                    bx, by = self.edge_centers[b_e]
                                    if np.hypot(vx-bx, vy-by) < 80.0: 
                                        is_near = True

                                if is_near:
                                    lvl = self.bin_levels[b_e]
                                    kg_in_bin = (lvl / 100.0) * self.BIN_MAX_WEIGHT_KG
                                    amt = min(kg_in_bin, self.MAX_CAPACITY_KG - self.current_load[t_id])
                                    self.current_load[t_id] += amt
                                    self.total_collected[t_id] += amt
                                    
                                    self.bin_collected[b_e] = True
                                    self.bin_levels[b_e] = 0.0
                                    del active_bins[b_e]
                                    
                                    if self.is_gui:
                                        try:
                                            traci.poi.setColor(f"BIN_{b_e}", (0, 255, 0, 255))
                                            traci.poi.setType(f"BIN_{b_e}", "Sạch ✅")
                                        except: pass
                                    
                                    print(f"⏰ [{time_str}] 🛑 [{t_id}] ĐỖ XE THU GOM {amt:.1f}kg tại {self.get_real_street_name(b_e)} (Mất 2 phút)...")
                                    self.working_time[t_id] = 120.0 
                                    self.target_bins[t_id] = "WORKING"
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
            try: curr_e = traci.vehicle.getRoadID(t)
            except: curr_e = ""

            if self.working_time[t] > 0:
                try: traci.vehicle.setSpeed(t, 0.0)
                except: pass
            else:
                # 🌟 TƯỚC QUYỀN ĐẠP PHANH BẬY CỦA AI NẾU ĐƯỜNG ĐANG VẮNG
                if act == 1: act = 2 
                try: 
                    if act == 0: traci.vehicle.setSpeed(t, 5.0) # Ép đi chậm 5m/s nếu AI rà phanh
                    else: traci.vehicle.setSpeed(t, -1.0) # Trả lại quyền tự động đi tối đa giới hạn đường cho SUMO
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

            if self.stuck_time[t] > 300:
                print(f"🚁 [{t}] Bị kẹt cứng vật lý. Gọi trực thăng cẩu về Trạm!")
                self.blacklist[t][curr_e] = 500
                self.stuck_time[t] = 0
                try: traci.vehicle.remove(t)
                except: pass

            if self.target_bins.get(t) == "": self.assign_urgent_target(t)
            
            next_states[t] = np.array([current_speed_kmh, avg_co2_per_km, float(self.stuck_time[t]), current_distance_km, (self.current_load[t]/self.MAX_CAPACITY_KG)*100.0, 100.0], dtype=np.float32)

        if all(self.bin_collected.values()):
            if not self.episode_completed:
                self.episode_completed = True
                print(f"\n🎉 [{time_str}] NHIỆM VỤ HOÀN THÀNH: Tất cả 54 điểm rác đô thị đã được thu gom xong!")
            
            for tid in self.truck_ids:
                if self.current_load[tid] <= 0 and self.target_bins.get(tid) != "WORKING":
                    self.is_done[tid] = True
                    terminated[tid] = True

        return next_states, {t: 0.0 for t in self.truck_ids}, terminated, {t: False for t in self.truck_ids}, {}

    def close(self):
        try: traci.close()
        except: pass