import os
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config

def generate_mixed_traffic():
    net_path = os.path.join(Config.BASE_DIR, 'envs', 'map.net.xml')
    cars_path = os.path.join(Config.BASE_DIR, 'envs', 'cars.rou.xml')
    motos_path = os.path.join(Config.BASE_DIR, 'envs', 'motos.rou.xml')
    
    randomTrips_script = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
    
    print("🚗 Đang tạo lộ trình Ô TÔ...")
    cmd_cars = [
        sys.executable, randomTrips_script, 
        "-n", net_path, 
        "-r", cars_path, 
        "-e", "3600", 
        "-p", "1.0", 
        "--vehicle-class", "passenger", 
        "--allow-fringe", 
        "--random",
        "--prefix", "car_"  # Dòng quan trọng: Đánh dấu ô tô là car_0, car_1...
    ]
    subprocess.run(cmd_cars, check=True)

    print("🏍️ Đang tạo lộ trình XE MÁY...")
    cmd_motos = [
        sys.executable, randomTrips_script, 
        "-n", net_path, 
        "-r", motos_path, 
        "-e", "3600", 
        "-p", "0.3", 
        "--vehicle-class", "motorcycle", 
        "--allow-fringe", 
        "--random",
        "--prefix", "moto_"  # Dòng quan trọng: Đánh dấu xe máy là moto_0, moto_1...
    ]
    subprocess.run(cmd_motos, check=True)
    
    print("✅ Đã tạo xong giao thông hỗn hợp!")

if __name__ == "__main__":
    generate_mixed_traffic()