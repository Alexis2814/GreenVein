import os

class Config:
    # === 1. THÔNG SỐ HỆ THỐNG MÔ PHỎNG (SUMO) ===
    GUI_MODE = True              # Bật/Tắt giao diện 3D của SUMO (True = Mở cửa sổ giả lập)
    STEP_LENGTH = 1.0            # Thời gian của 1 bước mô phỏng (giây)
    
    # === 2. THÔNG SỐ ĐẠI LÝ (AGENTS - XE RÁC) ===
    NUM_AGENTS = 3               # Tổng số xe rác hoạt động (Tác tử)
    VEHICLE_CAPACITY = 1000      # Tải trọng tối đa của 1 xe rác (kg)
    
    # === 3. THÔNG SỐ MÔI TRƯỜNG & BẢN ĐỒ ===
    NUM_BINS = 50                # Tổng số điểm thu gom (thùng rác) trên bản đồ
    DEPOT_NODE = "depot_001"     # ID của bãi tập kết rác (Sẽ map với file XML của SUMO sau)
    MAX_BIN_CAPACITY = 100       # Dung lượng tối đa của mỗi thùng rác (kg)
    
    # === 4. SIÊU THAM SỐ TRÍ TUỆ NHÂN TẠO (MADQN) ===
    LEARNING_RATE = 0.001        # Tốc độ học (Alpha) của mạng nơ-ron
    GAMMA = 0.99                 # Hệ số chiết khấu (Discount factor) cho phần thưởng tương lai
    EPSILON_START = 1.0          # Tỷ lệ khám phá ngẫu nhiên ban đầu (100% - Xe chạy thử nghiệm)
    EPSILON_END = 0.05           # Tỷ lệ khám phá tối thiểu (5% - Khi AI đã khôn)
    EPSILON_DECAY = 0.995        # Tốc độ giảm dần của sự ngẫu nhiên sau mỗi Episode
    BATCH_SIZE = 64              # Kích thước gói dữ liệu lấy từ Replay Buffer
    MEMORY_SIZE = 10000          # Kích thước bộ nhớ kinh nghiệm của AI
    
    # === 5. QUẢN LÝ ĐƯỜNG DẪN (PATHS) ===
    # Tự động nhận diện thư mục gốc của dự án để tránh lỗi đường dẫn khi chạy ở máy khác
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SUMO_NET_FILE = os.path.join(BASE_DIR, 'envs', 'map.net.xml')
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'src', 'models')

# Đoạn test nhanh khi chạy trực tiếp file này
if __name__ == "__main__":
    print("✅ Đã tải thành công cấu hình hệ thống GreenVein!")
    print(f"Thư mục gốc của dự án: {Config.BASE_DIR}")
    print(f"Số lượng xe rác hoạt động: {Config.NUM_AGENTS} xe")