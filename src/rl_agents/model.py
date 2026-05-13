import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Bộ não của Xe Rác AI (Deep Q-Network).
    Cấu trúc mạng Nơ-ron truyền thẳng (Feed-Forward Neural Network) 
    chuyên dùng để ước lượng giá trị của các hành động (Q-values).
    """
    def __init__(self, state_size, action_size, seed):
        """
        Khởi tạo kiến trúc mạng Nơ-ron.
        - state_size: Số lượng cảm biến đầu vào (Mặc định là 4: Vận tốc, g CO2/km, Rác A, Rác B)
        - action_size: Số lượng hành động đầu ra (Mặc định là 3: Rẽ Trái, Đi Thẳng, Rẽ Phải)
        """
        super(QNetwork, self).__init__()
        # Cố định seed để đảm bảo kết quả huấn luyện có thể tái lập (reproducible)
        self.seed = torch.manual_seed(seed)
        
        # 🧠 LỚP ẨN 1 (Hidden Layer 1): 
        # Nhận tín hiệu từ Môi trường (4 thông số) và phân tích thành 64 đặc trưng
        self.fc1 = nn.Linear(state_size, 64)
        
        # 🧠 LỚP ẨN 2 (Hidden Layer 2): 
        # Nhận 64 đặc trưng thô, suy luận sâu hơn để ra 64 đặc trưng cấp cao
        self.fc2 = nn.Linear(64, 64)
        
        # 🎯 LỚP ĐẦU RA (Output Layer): 
        # Tổng hợp thông tin và chốt lại thành các điểm số Q-value cho từng hành động
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """
        Quá trình lan truyền xuôi (Forward Propagation).
        Dữ liệu Môi trường chảy qua các "Tế bào nơ-ron" để biến thành Quyết định.
        """
        # Dùng hàm kích hoạt ReLU (Rectified Linear Unit) để khử tính tuyến tính.
        # Giúp AI học được các quy luật giao thông phức tạp (phi tuyến).
        x = F.relu(self.fc1(state)) 
        x = F.relu(self.fc2(x))
        
        # Lớp cuối cùng KHÔNG dùng ReLU vì điểm Reward (và Q-value) có thể là số âm (Bị phạt do xả nhiều CO2)
        return self.fc3(x)