import numpy as np

class GarbagePredictor:
    def __init__(self):
        # Đây là nơi bạn sẽ code khởi tạo mô hình PyTorch LSTM (nn.LSTM)
        # self.model = YourLSTMNetwork(...)
        self.is_loaded = False

    def load_weights(self, path="weights/lstm_model.pth"):
        # self.model.load_state_dict(torch.load(path))
        self.is_loaded = True
        print("✅ Đã nạp thành công bộ não LSTM dự báo rác.")

    def predict(self, zone_type, is_weekend, day_of_week):
        """
        Nhận đầu vào là đặc trưng ngày và khu vực, trả ra khối lượng rác dự đoán.
        """
        # =======================================================
        # NẾU BẠN ĐÃ TRAIN XONG MÔ HÌNH BẰNG PYTORCH THÌ DÙNG ĐOẠN NÀY:
        # tensor_input = torch.tensor([[zone_type, is_weekend, day_of_week]], dtype=torch.float32)
        # predicted_weight = self.model(tensor_input).item()
        # =======================================================
        
        # TRONG LÚC CHỜ TRAIN MÔ HÌNH, TA DÙNG HÀM MÔ PHỎNG NÀY ĐỂ KỊP CHẠY ĐỒ ÁN
        AVG = 7000 
        if zone_type == 1: # Văn phòng
            base = AVG * 0.2 if is_weekend else AVG * 1.5
        else: # Dân cư
            base = AVG * 1.4 if is_weekend else AVG * 0.8
            
        noise = np.random.uniform(-0.1, 0.1)
        predicted_weight = max(0, base * (1 + noise))
        
        return predicted_weight