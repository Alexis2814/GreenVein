import numpy as np

class RealWasteGenerator:
    def __init__(self, zone_type="residential", intensity=1.0):
        self.zone_type = zone_type
        self.intensity = intensity
        
    def get_fill_rate(self, step):
        # Giả sử 1 step = 1 giây, tính ra giờ trong ngày (0-23)
        hour = (step / 3600) % 24
        
        if self.zone_type == "residential":
            # Đỉnh lúc 7h sáng và 19h tối
            peak1 = np.exp(-0.5 * ((hour - 7)/1.5)**2)
            peak2 = np.exp(-0.5 * ((hour - 19)/2.0)**2)
            # GIẢM BASE RATE: Rác sinh ra rất từ từ (Mất ~12 tiếng để đầy)
            base_rate = (peak1 + peak2) * 0.0015 
        else:
            # Khu thương mại: Đỉnh lúc 12h và 20h, rác sinh ra ổn định ban ngày
            peak1 = np.exp(-0.5 * ((hour - 12)/3.0)**2)
            peak2 = np.exp(-0.5 * ((hour - 20)/2.0)**2)
            base_rate = (peak1 + peak2) * 0.0020 + 0.0005
            
        # Thêm một chút nhiễu ngẫu nhiên siêu nhỏ
        noise = np.random.normal(0, 0.0001)
        return max(0, (base_rate * self.intensity) + noise)

if __name__ == "__main__":
    gen = RealWasteGenerator("residential")
    print(f"Tốc độ sinh rác mỗi giây lúc 7h sáng: {gen.get_fill_rate(7*3600):.6f}%")