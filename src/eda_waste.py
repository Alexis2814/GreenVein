import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_waste_data():
    csv_path = "waste_history.csv"
    if not os.path.exists(csv_path):
        print("❌ Không tìm thấy file CSV!")
        return

    # 1. Đọc dữ liệu
    df = pd.read_csv(csv_path)
    print(f"📊 Đã nạp {len(df)} dòng dữ liệu từ file CSV.")

    # 2. Lọc dữ liệu 48 giờ đầu tiên (để nhìn cho rõ quy luật)
    # 48 giờ = 48 * 4 (mỗi 15p một bản ghi) = 192 bản ghi cho mỗi thùng
    unique_bins = df['Bin_ID'].unique()
    
    plt.figure(figsize=(15, 7))
    plt.style.use('ggplot')

    # Chọn 1 thùng Residential và 1 thùng Commercial tiêu biểu
    res_bin = df[df['Zone_Type'] == 'residential']['Bin_ID'].iloc[0]
    com_bin = df[df['Zone_Type'] == 'commercial']['Bin_ID'].iloc[0]

    for bin_id, label in [(res_bin, 'Khu Dân Cư'), (com_bin, 'Khu Thương Mại')]:
        bin_data = df[df['Bin_ID'] == bin_id].head(192) # Lấy 2 ngày đầu
        plt.plot(bin_data['Timestamp_Step'] / 3600, bin_data['Fill_Level_Percent'], 
                 label=f"{label} (ID: {bin_id})", linewidth=2)

    plt.title("PHÂN TÍCH CHU KỲ SINH RÁC TRONG 48 GIỜ ĐẦU TIÊN", fontsize=16, fontweight='bold')
    plt.xlabel("Thời gian (Giờ)", fontsize=12)
    plt.ylabel("Mức độ đầy rác (%)", fontsize=12)
    plt.axhline(100, color='red', linestyle='--', alpha=0.5, label='Ngưỡng tràn rác')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("🎨 Đang hiển thị biểu đồ...")
    plt.show()

if __name__ == "__main__":
    analyze_waste_data()