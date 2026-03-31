from environment import GreenVeinEnv
import time

def test_multi_agent():
    print("Khởi tạo môi trường GreenVeinEnv (Multi-Agent MADQN)...")
    env = GreenVeinEnv()
    
    # Hàm reset bây giờ sẽ tự động in ra Điểm Xuất Phát
    states, info = env.reset()
    
    for step in range(5):
        print(f"\n{'='*20} BƯỚC {step + 1} {'='*20}")
        
        actions = {truck_id: env.action_space.sample() for truck_id in env.truck_ids}
        next_states, rewards, terminated, truncated, info = env.step(actions)
        
        print("--- Tổng kết Điểm Reward ---")
        for truck_id in env.truck_ids:
            # 🌟 NÂNG CẤP: Làm tròn điểm số xuống 2 chữ số thập phân giúp log siêu gọn
            print(f"   💰 {truck_id} nhận được: {rewards[truck_id]:.2f} điểm")
            
        time.sleep(1.5) 

    env.close()
    print("\nHoàn thành bài test Multi-Agent!")

if __name__ == "__main__":
    test_multi_agent()