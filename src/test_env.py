from core.environment import GreenVeinEnv

def test_environment():
    print("Khởi tạo môi trường GreenVeinEnv (Có tích hợp LSTM)...")
    env = GreenVeinEnv()
    
    print("\nĐang Reset môi trường...")
    states, _ = env.reset()
    
    print("\nCho xe chạy thử 5 bước để test AI dự báo...")
    for step in range(5):
        # Cho xe chạy thẳng (action = 1)
        actions = {truck_id: 1 for truck_id in env.truck_ids}
        next_states, rewards, terminated, _, _ = env.step(actions)
        print(f"Bước {step + 1}: Chạy mượt mà, không có lỗi!")
        
    env.close()
    print("\n✅ Môi trường hoạt động hoàn hảo!")

if __name__ == "__main__":
    test_environment()