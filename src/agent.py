import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork

# --- CÁC SIÊU THAM SỐ TỐI ƯU HÓA (V2.0) ---
BUFFER_SIZE = int(1e5)  # Sức chứa bộ nhớ: 100,000 hành động gần nhất
BATCH_SIZE = 128        # 🌟 Tăng từ 64 lên 128: Nhìn nhận toàn cảnh tốt hơn trong một lần cập nhật
GAMMA = 0.90            # 🌟 Giảm từ 0.99 xuống 0.90: Trọng tâm vào việc thoát kẹt NGAY LẬP TỨC thay vì lo xa
TAU = 1e-3              # Tốc độ sao chép kiến thức (Soft update)
LR = 5e-4               # Tốc độ tiếp thu (Learning Rate)
UPDATE_EVERY = 2        # 🌟 Tăng từ 4 lên 2: Tần suất học (Backpropagation) tăng gấp đôi!

# Tự động dùng Card đồ họa (GPU) nếu máy fen có, không thì dùng CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Tài xế AI (Agent) tương tác với môi trường và tự học hỏi."""
    
    # 🌟 CẬP NHẬT QUAN TRỌNG: Đã nâng state_size lên 5 để nhận diện % Bụng xe
    def __init__(self, state_size=5, action_size=3, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # 🧠 Q-Network (Bộ Não)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # 📖 Bộ nhớ Kinh nghiệm
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0 # Bộ đếm thời gian

    def step(self, state, action, reward, next_state, done):
        """Lưu trải nghiệm vào bộ nhớ và quyết định xem đã đến lúc học chưa."""
        # Cất trải nghiệm vào thẻ nhớ
        self.memory.add(state, action, reward, next_state, done)
        
        # Cứ đi được UPDATE_EVERY bước, và bộ nhớ có đủ số lượng thì bắt đầu học
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Ra quyết định hành động dựa trên Trạng thái hiện tại (Epsilon-Greedy)."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-Greedy: Đổ xúc xắc
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Cập nhật trọng số của Mạng Nơ-ron dựa trên các trải nghiệm quá khứ."""
        states, actions, rewards, next_states, dones = experiences

        # Lấy điểm Q-value cao nhất mà Mạng Target dự đoán cho bước tiếp theo
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Tính điểm Q-value mục tiêu (Theo công thức Bellman Equation)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Điểm Q-value thực tế mà Mạng Local đang dự đoán
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Tính sai số (Loss) giữa Dự đoán và Thực tế bằng hàm MSE
        loss = F.mse_loss(Q_expected, Q_targets)

        # Lan truyền ngược (Backpropagation) để sửa sai cho bộ não
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Cập nhật từ từ kiến thức từ Mạng Local sang Mạng Target
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Sao chép mềm (trộn) trọng số để hệ thống không bị sốc."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Cuốn sổ nhật ký lưu lại lịch sử di chuyển."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Thêm 1 trang nhật ký mới."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Rút ngẫu nhiên một xấp nhật ký ra để ôn bài."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)