import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

MAX_BOXES = 5
FEATURE_DIM = 3
HEIGHTMAP_SHAPE = (10, 10)

class BoxEnvWithDelayedReward:
    def __init__(self, get_candidates_fn, reward_fn,get_heightmap_fn):
        self.get_candidates_fn = get_candidates_fn
        self.reward_fn = reward_fn
        self.get_heightmap_fn=get_heightmap_fn
        self.heightmap = None
        self.available_boxes = []
        self.selected_sequence = []

    def reset(self):
        self.heightmap = self.get_heightmap_fn()
        self.selected_sequence = []
        self.available_boxes = self.get_candidates_fn()
        return self._get_state()

    def _get_state(self):
        pad = [np.zeros(FEATURE_DIM) for _ in range(MAX_BOXES - len(self.available_boxes))]
        box_tensor = np.stack(self.available_boxes + pad)
        mask = np.array([1 if i < len(self.available_boxes) else 0 for i in range(MAX_BOXES)])
        return box_tensor, self.heightmap.copy(), mask

    def step(self, action_idx):
        done = False
        box = self.available_boxes[action_idx]
        #吧选择的box给下位机器
        self.selected_sequence.append(box)
        if len(self.get_candidates_fn())==0:
            done = True
        return self._get_state() if not done else (None, None, None), done

    def compute_total_reward(self):
        return self.reward_fn(self.selected_sequence)

class JointPolicy(nn.Module):
    def __init__(self, heightmap_shape=(10, 10), box_dim=3, hidden_dim=64):
        super().__init__()
        h, w = heightmap_shape
        self.height_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h * w, hidden_dim),
            nn.ReLU()
        )
        self.box_encoder = nn.Linear(box_dim, hidden_dim)
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, boxes, heightmap, mask):
        batch_size = boxes.shape[0]
        height_feat = self.height_encoder(heightmap).unsqueeze(1).repeat(1, MAX_BOXES, 1)
        box_feat = self.box_encoder(boxes)
        combined = torch.cat([box_feat, height_feat], dim=-1)
        scores = self.selector(combined).squeeze(-1)
        scores[~mask] = -1e9
        probs = torch.softmax(scores, dim=-1)
        return probs

def get_random_boxes():
    n = random.randint(2, MAX_BOXES)
    return [np.random.rand(3) for _ in range(n)]

def total_volume_reward(sequence):
    return sum(np.prod(b) for b in sequence)  # 示例：总 volume 作为奖励

# 初始化
env = BoxEnvWithDelayedReward(get_candidates_fn=get_random_boxes, reward_fn=total_volume_reward)
policy = JointPolicy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

# 训练循环
for epoch in range(300):
    boxes_np, heightmap_np, mask_np = env.reset()
    log_probs = []
    done = False

    while not done:
        boxes = torch.tensor(boxes_np, dtype=torch.float32).unsqueeze(0)
        heightmap = torch.tensor(heightmap_np, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask_np, dtype=torch.bool).unsqueeze(0)

        probs = policy(boxes, heightmap, mask)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_probs.append(m.log_prob(action))

        (boxes_np, heightmap_np, mask_np), done = env.step(action.item())

    reward = env.compute_total_reward()
    loss = -sum(log_probs) * reward  # 延迟奖励策略梯度

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"[Epoch {epoch}] total_reward = {reward:.3f}, num_steps = {len(log_probs)}")