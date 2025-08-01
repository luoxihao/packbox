import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from DataGenerator import DataGenerator
from dataclass import Box,Pallet
from suction_planner import SuctionPlanner
from socket_communicate_template import ClientCommunicator
# ---------------- 参数设置 ----------------
MAX_BOXES = 5
FEATURE_DIM = 3
HEIGHTMAP_SHAPE = (16, 10)
EPOCHS = 1  # 可根据需要改大
HIDDEN_DIM = 64
PRINT_EVERY = 10
MAX_STEPS=10
# ---------------- 环境定义 ----------------
class BoxEnvWithDelayedReward:
    def __init__(self, get_candidates_fn, reward_fn, get_heightmap_fn, max_steps=None):
        self.get_candidates_fn = get_candidates_fn
        self.reward_fn = reward_fn
        self.get_heightmap_fn = get_heightmap_fn
        self.max_steps = max_steps
        self.step_counter = 0
        self.heightmap = None
        self.available_boxes = []
        self.available_boxes_np = None
        self.suctions = []
        self.selected_sequence = []
        self.client = ClientCommunicator()
        self.client.connect_to_server()

    def reset(self):
        self.heightmap = self.get_heightmap_fn()
        self.selected_sequence = []

        self.client.wait_for_request_from_server()
        self.available_boxes,self.suctions = self.get_candidates_fn()
        self.available_boxes_np = [np.array([box.l, box.w, box.h]) for box in self.available_boxes]
        if len(self.available_boxes)>5:
            self.available_boxes = random.sample(self.available_boxes,5)
        self.step_counter = 0
        return self._get_state()

    def _get_state(self):
        pad = [np.zeros(FEATURE_DIM) for _ in range(MAX_BOXES - len(self.available_boxes))]
        box_tensor = np.stack(self.available_boxes_np + pad)
        mask = np.array([1 if i < len(self.available_boxes_np) else 0 for i in range(MAX_BOXES)])
        return box_tensor, self.heightmap.copy(), mask

    def step(self, action_idx):
        self.step_counter += 1
        box = self.available_boxes[action_idx]
        suction = self.suctions[action_idx]

        suction_coordinate,rotation,t_s = SuctionPlanner.suction_sem2coordinate(suction, box)
        #发出选择的箱子的位置和姿态
        self.client.reply_depallet_calculation(box_size=[box.w,box.l,box.h] if rotation else [box.l,box.w,box.h],
                                            start_coord=list(suction_coordinate),
                                            start_euler_z=[-90 if rotation else 0,180,0],
                                            position_offset=list(t_s))
        self.selected_sequence.append(box)
        #更新环境状态
        self.client.wait_for_request_from_server()
        self.available_boxes ,self.suctions= self.get_candidates_fn()
        # 没有获取到可以拿放的箱子
        if self.available_boxes is None:
            done = True
            return (None, None, None),done

        if len(self.available_boxes)>5:
            self.available_boxes = random.sample(self.available_boxes,5)
        self.heightmap = self.get_heightmap_fn()
        done = len(self.available_boxes) == 0

        #测试时使用 如果到maxstep就停止
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done = True

        return self._get_state() if not done else (None, None, None), done

    def compute_total_reward(self):
        return self.reward_fn(self.selected_sequence)
    def index2box(self, box_idx):
        return self.available_boxes[box_idx]

# ---------------- 策略网络 ----------------
class AdvancedJointPolicy(nn.Module):
    def __init__(self, heightmap_shape=(16, 10), box_dim=3, hidden_dim=64):
        super().__init__()
        self.height_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.box_embed = nn.Sequential(
            nn.LayerNorm(box_dim),
            nn.Linear(box_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.box_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, boxes, heightmap, mask):
        B = boxes.size(0)  # batch size
        # Heightmap encoding: [B, 1, H, W] → [B, hidden]
        heightmap = heightmap.unsqueeze(1)
        height_feat = self.height_encoder(heightmap)  # [B, hidden]

        # Box sequence encoding: [B, MAX_BOXES, 3] → [B, MAX_BOXES, hidden]
        box_feat = self.box_embed(boxes)
        box_encoded = self.box_transformer(box_feat)

        # Add height info to each box token
        height_feat_expanded = height_feat.unsqueeze(1).expand(-1, MAX_BOXES, -1)
        fused = box_encoded + height_feat_expanded

        # Scoring
        scores = self.scorer(fused).squeeze(-1)  # [B, MAX_BOXES]
        scores[~mask] = -1e9  # mask invalid boxes
        probs = torch.softmax(scores, dim=-1)
        return probs

# ---------------- 工具函数 ----------------
def get_random_boxes():
    n = random.randint(2, MAX_BOXES)
    return [np.random.rand(3) for _ in range(n)]

def get_heightmap():
    return np.random.rand(*HEIGHTMAP_SHAPE)

def total_volume_reward(sequence):
    return sum(np.prod(b) for b in sequence)
''
def box_transform(boxes_dict):
    """
    从相机得到的箱子数据转换为本地箱子数据格式
    """
    boxes=[]
    for i in range(boxes_dict['num']):
        l,w,h = boxes_dict['lwh'][i]
        x,y,z = boxes_dict['robot_left_down'][i]
        id = boxes_dict['id'][i]
        box = Box(l,w,h,x,y,z,id)
        boxes.append(box)
    return boxes

def randomNum_get_candidates():
    boxes = box_transform(DataGenerator.generate_box_data(random.randint(1, 5)))

    pallet = Pallet(1600, 1000, 1800)
    suction_template = Box(800, 600, 1)
    planner = SuctionPlanner(pallet, suction_template)

    accessibles, uids = planner.find_accessible_boxes(boxes)
    suctions = []
    targets = []
    for accessible in accessibles:
        suction = planner.find_suction_position(accessible, boxes)
        if suction:
            suctions.append(suction)
            targets.append(accessible)
    if len(suctions) ==0:
        return None,None

    return targets,suctions
# --------------函数形式--仅获取箱子和返回所有可能的吸盘位置
def get_all_suctions(boxes):
    pallet = Pallet(1600, 1000, 1800)
    suction_template = Box(800, 600, 1)
    planner = SuctionPlanner(pallet, suction_template)

    accessibles, uids = planner.find_accessible_boxes(boxes)
    all_suctions = []
    targets = []
    for accessible in accessibles:
        suctions = planner.find_suction_position_all(accessible, boxes)
        if suctions:
            all_suctions.extend(suctions)
            targets.extend([accessible for _ in range(len(suctions))])
    #
    # for i in range(len(all_suctions)):
    #     if targets[i].w>targets[i].l:
    #         targets[i].l,targets[i].w = targets[i].w,targets[i].l
    return targets,all_suctions

# ------------- 选最大的主函数-------------
def max_box_main():
    env = BoxEnvWithDelayedReward(
        get_candidates_fn=randomNum_get_candidates,
        reward_fn=total_volume_reward,
        get_heightmap_fn=get_heightmap,
        max_steps=MAX_STEPS
    )
    result = env.reset()
    boxes_np, heightmap_np, mask_np = result
    done = False
    while not done:
        volumes = np.prod(boxes_np, axis=1)

        # 找出最大体积对应的索引
        max_idx = np.argmax(volumes)
        action = max_idx
        result, done = env.step(action)

        if not done:
            boxes_np, heightmap_np, mask_np = result
# ---------------- 主函数 ----------------
def main():
    env = BoxEnvWithDelayedReward(
        # get_candidates_fn=get_random_boxes,
        get_candidates_fn=randomNum_get_candidates,
        reward_fn=total_volume_reward,
        get_heightmap_fn=get_heightmap,
        max_steps=MAX_STEPS
    )

    policy = AdvancedJointPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        result = env.reset()
        if result is None:
            continue
        boxes_np, heightmap_np, mask_np = result
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

            result, done = env.step(action.item())
            if result is not None:
                boxes_np, heightmap_np, mask_np = result

        reward = env.compute_total_reward()
        loss = -sum(log_probs) * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % PRINT_EVERY == 0:
            print(f"[Epoch {epoch}] Total reward = {reward:.3f}, Steps = {len(log_probs)}")
# ---------------- 测试函数 ----------------
def test_get_all_suctions():
    pallet = Pallet(1600, 1000, 1800)
    suction_template = Box(600, 800, 1)
    planner = SuctionPlanner(pallet, suction_template)
    boxes = planner.load_boxes_from_csv("./packed_boxes14.csv")
    while len(boxes)!=0:
        targets, all_suctions = get_all_suctions(boxes)
        for i in range(len(targets)):
            #这个是要获取的
            suction_xyz,rotation_or_not,box_offset= planner.suction_sem2coordinate(all_suctions[i],targets[i])
        if len(targets) == 0:
            print("noooooooooooooooo")
            break
        #我自己随机删除一个
        remove_id = targets[0].id
        boxes = [box for box in boxes if box.id != remove_id]
if __name__ == "__main__":
    # max_box_main()
    test_get_all_suctions()