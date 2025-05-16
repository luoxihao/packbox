# single_file_pallet_stacking.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import matplotlib.colors as mcolors
import itertools
from tqdm import tqdm
import math
import copy
# random.seed(0)
# ------------------------------
# Box 类
# ------------------------------
class Box:
    def __init__(self, l, w, h):
        self.l = l
        self.w = w
        self.h = h

    def orientations(self):
        return [(self.l, self.w, self.h), (self.w, self.l, self.h)]

    def key(self):
        return (self.l, self.w, self.h)


# ------------------------------
# Pallet 类
# ------------------------------
class Pallet:
    def __init__(self, l, w, h):
        self.l = l
        self.w = w
        self.h = h
        self.boxes = []


# ------------------------------
# Packer 码垛算法
# ------------------------------
def does_collide(x, y, z, bl, bw, bh, placed):
    for px, py, pz, pl, pw, ph, _ in placed:
        if (
            x < px + pl and x + bl > px and
            y < py + pw and y + bw > py and
            z < pz + ph and z + bh > pz
        ):
            return True  # 有重叠
    return False  # 无重叠

class Packer:
    def __init__(self, pallet):
        self.pallet = pallet

    def pack(self, boxes):
        placed = []
        unplaced = []
        free_spaces = [(0, 0, 0)]

        for box in boxes:
            placed_flag = False
            for orientation in box.orientations():
                bl, bw, bh = orientation
                for i, (x, y, z) in enumerate(free_spaces):
                    if (x + bl <= self.pallet.l and y + bw <= self.pallet.w and z + bh <= self.pallet.h and
                        not does_collide(x, y, z, bl, bw, bh, placed)):
                        placed.append((x, y, z, bl, bw, bh, box.key()))
                        self.pallet.boxes.append((x, y, z, bl, bw, bh, box.key()))
                        free_spaces.append((x + bl, y, z))
                        free_spaces.append((x, y + bw, z))
                        free_spaces.append((x, y, z + bh))
                        del free_spaces[i]
                        placed_flag = True
                        break
                if placed_flag:
                    break
            if not placed_flag:
                unplaced.append(box)

        return placed, unplaced

class GreedyPacker:
    def __init__(self, pallet):
        self.pallet = pallet
        self.packer  = Packer(pallet)
    def pack(self, boxes):
        boxes = sorted(boxes, key=lambda b: b.l * b.w * b.h, reverse=True)
        return self.packer.pack(boxes)


class SearchPacker:
    def __init__(self, pallet):
        self.pallet = pallet

    def pack(self, boxes):
        best_solution = []
        best_score = 0
        best_unplaced = []

        all_permutations = list(itertools.permutations(boxes))

        outer = tqdm(all_permutations, desc="全排列搜索", unit="perm", position=0)
        for perm in outer:
            placed = []
            free_spaces = [(0, 0, 0)]

            for box in perm:
                placed_flag = False
                for orientation in box.orientations():
                    bl, bw, bh = orientation
                    for i, (x, y, z) in enumerate(free_spaces):
                        if (x + bl <= self.pallet.l and y + bw <= self.pallet.w and z + bh <= self.pallet.h and
                            not does_collide(x, y, z, bl, bw, bh, placed)):
                            placed.append((x, y, z, bl, bw, bh, box.key()))
                            free_spaces.append((x + bl, y, z))
                            free_spaces.append((x, y + bw, z))
                            free_spaces.append((x, y, z + bh))
                            del free_spaces[i]
                            placed_flag = True
                            break
                    if placed_flag:
                        break
                if not placed_flag:
                    break  # 本排列失败，提前退出

            # 综合评估：利用率得分
            if placed:
                box_volume = sum(l * w * h for _, _, _, l, w, h, _ in placed)
                max_height = max((z + h) for _, _, z, _, _, h, _ in placed)
                pallet_base = self.pallet.l * self.pallet.w
                total_volume = self.pallet.l * self.pallet.w * self.pallet.h
                used_volume = pallet_base * max_height
                util_overall = box_volume / total_volume
                util_used = box_volume / used_volume
                score = 0.5 * util_overall + 0.5 * util_used
            else:
                score = 0

            if score > best_score:
                best_score = score
                best_solution = placed
                best_unplaced = [b for b in boxes if b.key() not in [p[6] for p in placed]]

        outer.close()

        self.pallet.boxes = best_solution
        return best_solution, best_unplaced



class GeneticPacker:
    def __init__(self, pallet, population_size=30, generations=50, mutation_rate=0.2):
        self.pallet = pallet
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def pack(self, boxes):
        def fitness(order):
            pallet_tmp = Pallet(self.pallet.l, self.pallet.w, self.pallet.h)
            packer = Packer(pallet_tmp)
            placed, _ = packer.pack(order)
            if not placed:
                return 0, placed

            box_volume = sum(l * w * h for _, _, _, l, w, h, _ in placed)
            max_height = max((z + h) for _, _, z, _, _, h, _ in placed)
            total_volume = pallet_tmp.l * pallet_tmp.w * pallet_tmp.h
            used_volume = pallet_tmp.l * pallet_tmp.w * max_height

            util_overall = box_volume / total_volume
            util_used = box_volume / used_volume

            # 综合评分（可调节权重）
            score = 0.5 * util_overall + 0.5 * util_used
            return score, placed
        

        def crossover(p1, p2):
            cut = random.randint(1, len(p1) - 2)
            child = p1[:cut] + [b for b in p2 if b not in p1[:cut]]
            return child

        def mutate(order):
            a, b = random.sample(range(len(order)), 2)
            order[a], order[b] = order[b], order[a]

        # 初始化种群，每个个体是 Box 实例的排列
        population = [random.sample(boxes, len(boxes)) for _ in range(self.population_size)]
        best_solution = []
        best_score = 0
        best_placed = []

        for _ in range(self.generations):
            scored = []
            for ind in population:
                vol, placed = fitness(ind)
                scored.append((vol, placed, ind))
            scored.sort(key=lambda x: x[0], reverse=True)

            if scored[0][0] > best_score:
                best_score = scored[0][0]
                best_placed = scored[0][1]
                best_solution = scored[0][2]

            parents = [ind for _, _, ind in scored[:self.population_size // 2]]
            next_gen = []
            while len(next_gen) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                child = crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    mutate(child)
                next_gen.append(child)
            population = next_gen

        # 最优个体重新打包
        final_pallet = Pallet(self.pallet.l, self.pallet.w, self.pallet.h)
        final_packer = Packer(final_pallet)
        placed, unplaced = final_packer.pack(best_solution)
        self.pallet.boxes = placed
        return placed, unplaced

class SimulatedAnnealingPacker:
    def __init__(self, pallet, initial_temp=1000, final_temp=1, alpha=0.95, max_iter=500):
        self.pallet = pallet
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha  # 温度衰减因子
        self.max_iter = max_iter

    def pack(self, boxes):
        def fitness(order):
            pallet_tmp = Pallet(self.pallet.l, self.pallet.w, self.pallet.h)
            packer = Packer(pallet_tmp)
            placed, _ = packer.pack(order)
            if not placed:
                return 0, placed

            box_volume = sum(l * w * h for _, _, _, l, w, h, _ in placed)
            max_height = max((z + h) for _, _, z, _, _, h, _ in placed)
            total_volume = pallet_tmp.l * pallet_tmp.w * pallet_tmp.h
            used_volume = pallet_tmp.l * pallet_tmp.w * max_height

            util_overall = box_volume / total_volume
            util_used = box_volume / used_volume

            # 综合评分（可调节权重）
            score = 0.5 * util_overall + 0.5 * util_used
            return score, placed

        def swap(box_list):
            new_list = box_list.copy()
            a, b = random.sample(range(len(new_list)), 2)
            new_list[a], new_list[b] = new_list[b], new_list[a]
            return new_list

        current_order = boxes[:]
        current_score, current_placed = fitness(current_order)
        best_order = current_order[:]
        best_score = current_score

        T = self.initial_temp
        iter_count = 0

        while T > self.final_temp and iter_count < self.max_iter:
            new_order = swap(current_order)
            new_score, _ = fitness(new_order)

            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_order = new_order
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_order = new_order

            T *= self.alpha
            iter_count += 1

        final_pallet = Pallet(self.pallet.l, self.pallet.w, self.pallet.h)
        final_packer = Packer(final_pallet)
        placed, unplaced = final_packer.pack(best_order)
        self.pallet.boxes = placed
        return placed, unplaced
# ------------------------------
# 可视化函数
# ------------------------------
def visualize_pallet(pallet, boxes, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique_keys = list({key for *_, key in boxes})
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    random.shuffle(colors)
    color_map = {key: colors[i % len(colors)] for i, key in enumerate(unique_keys)}

    for box in boxes:
        x, y, z, l, w, h, key = box
        draw_cube(ax, x, y, z, l, w, h, color_map[key])

    ax.set_xlim(0, pallet.l)
    ax.set_ylim(0, pallet.w)
    ax.set_zlim(0, pallet.h)
    ax.set_xlabel('Length')
    ax.set_ylabel('Width')
    ax.set_zlabel('Height')
    plt.tight_layout()
    plt.show()
    #plt.savefig(save_path)
    plt.close()


def draw_cube(ax, x, y, z, l, w, h, color):
    corners = [
        [x, y, z], [x + l, y, z], [x + l, y + w, z], [x, y + w, z],
        [x, y, z + h], [x + l, y, z + h], [x + l, y + w, z + h], [x, y + w, z + h]
    ]
    faces = [[corners[j] for j in [0,1,2,3]],
             [corners[j] for j in [4,5,6,7]],
             [corners[j] for j in [0,1,5,4]],
             [corners[j] for j in [2,3,7,6]],
             [corners[j] for j in [1,2,6,5]],
             [corners[j] for j in [0,3,7,4]]]
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.6, facecolors=color))


# ------------------------------
# 指标函数
# ------------------------------
def compute_metrics(pallet, boxes):
    total_volume = pallet.l * pallet.w * pallet.h
    box_volume = sum(l * w * h for _, _, _, l, w, h, _ in boxes)
    max_height = max((z + h) for _, _, z, _, _, h, _ in boxes) if boxes else 0
    used_volume = max_height * pallet.l * pallet.w
    return box_volume / total_volume, box_volume / used_volume, max_height


# ------------------------------
# 主函数入口
# ------------------------------
def test_utils(boxes, algothim,rounds=1000,low_box_num=40,high_box_num=80):
    utils=[]
    used_utils=[]
    for _ in tqdm(range(rounds)):
        boxes_used = random.choices(boxes, k=random.randint(low_box_num, high_box_num))

        packer = algothim(pallet)
        placed_boxes, unplaced_boxes = packer.pack(boxes_used)

        util, used_util, height = compute_metrics(pallet, placed_boxes)
        utils.append(util)
        used_utils.append(used_util)
    print(f"平均利用率: {sum(utils) / len(utils):.2f}")
    print(f"平均实际堆叠区域利用率: {sum(used_utils) / len(used_utils):.2f}")

if __name__ == '__main__':
    pallet = Pallet(1200, 1000, 1800)

    boxes = [Box(750,450,80),Box(700,500,80),Box(650,550,80),Box(600,500,96),
            Box(600,400,96),Box(550,450,96),Box(650,350,120),Box(667,333,120),
            Box(500,400,120),Box(750,250,120),Box(600,300,160),Box(600,250,160),
            Box(500,300,160),Box(500,250,160),Box(600,200,192),Box(400,300,192),
            Box(400,250,192),Box(400,200,192),Box(500,150,240),Box(300,250,240),
            Box(400,150,240),Box(300,200,240),Box(300,150,320),Box(200,150,320),
            Box(200,100,320)]
#     boxes = [
#     Box(1200, 1000, 200),  # 整托箱，矮而宽
#     Box(1200, 500, 250),   # 2×并排宽度铺满
#     Box(1200, 250, 300),   # 4×并排宽度铺满
#     Box(1200, 200, 250),   # 5×并排宽度铺满
#     Box(1200, 125, 150),   # 8×并排宽度铺满
#     Box(1200, 100, 300),   # 10×并排宽度铺满
#     Box(1000, 600, 300),   # 2×首尾拼接铺满长度
#     Box(1000, 400, 350),   # 3×拼接铺满长度
#     Box(1000, 300, 250),   # 4×拼接铺满长度
#     Box(1000, 200, 300),   # 6×拼接铺满长度
#     Box(1000, 150, 250),   # 8×拼接铺满长度
#     Box(1000, 100, 200),   # 12×拼接铺满长度
#     Box(600, 600, 400),    # 1/4托盘面积箱，近似方形
#     Box(600, 400, 300),    # 欧标600×400箱
#     Box(800, 500, 250),    # 与500×400组合密拼
#     Box(500, 400, 400),    # 与800×500组合匹配
#     Box(800, 250, 300),    # 窄长箱，可与400×250错位
#     Box(400, 250, 350),    # 小型矮箱，搭配800×250
#     Box(900, 500, 250),    # 与300×500组合满铺
#     Box(500, 300, 400),    # 与900×500组合或4×4密铺
#     Box(600, 500, 350),    # 2×2正好铺满一层托盘
#     Box(610, 377, 233),    # 黄金比例箱（费波纳奇）
#     Box(300, 250, 600),    # 窄高箱，4×4满铺托盘面
#     Box(400, 300, 250),    # 欧标1/8模块箱
#     Box(250, 200, 500)     # 超小箱，6×4 或 5×5 满铺组合
# ]

    print("Testing GeneticPacker...")
    test_utils(boxes, GeneticPacker) #1 （0.58，0.68）
    print("Testing GreedyPacker...")
    test_utils(boxes, GreedyPacker)#1 （0.58，0.62）
    print("Testing SimulatedAnnealingPacker...")
    test_utils(boxes, SimulatedAnnealingPacker) #1 （0.59，0.69）
    print("Testing Packer...")
    test_utils(boxes, Packer)#1 （0.57，0.62）
    print("Testing SearchPacker...")
    test_utils(boxes, SearchPacker,low_box_num=5,high_box_num=10) #1 （）

    # boxes_used = random.choices(boxes, k=random.randint(5, 10))

    # packer = SearchPacker(pallet)
    # placed_boxes, unplaced_boxes = packer.pack(boxes_used)

    # util, used_util, height = compute_metrics(pallet, placed_boxes)
    # print(f"使用箱子的数量：{len(boxes_used)}")
    # print(f"成功放置数量：{len(placed_boxes)}，未放置数量：{len(unplaced_boxes)}")
    # print(f"托盘总体利用率: {util:.2f}, 空隙率: {1-util:.2f}")
    # print(f"实际堆叠区域利用率: {used_util:.2f}, 空隙率: {1-used_util:.2f}, 最大堆叠高度: {height:.1f}mm")
    # visualize_pallet(pallet, placed_boxes, 'result.png')

    
    
    
    
    
    
    # visualize_pallet(pallet, placed_boxes, 'result.png')
