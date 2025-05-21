import numpy as np
from numpy import percentile
from scipy.stats import sem

from dataclass import Pallet,Box
from tqdm import tqdm
import random
from utils import compute_metrics, VoxelCollisionChecker,load_boxes_from_json   
from basepacker import Packer, BinPacker
from sort import GreedyPacker, SearchPacker, GeneticPacker, SimulatedAnnealingPacker,RandomPacker
from visualize import  visualize_pallet ,visualize_pallet_open3d
pallet = Pallet(1200, 1000, 1800)

# boxes = [Box(750,450,80),Box(700,500,80),Box(650,550,80),Box(600,500,96),
#         Box(600,400,96),Box(550,450,96),Box(650,350,120),Box(667,333,120),
#         Box(500,400,120),Box(750,250,120),Box(600,300,160),Box(600,250,160),
#         Box(500,300,160),Box(500,250,160),Box(600,200,192),Box(400,300,192),
#         Box(400,250,192),Box(400,200,192),Box(500,150,240),Box(300,250,240),
#         Box(400,150,240),Box(300,200,240),Box(300,150,320),Box(200,150,320),
#         Box(200,100,320)]
# boxes = [
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

# import wandb


# import numpy as np  # 需导入

def Test_utils(boxes, algorithm_class, packer_class, rounds=100, low_box_num=15, high_box_num=15):
    algo_name = algorithm_class.__name__
    packer_name = packer_class.__name__ if packer_class else 'None'
    run_name = f"{algo_name}_{packer_name}_{low_box_num}_{high_box_num}"
    print(f"开始测试 {run_name}")
    utils = []
    used_utils = []
    un_packing_num = []

    for i in tqdm(range(rounds)):
        boxes_used = random.choices(boxes, k=random.randint(low_box_num, high_box_num))
        packer = algorithm_class(pallet, packer_class)
        placed_boxes, unplaced_boxes = packer.pack(boxes_used)

        VCC = VoxelCollisionChecker()
        VCC.add_cubes(placed_boxes)
        if VCC.check_collision():
            print("❌ 碰撞检测失败")
            continue

        util, used_util, height = compute_metrics(pallet, placed_boxes)
        util = float(util)
        used_util = float(used_util)
        utils.append(util)
        used_utils.append(used_util)
        un_packing_num.append(len(unplaced_boxes))
        # visualize_pallet_open3d(pallet, placed_boxes)
    utils_np = np.array(utils)
    used_utils_np = np.array(used_utils)

    avg_util = utils_np.mean()
    avg_used_util = used_utils_np.mean()
    avg_unplaced = sum(un_packing_num) / len(un_packing_num)

    print(f"\n✅ 平均利用率: {avg_util:.2f}")
    print(f"✅ 平均实际堆叠区域利用率: {avg_used_util:.2f}")
    print(f"✅ 平均未放置数量: {avg_unplaced:.2f}")

    print(f"\n📊 利用率：最大值: {utils_np.max():.2f}, 最小值: {utils_np.min():.2f}, 标准差: {utils_np.std(ddof=1):.2f}")
    print(f"📊 实际堆叠区域利用率：最大值: {used_utils_np.max():.2f}, 最小值: {used_utils_np.min():.2f}, 标准差: {used_utils_np.std(ddof=1):.2f}")
    print("\n📈 补充统计信息：")

    # 中位数
    print(f"🔸 利用率中位数: {np.median(utils_np):.2f}")
    print(f"🔸 实际堆叠区域利用率中位数: {np.median(used_utils_np):.2f}")

    # 四分位间距
    iqr_util = np.percentile(utils_np, 75) - np.percentile(utils_np, 25)
    iqr_used_util = np.percentile(used_utils_np, 75) - np.percentile(used_utils_np, 25)
    print(f"🔸 利用率 IQR (Q3-Q1): {iqr_util:.2f}")
    print(f"🔸 实际利用率 IQR (Q3-Q1): {iqr_used_util:.2f}")

    # 置信区间
    conf_interval_util = 1.96 * sem(utils_np)
    conf_interval_used_util = 1.96 * sem(used_utils_np)
    print(f"🔸 利用率 95%置信区间: {avg_util:.2f} ± {conf_interval_util:.2f}")
    print(f"🔸 实际利用率 95%置信区间: {avg_used_util:.2f} ± {conf_interval_used_util:.2f}")

    # 10%、90%分位数
    p10, p90 = percentile(utils_np, [10, 90])
    print(f"🔸 利用率分布 P10: {p10:.2f}, P90: {p90:.2f}")


if __name__ == '__main__':
    boxes = load_boxes_from_json("./binery6_best_individual/best_boxes0.8669.json")
    rounds = 100
    Test_utils(boxes, RandomPacker,Packer,rounds=rounds)
    Test_utils(boxes, RandomPacker,BinPacker,rounds=rounds)

    Test_utils(boxes, GreedyPacker,Packer,rounds=rounds)
    Test_utils(boxes, GreedyPacker,BinPacker,rounds = rounds)

    # Test_utils(boxes, GeneticPacker,Packer)
    # Test_utils(boxes, GeneticPacker,EMSLBPacker)


    # Test_utils(boxes, SimulatedAnnealingPacker,Packer)
    # Test_utils(boxes, SimulatedAnnealingPacker,EMSLBPacker)


    # Test_utils(boxes,GeneticPacker,BinPacker)

    # Test_utils(boxes, SimulatedAnnealingPacker,BinPacker)
    

