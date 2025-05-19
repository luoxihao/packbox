from dataclass import Pallet,Box
from tqdm import tqdm
import random
from utils import compute_metrics, VoxelCollisionChecker
from basepacker import Packer, EMSLBPacker
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
boxes = [
    Box(1200, 1000, 200),  # 整托箱，矮而宽
    Box(1200, 500, 250),   # 2×并排宽度铺满
    Box(1200, 250, 300),   # 4×并排宽度铺满
    Box(1200, 200, 250),   # 5×并排宽度铺满
    Box(1200, 125, 150),   # 8×并排宽度铺满
    Box(1200, 100, 300),   # 10×并排宽度铺满
    Box(1000, 600, 300),   # 2×首尾拼接铺满长度
    Box(1000, 400, 350),   # 3×拼接铺满长度
    Box(1000, 300, 250),   # 4×拼接铺满长度
    Box(1000, 200, 300),   # 6×拼接铺满长度
    Box(1000, 150, 250),   # 8×拼接铺满长度
    Box(1000, 100, 200),   # 12×拼接铺满长度
    Box(600, 600, 400),    # 1/4托盘面积箱，近似方形
    Box(600, 400, 300),    # 欧标600×400箱
    Box(800, 500, 250),    # 与500×400组合密拼
    Box(500, 400, 400),    # 与800×500组合匹配
    Box(800, 250, 300),    # 窄长箱，可与400×250错位
    Box(400, 250, 350),    # 小型矮箱，搭配800×250
    Box(900, 500, 250),    # 与300×500组合满铺
    Box(500, 300, 400),    # 与900×500组合或4×4密铺
    Box(600, 500, 350),    # 2×2正好铺满一层托盘
    Box(610, 377, 233),    # 黄金比例箱（费波纳奇）
    Box(300, 250, 600),    # 窄高箱，4×4满铺托盘面
    Box(400, 300, 250),    # 欧标1/8模块箱
    Box(250, 200, 500)     # 超小箱，6×4 或 5×5 满铺组合
]

import wandb


def Test_utils(boxes, algorithm_class, packer_class, rounds=1, low_box_num=40, high_box_num=60):
    algo_name = algorithm_class.__name__
    packer_name = packer_class.__name__ if packer_class else 'None'
    run_name = f"{algo_name}_{packer_name}_{low_box_num}_{high_box_num}"

    # 初始化wandb运行
    wandb.init(project="packing_project", name=run_name, reinit=True)
    wandb.config.update({
        "algorithm": algo_name,
        "packer": packer_name,
        "rounds": rounds,
        "low_box_num": low_box_num,
        "high_box_num": high_box_num,
    })

    utils = []
    used_utils = []
    un_packing_num = []

    for i in tqdm(range(rounds)):
        # boxes_used = random.choices(boxes, k=random.randint(low_box_num, high_box_num))
        boxes_used = boxes
        packer = algorithm_class(pallet, packer_class)
        placed_boxes, unplaced_boxes = packer.pack(boxes_used)

        VCC = VoxelCollisionChecker()
        VCC.add_cubes(placed_boxes)
        if VCC.check_collision():
            print("❌ 碰撞检测失败")
            continue

        util, used_util, height = compute_metrics(pallet, placed_boxes)
        utils.append(util)
        used_utils.append(used_util)
        un_packing_num.append(len(unplaced_boxes))

        # 每轮训练记录指标
        wandb.log({
            "round": i,
            "utilization": util,
            "used_utilization": used_util,
            "unplaced_boxes": len(unplaced_boxes)
        })
    visualize_pallet_open3d(pallet, placed_boxes)
    avg_util = sum(utils) / len(utils)
    avg_used_util = sum(used_utils) / len(used_utils)
    avg_unplaced = sum(un_packing_num) / len(un_packing_num)

    # 记录平均指标
    wandb.log({
        "avg_utilization": avg_util,
        "avg_used_utilization": avg_used_util,
        "avg_unplaced_boxes": avg_unplaced
    })
    print(f"✅ 平均利用率: {avg_util:.2f}")
    print(f"✅ 平均实际堆叠区域利用率: {avg_used_util:.2f}")
    print(f"✅ 平均未放置数量: {avg_unplaced:.2f}")

    wandb.finish()


if __name__ == '__main__':
    Test_utils(boxes, RandomPacker,Packer)
    Test_utils(boxes, RandomPacker,EMSLBPacker)
    
    Test_utils(boxes, GeneticPacker,Packer)
    Test_utils(boxes, GeneticPacker,EMSLBPacker)


    Test_utils(boxes, SimulatedAnnealingPacker,Packer)
    Test_utils(boxes, SimulatedAnnealingPacker,EMSLBPacker)

    Test_utils(boxes, GreedyPacker,Packer)
    Test_utils(boxes, GreedyPacker,EMSLBPacker)


    
    # Test_utils(boxes, SearchPacker,None,rounds=1000,low_box_num=5,high_box_num=10)

