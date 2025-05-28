import numpy as np
from numpy import percentile
from scipy.stats import sem

from dataclass import Pallet,Box
from tqdm import tqdm
import random
from utils import compute_metrics, VoxelCollisionChecker,load_boxes_from_json  ,cluster_boxes
from basepacker import Packer, BinPacker
from sort import GreedyPacker, SearchPacker, GeneticPacker, SimulatedAnnealingPacker,RandomPacker
from visualize import  visualize_pallet ,visualize_pallet_open3d

pallet = Pallet(1600, 1000, 1800)

def Test_utils(boxes, algorithm_class, packer_class, rounds=100, low_box_num=20, high_box_num=20,is_random = False):
    algo_name = algorithm_class.__name__
    packer_name = packer_class.__name__ if packer_class else 'None'
    run_name = f"{algo_name}_{packer_name}_{low_box_num}_{high_box_num}"
    print(f"开始测试 {run_name}")
    utils = []
    used_utils = []
    un_packing_num = []

    for _ in tqdm(range(rounds)):
        if is_random:
            boxes_used = random.choices(boxes, k=random.randint(low_box_num, high_box_num))
        else:
            boxes_used = boxes
        random.shuffle(boxes_used)

        packer = algorithm_class(pallet, packer_class)
        placed_boxes, unplaced_boxes = packer.pack(boxes_used)

        # VCC = VoxelCollisionChecker()
        # VCC.add_cubes(placed_boxes)
        # if VCC.check_collision():
        #     print("❌ 碰撞检测失败")
        #     continue

        util, used_util, height = compute_metrics(pallet, placed_boxes)
        util = float(util)
        used_util = float(used_util)
        utils.append(util)
        used_utils.append(used_util)
        un_packing_num.append(len(unplaced_boxes))
        visualize_pallet_open3d(pallet, placed_boxes)
    utils_np = np.array(utils)
    used_utils_np = np.array(used_utils)

    avg_util = utils_np.mean()
    avg_used_util = used_utils_np.mean()
    avg_unplaced = sum(un_packing_num) / len(un_packing_num)

    conf_interval_util = 1.96 * sem(utils_np)
    conf_interval_used_util = 1.96 * sem(used_utils_np)
    print("-----------------------------------------------------------------------------------------------")
    print(f"✅ 平均利用率:               {avg_util:.4f}")
    print(f"✅ 平均实际堆叠区域利用率:     {avg_used_util:.4f}")
    print(f"✅ 平均未放置数量:             {avg_unplaced:.4f}")
    # print()
    # print(f"📊 利用率：最大值: {utils_np.max():.4f}  最小值: {utils_np.min():.4f}  标准差: {utils_np.std(ddof=1):.4f}")
    # print(
    #     f"📊 实际堆叠区域利用率：最大值: {used_utils_np.max():.4f}  最小值: {used_utils_np.min():.4f}  标准差: {used_utils_np.std(ddof=1):.4f}")
    # print()
    # print(f"🔸 利用率 95%置信区间:         {avg_util:.4f} ± {conf_interval_util:.4f}")
    # print(f"🔸 实际利用率 95%置信区间:      {avg_used_util:.4f} ± {conf_interval_used_util:.4f}")


if __name__ == '__main__':
    # "./numBoxes25_pop5000_mut0.15_bitLen6_minDim150_maxDim800_step100/55bestBoxes0.7013.json"
    #  "./numBoxes25_pop5000_mut0.15_bitLen6_minDim150_maxDim800_step100/61bestBoxes0.7130.json"
    boxes = load_boxes_from_json(
        "./2/13bestBoxes1.8742.json")
    rounds = 5
    is_random = False
    boxes = cluster_boxes(boxes, n_clusters=1)
    for used_boxes in boxes:
        Test_utils(used_boxes, RandomPacker,Packer,rounds=rounds,is_random=is_random)
        # Test_utils(used_boxes, RandomPacker,BinPacker,rounds=rounds,is_random=is_random)


    # Test_utils(boxes, RandomPacker,Packer,rounds=rounds)
    # Test_utils(boxes, RandomPacker,BinPacker,rounds=rounds)

    # Test_utils(boxes, GreedyPacker,Packer,rounds=rounds)
    # Test_utils(boxes, GreedyPacker,BinPacker,rounds = rounds)

    # Test_utils(boxes, GeneticPacker,Packer)
    # Test_utils(boxes, GeneticPacker,EMSLBPacker)


    # Test_utils(boxes, SimulatedAnnealingPacker,Packer)
    # Test_utils(boxes, SimulatedAnnealingPacker,EMSLBPacker)


    # Test_utils(boxes,GeneticPacker,BinPacker)

    # Test_utils(boxes, SimulatedAnnealingPacker,BinPacker)
    

