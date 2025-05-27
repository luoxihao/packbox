import os
import json
import numpy as np
from numpy import percentile
from scipy.stats import sem
from dataclass import Pallet, Box
from tqdm import tqdm
import random
from utils import compute_metrics, load_boxes_from_json, cluster_boxes
from basepacker import Packer
from sort import RandomPacker

pallet = Pallet(1600, 1000, 3000)


def Test_utils(boxes, algorithm_class, packer_class, rounds=100, low_box_num=20, high_box_num=20, is_random=False):
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

        util, used_util, _ = compute_metrics(pallet, placed_boxes)
        utils.append(float(util))
        used_utils.append(float(used_util))
        un_packing_num.append(len(unplaced_boxes))

    avg_util = np.mean(used_utils)
    return avg_util


def batch_test_from_folder(folder_path, rounds=1000, is_random=False):
    best_score = -1
    best_file = None

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            full_path = os.path.join(folder_path, file_name)
            boxes = load_boxes_from_json(full_path)
            clustered = cluster_boxes(boxes, n_clusters=1)  # 可选聚类

            for group in clustered:
                score = Test_utils(group, RandomPacker, Packer, rounds=rounds, is_random=is_random)
                print(f"\n📦 {file_name} 的平均利用率为: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_file = file_name

    print("\n🏆 最佳文件:", best_file, "利用率:", f"{best_score:.4f}")
    return best_file, best_score


if __name__ == '__main__':
    folder = "./numBoxes25_pop5000_mut0.15_bitLen6_minDim150_maxDim800_step100"  # 请替换为你的json目录
    best_file, best_score = batch_test_from_folder(folder, rounds=1000, is_random=False)
