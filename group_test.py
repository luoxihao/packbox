import os
import json
import numpy as np
from collections import Counter
from numpy import percentile
from scipy.stats import sem
from dataclass import Pallet, Box
from tqdm import tqdm
import random
from utils import compute_metrics, load_boxes_from_json, cluster_boxes
from basepacker import Packer
from sort import RandomPacker
import csv

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
    results = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            full_path = os.path.join(folder_path, file_name)
            boxes = load_boxes_from_json(full_path)
            clustered = cluster_boxes(boxes, n_clusters=1)  # 可选聚类

            for group in clustered:
                score = Test_utils(group, RandomPacker, Packer, rounds=rounds, is_random=is_random)

                length_width_values = []
                for box in group:
                    length_width_values.append(box.l)
                    length_width_values.append(box.w)
                length_width_count = Counter(length_width_values)
                length_width_unique_count = len(length_width_count)

                height_values = [box.h for box in group]
                height_count = Counter(height_values)
                height_unique_count = len(height_count)

                results.append({
                    "file_name": file_name,
                    "score": score,
                    "length_width_count": length_width_count,
                    "height_count": height_count,
                    "length_width_unique_count": length_width_unique_count,
                    "height_unique_count": height_unique_count
                })

    results.sort(key=lambda x: x["score"], reverse=True)

    if results:
        best = results[0]
        print("\n🏆 最佳文件:", best["file_name"], "利用率:", f"{best['score']:.4f}")
    else:
        print("目录内没有符合条件的json文件。")

    csv_path = os.path.join(folder_path, "batch_test_results.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_name', 'score', 'length_width_unique_count', 'height_unique_count',
                      'length_width_count', 'height_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for res in results:
            writer.writerow({
                'file_name': res['file_name'],
                'score': f"{res['score']:.4f}",
                'length_width_unique_count': res['length_width_unique_count'],
                'height_unique_count': res['height_unique_count'],
                'length_width_count': dict(res['length_width_count']),
                'height_count': dict(res['height_count']),
            })

    print(f"\n✅ 统计结果已保存至: {csv_path}")

    return (best["file_name"], best["score"]) if results else (None, None)



if __name__ == '__main__':
    folder = "./nb25_pop50_mut0.15_CR0.1_lwBL5s10_hBL3s40"  # 请替换为你的json目录
    best_file, best_score = batch_test_from_folder(folder, rounds=1000, is_random=False)
