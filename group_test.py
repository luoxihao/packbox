import os
import json
import numpy as np
from collections import Counter
from tqdm import tqdm
import random
import csv
from dataclass import Pallet, Box
from utils import compute_metrics, load_boxes_from_json, cluster_boxes
from basepacker import Packer,BinPacker
from sort import RandomPacker

class BatchBoxTester:
    def __init__(self, pallet: Pallet, folder_path: str, rounds: int = 1000, is_random: bool = False):
        self.pallet = pallet
        self.folder_path = folder_path
        self.rounds = rounds
        self.is_random = is_random
        self.results = []

    def test_utils(self, boxes, algorithm_class, packer_class,rounds = None):
        utils = []
        used_utils = []
        un_packing_num = []
        if rounds is None:
            rounds = self.rounds

        for _ in tqdm(range(rounds)):
            if self.is_random:
                boxes_used = random.choices(boxes, k=random.randint(20, 20))
            else:
                boxes_used = boxes
            random.shuffle(boxes_used)

            packer = algorithm_class(self.pallet, packer_class)
            placed_boxes, unplaced_boxes = packer.pack(boxes_used)

            util, used_util, _ = compute_metrics(self.pallet, placed_boxes)
            utils.append(float(util))
            used_utils.append(float(used_util))
            un_packing_num.append(len(unplaced_boxes))

        avg_util = np.mean(used_utils)
        return avg_util

    def batch_test(self):
        for file_name in os.listdir(self.folder_path):
            if not file_name.endswith(".json") or file_name == "config.json":
                continue
            full_path = os.path.join(self.folder_path, file_name)
            boxes = load_boxes_from_json(full_path)
            clustered = cluster_boxes(boxes, n_clusters=1)  # 可选聚类

            for group in clustered:
                score = self.test_utils(group, RandomPacker, Packer)
                sota_score = self.test_utils(group, RandomPacker, BinPacker,rounds=10)
                length_width_values = []
                for box in group:
                    length_width_values.append(box.l)
                    length_width_values.append(box.w)
                length_width_count = Counter(length_width_values)
                length_width_unique_count = len(length_width_count)

                height_values = [box.h for box in group]
                height_count = Counter(height_values)
                height_unique_count = len(height_count)

                self.results.append({
                    "file_name": file_name,
                    "score": score,
                    "sota_score":sota_score,
                    "length_width_count": length_width_count,
                    "height_count": height_count,
                    "length_width_unique_count": length_width_unique_count,
                    "height_unique_count": height_unique_count
                })

        self.results.sort(key=lambda x: x["score"], reverse=True)

        if self.results:
            best = self.results[0]
            print("\n🏆 最佳文件:", best["file_name"], "利用率:", f"{best['score']:.4f}")
        else:
            print("目录内没有符合条件的json文件。")

        return self.results

    def save_to_csv(self, csv_filename="batch_test_results.csv"):
        if not self.results:
            print("无结果，无法保存CSV。")
            return

        csv_path = os.path.join(self.folder_path, csv_filename)
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file_name', 'score','sota_score', 'length_width_unique_count', 'height_unique_count',
                          'length_width_count', 'height_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for res in self.results:
                writer.writerow({
                    'file_name': res['file_name'],
                    'score': f"{res['score']:.4f}",
                    'sota_score': f"{res['sota_score']:.4f}",
                    'length_width_unique_count': res['length_width_unique_count'],
                    'height_unique_count': res['height_unique_count'],
                    'length_width_count': dict(res['length_width_count']),
                    'height_count': dict(res['height_count']),
                })

        print(f"\n✅ 统计结果已保存至: {csv_path}")



if __name__ == '__main__':
    pallet = Pallet(1600, 1000, 3000)
    folder = "./nb25_pop50_mut0.15_CR0.1_lwBL5s10_hBL3s40"  # 请替换为你的json目录

    tester = BatchBoxTester(pallet, folder, rounds=1000, is_random=False)
    results = tester.batch_test()
    tester.save_to_csv()
