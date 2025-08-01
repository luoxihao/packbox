# {
#     "num": 2,
#     "id":[1, 2],
#     "lwh":[(2,2,2),(2,2,2)],
#     "camera_left_down":[(0,0,0),(0,0,0)], # 相机坐标系下每个货箱左下角坐标
#     "robot_left_down":[(0,0,0),(0,0,0)], # base 坐标系下每个货箱左下角坐标
# }

import random
from typing import List, Tuple, Dict
from dataclass import Box
import csv
class DataGenerator:
    def __init__(self):
        self.loaded_boxes=[]
    @staticmethod
    def generate_box_data(num_boxes: int) -> Dict:
        """
        生成指定数量的箱子数据，包括尺寸、相机坐标和机器人坐标
        """
        box_data = {
            "num": num_boxes,
            "id": [],
            "lwh": [],
            "camera_left_down": [],
            "robot_left_down": []
        }

        for i in range(1, num_boxes + 1):
            box_data["id"].append(i)

            # 固定尺寸或使用随机尺寸
            # l, w, h = 2, 2, 2
            l = round(random.uniform(1.0, 3.0), 2)
            w = round(random.uniform(1.0, 3.0), 2)
            h = round(random.uniform(1.0, 3.0), 2)
            box_data["lwh"].append((l, w, h))

            # 相机坐标系下的左下角
            cx = round(random.uniform(0.0, 5.0), 2)
            cy = round(random.uniform(0.0, 5.0), 2)
            cz = 0.0
            box_data["camera_left_down"].append((cx, cy, cz))

            # 机器人 base 坐标系下的左下角
            rx = round(random.uniform(0.0, 10.0), 2)
            ry = round(random.uniform(0.0, 10.0), 2)
            rz = 0.0
            box_data["robot_left_down"].append((rx, ry, rz))

        return box_data
    def load_boxes_from_csv(self, csv_path: str) -> List[Box]:
        boxes = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                box = Box(
                    l=float(row['l']), w=float(row['w']), h=float(row['h']),
                    x=float(row['x']), y=float(row['y']), z=float(row['z']),
                    box_id=int(row['uid'])
                )
                boxes.append(box)
        return boxes
    def remove_loade_boxes(self, uid):
        self.loaded_boxes = [box for box in self.loaded_boxes if box.id != uid]
# 示例调用
if __name__ == "__main__":
    generator = DataGenerator()
    data = generator.generate_box_data(num_boxes=2)
    print(data)