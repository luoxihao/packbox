import csv

import numpy as np

from dataclass import Pallet,Box
from visualize import visualize_pallet_open3d
from typing import List, Tuple
class SuctionPlanner:
    def __init__(self, pallet: Pallet, suction_template: Box):
        self.pallet = pallet
        self.suction_template = suction_template

    def load_boxes_from_csv(self, csv_path: str,resize=False) -> List[Box]:
        factor = 1 if not resize else 100
        boxes = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                box = Box(
                    l=float(row['l'])*factor, w=float(row['w'])*factor, h=float(row['h'])*factor,
                    x=float(row['x'])*factor, y=float(row['y'])*factor, z=float(row['z'])*factor,
                    box_id=int(row['uid'])
                )
                boxes.append(box)
        return boxes

    def find_accessible_boxes(self, boxes: List[Box]) -> Tuple[List[Box], List[int]]:
        accessible_boxes = []
        accessible_uids = []
        for box in boxes:
            if not any(box.is_covered_by(other) for other in boxes if other.id != box.id):
                accessible_boxes.append(box)
                accessible_uids.append(box.id)
        return accessible_boxes, accessible_uids

    def check_collision(self, candidate_box: Box, boxes: List[Box]) -> bool:
        eps = 1e-9  # 数值稳定性(可按需要调大/调小)

        # 候选吸盘 AABB（x,y,z 为最小角）
        ori = getattr(candidate_box, 'orientation', 0) % 360
        # cl, cw = (candidate_box.l, candidate_box.w) if ori in (0, 180) else (candidate_box.w, candidate_box.l)
        cl,cw = (candidate_box.l, candidate_box.w)
        cx_min, cy_min, cz_min = candidate_box.x, candidate_box.y, candidate_box.z
        cx_max, cy_max, cz_max = cx_min + cl, cy_min + cw, cz_min + candidate_box.h

        for box in boxes:
            if box is candidate_box:
                continue

            borient = getattr(box, 'orientation', 0) % 360
            bl, bw = (box.l, box.w) if borient in (0, 180) else (box.w, box.l)
            bx_min, by_min, bz_min = box.x, box.y, box.z
            bx_max, by_max, bz_max = bx_min + bl, by_min + bw, bz_min + box.h

            # 分离轴“不重叠”条件（任一轴分离即不相交；贴边：<= / >= 视为不相交）
            separated = (
                    cx_max <= bx_min + eps or cx_min >= bx_max - eps or
                    cy_max <= by_min + eps or cy_min >= by_max - eps or
                    cz_max <= bz_min + eps or cz_min >= bz_max - eps
            )
            if not separated:
                return True  # 存在重叠 -> 碰撞

        return False  # 无任何重叠 -> 不碰撞（贴边也算不碰撞）

    def has_touching_box_underneath(self, suction_box: Box, boxes: List[Box]) -> bool:
        for box in boxes:
            if box.z_top() == suction_box.z:
                if suction_box.xy_overlap(box):
                    return True
        return False
    #找到匹配的就返回
    def find_suction_position(self, target_box: Box, all_boxes: List[Box]) -> Box:
        others = [b for b in all_boxes if b is not target_box]
        target_top_z = target_box.z_top()
        target_cx, target_cy = target_box.center()

        for ori in [0, 90]:
            sl, sw = (self.suction_template.l, self.suction_template.w) if ori % 180 == 0 else (self.suction_template.w, self.suction_template.l)
            tl, tw = (target_box.l, target_box.w)
            if sl < tl and sw < tw:
                suction = self.suction_template.copy()
                suction.l, suction.w = sl, sw
                suction.set_center(target_cx, target_cy)
                suction.z = target_top_z
                suction.orientation = ori
                if not self.check_collision(suction, others):
                    return suction

        corner_signs = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        for ori in [0, 90]:
            sl, sw = (self.suction_template.l, self.suction_template.w) if ori % 180 == 0 else (self.suction_template.w, self.suction_template.l)
            tl, tw = (target_box.l, target_box.w)
            eps = 1e-6
            # 如果目标箱子是正方形，直接通过
            if abs(tl - tw) <= eps:
                pass
            # 如果长短边方向匹配才通过
            elif not ((sl > sw and tl > tw) or (sw > sl and tw > tl)):
                print("uid", target_box.id, "at rotation", ori, "not long-to-long short-to-short")
                continue

            for sx, sy in corner_signs:
                suction = self.suction_template.copy()
                suction.id = target_box.id
                suction.l, suction.w = sl, sw
                corner_x = target_box.x + (sx + 1) * tl / 2
                corner_y = target_box.y + (sy + 1) * tw / 2
                suction.x = corner_x - (sx + 1) * sl / 2
                suction.y = corner_y - (sy + 1) * sw / 2
                suction.z = target_top_z
                suction.orientation = ori

                # visualize_boxes = [(b.x, b.y, b.z, b.l, b.w, b.h, b.id) for b in all_boxes]
                # visualize_suctions = [(suction.x, suction.y, suction.z, suction.l, suction.w, suction.h, suction.id)]
                # visualize_pallet_open3d(self.pallet, visualize_boxes, accessible_boxes_uid=[target_box.id],
                #                         suctions=visualize_suctions)

                if self.has_touching_box_underneath(suction, others):
                    print("uid",target_box.id,"in ",sx,sy,"has_touching_box_underneath")
                    continue
                if not self.check_collision(suction, others):
                    return suction
                print("uid", target_box.id, "in ", sx, sy, "has_collision")


        return None
    def find_suction_position_all(self, target_box: Box, all_boxes: List[Box]) -> List[Box]:
        others = [b for b in all_boxes if b is not target_box]
        target_top_z = target_box.z_top()
        target_cx, target_cy = target_box.center()
        suctions= []
        for ori in [0, 90]:
            sl, sw = (self.suction_template.l, self.suction_template.w) if ori % 180 == 0 else (self.suction_template.w, self.suction_template.l)
            tl, tw = (target_box.l, target_box.w)
            if sl < tl and sw < tw:
                suction = self.suction_template.copy()
                suction.l, suction.w = sl, sw
                suction.set_center(target_cx, target_cy)
                suction.z = target_top_z
                suction.orientation = ori
                if not self.check_collision(suction, others):
                    return [suction]

        corner_signs = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        for ori in [0, 90]:
            sl, sw = (self.suction_template.l, self.suction_template.w) if ori % 180 == 0 else (self.suction_template.w, self.suction_template.l)
            tl, tw = (target_box.l, target_box.w)

            eps = 1e-6
            # 如果目标箱子是正方形，直接通过
            if abs(tl - tw) <= eps:
                pass
            # 如果长短边方向匹配才通过
            elif not ((sl > sw and tl > tw) or (sw > sl and tw > tl)):
                print("uid", target_box.id, "at rotation", ori, "not long-to-long short-to-short")
                continue
            for sx, sy in corner_signs:
                suction = self.suction_template.copy()
                suction.id = target_box.id
                suction.l, suction.w = sl, sw
                corner_x = target_box.x + (sx + 1) * tl / 2
                corner_y = target_box.y + (sy + 1) * tw / 2
                suction.x = corner_x - (sx + 1) * sl / 2
                suction.y = corner_y - (sy + 1) * sw / 2
                suction.z = target_top_z
                suction.orientation = ori
                suction.corner=(corner_x,corner_y)
                if self.has_touching_box_underneath(suction, others):
                    print("uid",target_box.id,"in ",sx,sy,"has_touching_box_underneath")
                    continue
                if not self.check_collision(suction, others):
                    suctions.append(suction)
                print("uid", target_box.id, "in ", sx, sy, "has_collision")


        return suctions
    def get_accessible_target(self,boxes):
        accessibles, uids = self.find_accessible_boxes(boxes)
        suctions = []
        targets = []
        for accessible in accessibles:
            suction = self.find_suction_position(accessible, boxes)
            if suction:
                suctions.append(suction)
                targets.append(accessible)
        return suctions,targets
    def run_demo(self, csv_path: str):
        boxes = self.load_boxes_from_csv(csv_path,True)
        while boxes:
            accessibles, uids = self.find_accessible_boxes(boxes)
            suctions = []
            targets = []
            for accessible in accessibles:
                suction = self.find_suction_position(accessible, boxes)
                if suction:
                    suctions.append(suction)
                    targets.append(accessible)

            visualize_boxes = [(b.x, b.y, b.z, b.l, b.w, b.h, b.id) for b in boxes]
            visualize_suctions = [(s.x, s.y, s.z, s.l, s.w, s.h, s.id) for s in suctions]
            visualize_pallet_open3d(self.pallet, visualize_boxes, accessible_boxes_uid=uids, suctions=visualize_suctions)

            if not accessibles:
                print("❌ 没有可拿出的箱子，结束。")
                break

            delete_uid = targets[0].id
            print(f"🗑️ 删除箱子 UID: {delete_uid}")
            boxes = [box for box in boxes if box.id != delete_uid]
    @staticmethod
    def suction_sem2coordinate(suction: Box, target_box: Box):
        # 计算吸盘的质心坐标（x, y 是底面中心，z 为底面高度）
        x = suction.x + suction.l / 2.0
        y = suction.y + suction.w / 2.0
        z = suction.z

        # 吸盘质心作为坐标系原点
        s_o = np.array([x, y, z])

        # 计算目标箱子的质心坐标
        tx = target_box.x + target_box.l / 2.0
        ty = target_box.y + target_box.w / 2.0
        tz = target_box.z + target_box.h / 2.0
        t_o = np.array([tx, ty, tz])

        # 计算目标箱子质心在吸盘坐标系下的相对位置（平移向量）
        t_s = t_o - s_o
        if suction.orientation == 90:
            R_xy_cw = np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ])
            t_s = R_xy_cw @ t_s
        # 返回值：
        # 1. 吸盘的质心坐标 (x, y, z)
        # 2. 吸盘是否旋转 90 度（用于后续姿态控制）
        # 3. 目标箱子质心相对于吸盘坐标系的平移向量（元组格式）
        return (x, y, z), suction.orientation == 90, tuple(t_s)


if __name__ == "__main__":
    pallet = Pallet(1600, 1000, 1800)
    suction_template = Box(700, 800, 1)
    planner = SuctionPlanner(pallet, suction_template)
    planner.run_demo("./packed_boxes26.csv")

