import csv
from dataclass import Pallet,Box
from visualize import visualize_pallet_open3d
from typing import List, Tuple
class SuctionPlanner:
    def __init__(self, pallet: Pallet, suction_template: Box):
        self.pallet = pallet
        self.suction_template = suction_template

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

    def find_accessible_boxes(self, boxes: List[Box]) -> Tuple[List[Box], List[int]]:
        accessible_boxes = []
        accessible_uids = []
        for box in boxes:
            if not any(box.is_covered_by(other) for other in boxes if other.id != box.id):
                accessible_boxes.append(box)
                accessible_uids.append(box.id)
        return accessible_boxes, accessible_uids

    def check_collision(self, candidate_box: Box, boxes: List[Box]) -> bool:
        orientation = candidate_box.orientation % 360
        cl, cw = (candidate_box.l, candidate_box.w) if orientation in [0, 180] else (candidate_box.w, candidate_box.l)
        half_l, half_w = cl / 2.0, cw / 2.0
        cx, cy, cz = candidate_box.x, candidate_box.y, candidate_box.z
        cand_x_min = cx - half_l
        cand_x_max = cx + half_l
        cand_y_min = cy - half_w
        cand_y_max = cy + half_w
        cand_z_min = cz
        cand_z_max = cz + candidate_box.h

        for box in boxes:
            if box is candidate_box:
                continue
            borient = getattr(box, 'orientation', 0) % 360
            bl, bw = (box.l, box.w) if borient in [0, 180] else (box.w, box.l)
            half_bl, half_bw = bl / 2.0, bw / 2.0
            bx = box.x + box.l / 2.0
            by = box.y + box.w / 2.0
            bz = box.z
            box_x_min = bx - half_bl
            box_x_max = bx + half_bl
            box_y_min = by - half_bw
            box_y_max = by + half_bw
            box_z_min = bz
            box_z_max = bz + box.h
            if (cand_x_min < box_x_max and cand_x_max > box_x_min and
                cand_y_min < box_y_max and cand_y_max > box_y_min and
                cand_z_min < box_z_max and cand_z_max > box_z_min):
                return True
        return False

    def has_touching_box_underneath(self, suction_box: Box, boxes: List[Box]) -> bool:
        for box in boxes:
            if box.z_top() >= suction_box.z:
                if suction_box.xy_overlap(box):
                    return True
        return False

    def find_suction_position(self, target_box: Box, all_boxes: List[Box]) -> Box:
        others = [b for b in all_boxes if b is not target_box]
        target_top_z = target_box.z_top()
        target_cx, target_cy = target_box.center()

        for ori in [0, 90]:
            sl, sw = (self.suction_template.l, self.suction_template.w) if ori % 180 == 0 else (self.suction_template.w, self.suction_template.l)
            tl, tw = (target_box.l, target_box.w) if ori % 180 == 0 else (target_box.w, target_box.l)
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
            tl, tw = (target_box.l, target_box.w) if ori % 180 == 0 else (target_box.w, target_box.l)
            for sx, sy in corner_signs:
                suction = self.suction_template.copy()
                suction.l, suction.w = sl, sw
                corner_x = target_box.x + (sx + 1) * tl / 2
                corner_y = target_box.y + (sy + 1) * tw / 2
                suction.x = corner_x - (sx + 1) * sl / 2
                suction.y = corner_y - (sy + 1) * sw / 2
                suction.z = target_top_z
                suction.orientation = ori
                if self.has_touching_box_underneath(suction, others):
                    continue
                if not self.check_collision(suction, others):
                    return suction

        return None

    def run(self, csv_path: str):
        boxes = self.load_boxes_from_csv(csv_path)
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


if __name__ == "__main__":
    pallet = Pallet(1600, 1000, 1800)
    suction_template = Box(800, 600, 1)
    planner = SuctionPlanner(pallet, suction_template)
    planner.run("./packed_boxes.csv")

