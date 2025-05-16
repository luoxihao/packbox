from utils import check_available, split_ems,merge_ems
from visualize import visualize_pallet_open3d
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
                        check_available(x, y, z, bl, bw, bh, placed)):
                        placed.append((x, y, z, bl, bw, bh, box.key()))
                        # self.pallet.boxes.append((x, y, z, bl, bw, bh, box.key()))
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
    


class EMSLBPacker:
    def __init__(self, pallet):
        self.pallet = pallet

    def pack(self, boxes):
        placed = []
        unplaced = []
        ems_list = [(0, 0, 0, self.pallet.l, self.pallet.w, self.pallet.h)]

        for box in boxes:
            placed_flag = False
            for orientation in box.orientations():
                bl, bw, bh = orientation
                for i, (ex, ey, ez, el, ew, eh) in enumerate(ems_list):
                    if bl <= el and bw <= ew and bh <= eh:
                        if  check_available(ex, ey, ez, bl, bw, bh, placed):
                            new_box = (ex, ey, ez, bl, bw, bh, box.key())
                            placed.append(new_box)
                            # self.pallet.boxes.append(new_box)

                            old_ems = ems_list.pop(i)
                            new_ems = split_ems(old_ems, new_box[:6])
                            ems_list.extend(new_ems)
                            ems_list = merge_ems(ems_list)

                            placed_flag = True
                            break
                if placed_flag:
                    break
            if not placed_flag:
                unplaced.append(box)
        return placed, unplaced
