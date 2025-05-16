from utils import does_collide, split_ems,merge_ems
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
                        not does_collide(x, y, z, bl, bw, bh, placed)):
                        placed.append((x, y, z, bl, bw, bh, box.key()))
                        self.pallet.boxes.append((x, y, z, bl, bw, bh, box.key()))
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
                        if not does_collide(ex, ey, ez, bl, bw, bh, placed):
                            new_box = (ex, ey, ez, bl, bw, bh, box.key())
                            placed.append(new_box)
                            self.pallet.boxes.append(new_box)

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

class EMSMULPacker:
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
                    for dx in [0, el - bl]:
                        for dy in [0, ew - bw]:
                            for dz in [0, eh - bh]:
                                px, py, pz = ex + dx, ey + dy, ez + dz
                                if px + bl <= self.pallet.l and py + bw <= self.pallet.w and pz + bh <= self.pallet.h:
                                    if not does_collide(px, py, pz, bl, bw, bh, placed):
                                        new_box = (px, py, pz, bl, bw, bh, box.key())
                                        placed.append(new_box)
                                        self.pallet.boxes.append(new_box)
                                        old_ems = ems_list.pop(i)
                                        new_ems = split_ems(old_ems, new_box[:6])
                                        ems_list.extend(new_ems)
                                        ems_list = merge_ems(ems_list)
                                        placed_flag = True
                                        break
                            if placed_flag:
                                break
                        if placed_flag:
                            break
                    if placed_flag:
                        break
                if placed_flag:
                    break
            if not placed_flag:
                unplaced.append(box)
        return placed, unplaced

class EMSMACSPacker:
    def __init__(self, pallet):
        self.pallet = pallet

    def pack(self, boxes):
        placed = []
        unplaced = []
        ems_list = [(0, 0, 0, self.pallet.l, self.pallet.w, self.pallet.h)]

        for box in boxes:
            best_score = -1
            best_box = None
            best_ems_index = None
            best_split = None

            for orientation in box.orientations():
                bl, bw, bh = orientation
                for i, (ex, ey, ez, el, ew, eh) in enumerate(ems_list):
                    for dx in [0, el - bl]:
                        for dy in [0, ew - bw]:
                            for dz in [0, eh - bh]:
                                px, py, pz = ex + dx, ey + dy, ez + dz
                                if px + bl <= self.pallet.l and py + bw <= self.pallet.w and pz + bh <= self.pallet.h:
                                    if not does_collide(px, py, pz, bl, bw, bh, placed):
                                        temp_box = (px, py, pz, bl, bw, bh, box.key())
                                        used_vol = sum(l * w * h for _, _, _, l, w, h, _ in placed) + bl * bw * bh
                                        remaining_vol = self.pallet.l * self.pallet.w * self.pallet.h - used_vol
                                        if remaining_vol > best_score:
                                            best_score = remaining_vol
                                            best_box = temp_box
                                            best_ems_index = i
                                            best_split = split_ems(ems_list[i], temp_box[:6])

            if best_box:
                placed.append(best_box)
                self.pallet.boxes.append(best_box)
                del ems_list[best_ems_index]
                ems_list.extend(best_split)
            else:
                unplaced.append(box)

        return placed, unplaced