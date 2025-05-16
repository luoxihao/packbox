def does_collide(x, y, z, bl, bw, bh, placed):
    for px, py, pz, pl, pw, ph, _ in placed:
        if (
            x < px + pl and x + bl > px and
            y < py + pw and y + bw > py and
            z < pz + ph and z + bh > pz
        ):
            return True  # 有重叠
    return False  # 无重叠

def split_ems(ems, box_pos):
    x, y, z, l, w, h = ems
    bx, by, bz, bl, bw, bh = box_pos

    if (bx >= x + l or bx + bl <= x or
        by >= y + w or by + bw <= y or
        bz >= z + h or bz + bh <= z):
        return [ems]

    new_spaces = []
    if bx > x:
        new_spaces.append((x, y, z, bx - x, w, h))
    if bx + bl < x + l:
        new_spaces.append((bx + bl, y, z, x + l - (bx + bl), w, h))
    if by > y:
        new_spaces.append((x, y, z, l, by - y, h))
    if by + bw < y + w:
        new_spaces.append((x, by + bw, z, l, y + w - (by + bw), h))
    if bz > z:
        new_spaces.append((x, y, z, l, w, bz - z))
    if bz + bh < z + h:
        new_spaces.append((x, y, bz + bh, l, w, z + h - (bz + bh)))

    return new_spaces
def merge_ems(ems_list):
    merged = []
    for i, ems1 in enumerate(ems_list):
        x1, y1, z1, l1, w1, h1 = ems1
        contained = False
        for j, ems2 in enumerate(ems_list):
            if i != j:
                x2, y2, z2, l2, w2, h2 = ems2
                if (x1 >= x2 and y1 >= y2 and z1 >= z2 and
                    x1 + l1 <= x2 + l2 and y1 + w1 <= y2 + w2 and z1 + h1 <= z2 + h2):
                    contained = True
                    break
        if not contained:
            merged.append(ems1)
    return merged

def compute_metrics(pallet, boxes):
    total_volume = pallet.l * pallet.w * pallet.h
    box_volume = sum(l * w * h for _, _, _, l, w, h, _ in boxes)
    max_height = max((z + h) for _, _, z, _, _, h, _ in boxes) if boxes else 0
    used_volume = max_height * pallet.l * pallet.w
    return box_volume / total_volume, box_volume / used_volume, max_height