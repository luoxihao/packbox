from utils import check_available, split_ems,merge_ems
from visualize import visualize_pallet_open3d
import random
from bin_packing.py3dbp import Packer, Bin, Item, Painter
binpacker = Packer
from decimal import Decimal
class BinPacker:
    def __init__(self, pallet):
        self.pallet = pallet
        self.box = Bin(
        partno='example0',
        WHD=(pallet.l,pallet.w,pallet.h),
        max_weight=9e10,
        corner=0,
        put_type=2#顶部开孔的容器 托盘
        )
        self.id = 0
        self.packer = binpacker()
        self.packer.addBin(self.box)
    def pack(self, boxes):
        for box in boxes:
            self.packer.addItem(self._box2item(box))
        self.packer.pack(
            bigger_first=True,
            distribute_items=False,
            fix_point=True, # Try switching fix_point=True/False to compare the results
            check_stable=True,
            support_surface_ratio=0.75,
            number_of_decimals=0
        )
        
        placed=[]
        unplaced = []

        for bin in self.packer.bins:
            for item in bin.items:
                
                placed.append((item.position[0],
                                item.position[1],
                                item.position[2],
                                item.getDimension()[0],
                                item.getDimension()[1],
                                item.getDimension()[2],
                                (item.width,item.height,item.depth)
                                ))
            for item in bin.unfitted_items:
                unplaced.append((item.position[0],
                                item.position[1],
                                item.position[2],
                                item.getDimension()[0],
                                item.getDimension()[1],
                                item.getDimension()[2],
                                (item.width,item.height,item.depth)
                                ))
        return placed, unplaced


    def _box2item(self, box):
        def random_color():
            return "#{:06X}".format(random.randint(0, 0xFFFFFF))
        
        item=Item(
            partno='id:{}'.format(str(self.id)),
            name='box',
            typeof='cube',
            WHD=(box.l, box.w, box.h), 
            weight=1,
            level=1,
            loadbear=1e10,
            updown=True,
            color=random_color()
        )
        self.id+=1
        return item
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
                        # 添加新空间
                        free_spaces.append((x + bl, y, z))
                        free_spaces.append((x, y + bw, z))
                        free_spaces.append((x, y, z + bh))

                        # 删除当前已用空间
                        del free_spaces[i]

                        # 按 z升 y升 x升 排序（小优先）
                        free_spaces.sort(key=lambda pos: (pos[2], pos[1], pos[0]))
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
