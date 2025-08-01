class Box:
    def __init__(self, l, w, h, x=0.0, y=0.0, z=0.0, box_id=None):
        self.l = l
        self.w = w
        self.h = h
        self.x = x  # 左下角 x
        self.y = y  # 左下角 y
        self.z = z  # 左下角 z
        self.id = box_id  # 可选：箱子编号
        self.orientation=0
        self.corner=None

    def orientations(self):
        """返回该箱子的两个旋转方向（长宽互换）"""
        return [(self.l, self.w, self.h), (self.w, self.l, self.h)]

    def key(self):
        """唯一标识一个尺寸"""
        return (self.l, self.w, self.h)

    def volume(self):
        return self.l * self.w * self.h

    def x_range(self):
        return self.x, self.x + self.l

    def y_range(self):
        return self.y, self.y + self.w

    def z_top(self):
        return self.z + self.h
    def set_center(self, cx, cy):
        self.x = cx - self.l / 2.0
        self.y = cy - self.w / 2.0
    def xy_overlap(self, other):
        """判断两个箱子在 xy 平面是否有重叠"""
        ax1, ax2 = self.x_range()
        ay1, ay2 = self.y_range()
        bx1, bx2 = other.x_range()
        by1, by2 = other.y_range()

        return (ax1 < bx2 and bx1 < ax2) and (ay1 < by2 and by1 < ay2)

    def is_covered_by(self, other):
        # 条件1：XY平面投影有重叠
        if not self.xy_overlap(other):
            return False
        # 条件2：other箱子的底部不在self箱子下面（确保比较的是上方箱子）
        if other.z < self.z:
            return False
        # 条件3：other箱子的底部位置在self箱子顶面之下或持平
        return other.z <= self.z_top()
    def copy(self):
        return Box(self.l, self.w, self.h, self.x, self.y, self.z, self.id)

    def __repr__(self):
        return f"Box(id={self.id}, pos=({self.x:.2f}, {self.y:.2f}, {self.z:.2f}), size=({self.l:.2f}, {self.w:.2f}, {self.h:.2f}))"
    def center(self):
        return self.x + self.l / 2, self.y + self.w / 2

# ------------------------------
# Pallet 类
# ------------------------------
class Pallet:
    def __init__(self, l, w, h):
        self.l = l
        self.w = w
        self.h = h
        self.boxes = []
