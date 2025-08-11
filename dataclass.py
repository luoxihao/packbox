class Box:
    def __init__(self, l, w, h, x=0.0, y=0.0, z=0.0, box_id=None, orientation=0):
        self.l = l  # 原始长度（沿x）
        self.w = w  # 原始宽度（沿y）
        self.h = h
        self.x = x  # 最小角 x（世界坐标）
        self.y = y  # 最小角 y（世界坐标）
        self.z = z  # 最小角 z
        self.id = box_id
        self.orientation = orientation % 360  # 朝向(度数)

    # ------------------------------
    # 尺寸相关（考虑旋转）
    # ------------------------------
    @property
    def lx(self):
        """旋转后沿世界 x 方向的尺寸"""
        return self.l if self.orientation in (0, 180) else self.w

    @property
    def ly(self):
        """旋转后沿世界 y 方向的尺寸"""
        return self.w if self.orientation in (0, 180) else self.l

    # ------------------------------
    # 范围 & 中心
    # ------------------------------
    def x_range(self):
        return self.x, self.x + self.lx

    def y_range(self):
        return self.y, self.y + self.ly

    def z_top(self):
        return self.z + self.h

    def center(self):
        return self.x + self.lx / 2.0, self.y + self.ly / 2.0

    def set_center(self, cx, cy):
        """设置旋转后中心到(cx, cy)"""
        self.x = cx - self.lx / 2.0
        self.y = cy - self.ly / 2.0

    # ------------------------------
    # 重叠 & 覆盖判断
    # ------------------------------
    def xy_overlap(self, other, eps: float = 1e-9):
        """判断两个箱子在XY平面是否有重叠（贴边不算重叠）"""
        ax1, ax2 = self.x_range()
        ay1, ay2 = self.y_range()
        bx1, bx2 = other.x_range()
        by1, by2 = other.y_range()

        separated = (
            ax2 <= bx1 + eps or ax1 >= bx2 - eps or
            ay2 <= by1 + eps or ay1 >= by2 - eps
        )
        return not separated

    def is_covered_by(self, other, eps: float = 1e-9):
        """判断当前箱子是否被另一个箱子覆盖"""
        if not self.xy_overlap(other, eps):
            return False
        if other.z < self.z - eps:
            return False
        return other.z <= self.z_top() + eps

    # ------------------------------
    # 其他工具
    # ------------------------------
    def copy(self):
        return Box(self.l, self.w, self.h, self.x, self.y, self.z, self.id, self.orientation)

    def __repr__(self):
        return (f"Box(id={self.id}, pos=({self.x:.2f}, {self.y:.2f}, {self.z:.2f}), "
                f"size=({self.lx:.2f}, {self.ly:.2f}, {self.h:.2f}), ori={self.orientation})")

# ------------------------------
# Pallet 类
# ------------------------------
class Pallet:
    def __init__(self, l, w, h):
        self.l = l
        self.w = w
        self.h = h
        self.boxes = []
