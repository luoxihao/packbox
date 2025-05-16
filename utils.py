import open3d as o3d

from numba import njit, prange

class VoxelCollisionChecker:
    def __init__(self, voxel_size=1):
        """
        voxel_size: 体素边长（单位同立方体尺寸）
        """
        self.voxel_size = voxel_size
        self.cubes = []  # 存放所有立方体数据，格式：(x,y,z,l,w,h,_)
        self.min_bound = None
        self.max_bound = None
        self.voxel_grid = None

    def add_cubes(self, cubes):
        """
        批量添加立方体数据
        cubes: iterable，每个元素为 (x,y,z,l,w,h,key)
        """
        self.cubes = [_[:6] for _ in cubes]

    def _compute_bounds(self):
        """
        计算整体包围盒边界
        """
        cubes_np = np.array(self.cubes)
        self.min_bound = np.min(cubes_np[:, :3], axis=0)
        self.max_bound = np.max(cubes_np[:, :3] + cubes_np[:, 3:], axis=0)

    def _compute_voxel_indices(self, cubes_np):
        starts = np.floor((cubes_np[:, :3] - self.min_bound) / self.voxel_size).astype(np.int32)
        ends = np.ceil((cubes_np[:, :3] + cubes_np[:, 3:] - self.min_bound) / self.voxel_size).astype(np.int32)
        return starts, ends

    @staticmethod
    @njit(parallel=True)
    def _fill_voxel_grid(voxel_grid, starts, ends):
        collided = False
        for i in prange(starts.shape[0]):
            xs, ys, zs = starts[i]
            xe, ye, ze = ends[i]
            for x in range(xs, xe):
                for y in range(ys, ye):
                    for z in range(zs, ze):
                        if voxel_grid[x, y, z]:
                            collided = True
                        voxel_grid[x, y, z] = True
        return collided

    def check_collision(self):
        """
        构建体素网格并检查所有立方体是否发生碰撞
        返回 True 表示有碰撞，False 表示无碰撞
        """
        if not self.cubes:
            return False

        self._compute_bounds()
        cubes_np = np.array(self.cubes)

        grid_shape = np.ceil((self.max_bound - self.min_bound) / self.voxel_size).astype(np.int32)
        self.voxel_grid = np.zeros(grid_shape, dtype=np.bool_)

        starts, ends = self._compute_voxel_indices(cubes_np)
        collided = self._fill_voxel_grid(self.voxel_grid, starts, ends)
        return collided















def make_o3d_box(x, y, z, l, w, h):
    mesh = o3d.geometry.TriangleMesh.create_box(width=l, height=w, depth=h)
    mesh.translate((x, y, z))
    return mesh
class collision_check:
    def __init__(self):
        pass
    @staticmethod
    def base_check(x, y, z, bl, bw, bh, placed):
        for px, py, pz, pl, pw, ph, _ in placed:
            if (
                    x < px + pl and x + bl > px and
                    y < py + pw and y + bw > py and
                    z < pz + ph and z + bh > pz
            ):
                return True  # 有重叠
        return False  # 无重叠
    @staticmethod
    def open3d_check(x, y, z, l, w, h, placed):
        def aabb_overlap(a, b):
            a_min = a.get_min_bound()
            a_max = a.get_max_bound()
            b_min = b.get_min_bound()
            b_max = b.get_max_bound()
            return (a_min[0] < b_max[0] and a_max[0] > b_min[0] and
                    a_min[1] < b_max[1] and a_max[1] > b_min[1] and
                    a_min[2] < b_max[2] and a_max[2] > b_min[2])

        new_box = make_o3d_box(x, y, z, l, w, h)
        new_aabb = new_box.get_axis_aligned_bounding_box()
        for bx, by, bz, bl, bw, bh, _ in placed:
            existing_box = make_o3d_box(bx, by, bz, bl, bw, bh)
            existing_aabb = existing_box.get_axis_aligned_bounding_box()
            if aabb_overlap(new_aabb, existing_aabb):
                return True
        return False

def check_available(x, y, z, bl, bw, bh, placed):
    return not collision_check.base_check(x, y, z, bl, bw, bh, placed) and  is_supported((x, y, z, bl, bw, bh), placed)
    # return collision_check.open3d_check(x, y, z, bl, bw, bh, placed)

import numpy as np
from scipy.spatial import ConvexHull



def is_supported(new_box, placed_boxes, z_eps=1):
    new_box = make_o3d_box(*new_box)
    placed_boxes = [make_o3d_box(x,y,z,bl,bw,bh) for (x,y,z,bl,bw,bh,key) in placed_boxes]
    def compute_convex_hull_2d(points):
        if len(points) < 3:
            return points
        points = np.array(points)
        hull = ConvexHull(points)
        return points[hull.vertices]

    def point_in_polygon(point, polygon):
        # 射线法判断 point 是否在 polygon 内
        x, y = point
        num = len(polygon)
        inside = False
        for i in range(num):
            xi, yi = polygon[i]
            xj, yj = polygon[(i + 1) % num]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
        return inside

    x_min, y_min, z_min = new_box.get_min_bound()
    x_max, y_max, z_max = new_box.get_max_bound()
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    support_points = []

    # 地面支撑
    if z_min <= z_eps:
        support_points += [(x_min, y_min), (x_min, y_max),
                           (x_max, y_min), (x_max, y_max)]

    for box in placed_boxes:
        bx_min, by_min, bz_min = box.get_min_bound()
        bx_max, by_max, bz_max = box.get_max_bound()

        if abs(z_min - bz_max) < z_eps:
            # 计算底面和顶面的交集
            overlap_x_min = max(x_min, bx_min)
            overlap_x_max = min(x_max, bx_max)
            overlap_y_min = max(y_min, by_min)
            overlap_y_max = min(y_max, by_max)
            if overlap_x_max > overlap_x_min and overlap_y_max > overlap_y_min:
                support_points += [
                    (overlap_x_min, overlap_y_min),
                    (overlap_x_min, overlap_y_max),
                    (overlap_x_max, overlap_y_min),
                    (overlap_x_max, overlap_y_max)
                ]

    if len(support_points) < 3:
        return False  # 至少需要一个三角形支撑区域

    hull = compute_convex_hull_2d(support_points)
    return point_in_polygon((cx, cy), hull)






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
def _try_merge(ems1, ems2):
    x1, y1, z1, l1, w1, h1 = ems1
    x2, y2, z2, l2, w2, h2 = ems2

    # 沿 X 轴合并
    if y1 == y2 and w1 == w2 and z1 == z2 and h1 == h2:
        if x1 + l1 == x2:
            return (x1, y1, z1, l1 + l2, w1, h1)
        if x2 + l2 == x1:
            return (x2, y2, z2, l1 + l2, w1, h1)

    # 沿 Y 轴合并
    if x1 == x2 and l1 == l2 and z1 == z2 and h1 == h2:
        if y1 + w1 == y2:
            return (x1, y1, z1, l1, w1 + w2, h1)
        if y2 + w2 == y1:
            return (x2, y2, z2, l1, w1 + w2, h1)

    # 沿 Z 轴合并
    if x1 == x2 and l1 == l2 and y1 == y2 and w1 == w2:
        if z1 + h1 == z2:
            return (x1, y1, z1, l1, w1, h1 + h2)
        if z2 + h2 == z1:
            return (x2, y2, z2, l1, w1, h1 + h2)

    return None

def merge_ems(ems_list):
    ems_list = ems_list[:]  # 复制，避免修改外部
    merged = True
    while merged:
        merged = False
        new_ems_list = []
        skip = set()
        for i in range(len(ems_list)):
            if i in skip:
                continue
            for j in range(i + 1, len(ems_list)):
                if j in skip:
                    continue
                merged_box = _try_merge(ems_list[i], ems_list[j])
                if merged_box is not None:
                    new_ems_list.append(merged_box)
                    skip.add(i)
                    skip.add(j)
                    merged = True
                    break
            else:
                # 只有当内层循环没有break时，才执行这句（无合并）
                if i not in skip:
                    new_ems_list.append(ems_list[i])
        ems_list = new_ems_list
    return ems_list


def compute_metrics(pallet, boxes):
    total_volume = pallet.l * pallet.w * pallet.h
    box_volume = sum(l * w * h for _, _, _, l, w, h, _ in boxes)
    max_height = max((z + h) for _, _, z, _, _, h, _ in boxes) if boxes else 0
    used_volume = max_height * pallet.l * pallet.w
    return box_volume / total_volume, box_volume / used_volume, max_height