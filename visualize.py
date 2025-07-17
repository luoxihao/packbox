import random
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d


import hashlib

def uid_to_rgb(uid,epsilon=0.03):
    """将uid哈希为RGB颜色（范围0-1）"""
    uid_bytes = str(uid).encode('utf-8')
    digest = hashlib.md5(uid_bytes).hexdigest()
    r = int(digest[0:2], 16) / 255.0
    g = int(digest[2:4], 16) / 255.0
    b = int(digest[4:6], 16) / 255.0
    # 添加微小扰动，确保不超过 [0, 1]
    r = min(max(r + random.uniform(-epsilon, epsilon), 0.0), 1.0)
    g = min(max(g + random.uniform(-epsilon, epsilon), 0.0), 1.0)
    b = min(max(b + random.uniform(-epsilon, epsilon), 0.0), 1.0)
    return (0, g, b)

import open3d as o3d
import random

def visualize_pallet_open3d(pallet, boxes, accessible_boxes_uid=None, suctions=None):
    geometries = []

    # 1. 托盘线框
    pallet_box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(min_bound=(0, 0, 0), max_bound=(pallet.l, pallet.w, pallet.h))
    )
    pallet_box.paint_uniform_color([0, 0, 0])
    geometries.append(pallet_box)

    # 2. 箱子及其线框
    for box in boxes:
        x, y, z, l, w, h, uid = box
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=l, height=w, depth=h)
        mesh_box.translate((x, y, z))

        # 着色逻辑
        if accessible_boxes_uid and uid in accessible_boxes_uid:
            mesh_box.paint_uniform_color([
                1.0,
                min(max(random.uniform(-0.3, 0.3), 0.0), 1.0),
                min(max(random.uniform(-0.3, 0.3), 0.0), 1.0)
            ])
        else:
            mesh_box.paint_uniform_color(uid_to_rgb(uid))  # 哈希色

        geometries.append(mesh_box)

        # 加线框
        bbox = mesh_box.get_axis_aligned_bounding_box()
        lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
        lineset.paint_uniform_color([0.2, 0.2, 0.2])  # 深灰色边框
        geometries.append(lineset)

    # 3. 吸盘（可选）
    if suctions:
        for suction in suctions:
            x, y, z, l, w, h, uid = suction
            mesh_box = o3d.geometry.TriangleMesh.create_box(width=l, height=w, depth=h)
            mesh_box.translate((x, y, z))
            mesh_box.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色

            geometries.append(mesh_box)

            # 吸盘线框
            bbox = mesh_box.get_axis_aligned_bounding_box()
            lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
            lineset.paint_uniform_color([0.1, 0.1, 0.1])  # 更深的边框色
            geometries.append(lineset)

    # 4. 坐标系
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=min(pallet.l, pallet.w, pallet.h) * 0.2, origin=[0, 0, 0]
    )
    geometries.append(axis)

    # 5. 显示
    o3d.visualization.draw_geometries(geometries)



def visualize_pallet(pallet, boxes, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique_keys = list({key for *_, key in boxes})
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    random.shuffle(colors)
    color_map = {key: colors[i % len(colors)] for i, key in enumerate(unique_keys)}

    for box in boxes:
        x, y, z, l, w, h, key = box
        draw_cube(ax, x, y, z, l, w, h, color_map[key])

    ax.set_xlim(0, pallet.l)
    ax.set_ylim(0, pallet.w)
    ax.set_zlim(0, pallet.h)
    ax.set_xlabel('Length')
    ax.set_ylabel('Width')
    ax.set_zlabel('Height')
    plt.tight_layout()
    plt.show()
    #plt.savefig(save_path)
    plt.close()


def draw_cube(ax, x, y, z, l, w, h, color):
    corners = [
        [x, y, z], [x + l, y, z], [x + l, y + w, z], [x, y + w, z],
        [x, y, z + h], [x + l, y, z + h], [x + l, y + w, z + h], [x, y + w, z + h]
    ]
    faces = [[corners[j] for j in [0,1,2,3]],
             [corners[j] for j in [4,5,6,7]],
             [corners[j] for j in [0,1,5,4]],
             [corners[j] for j in [2,3,7,6]],
             [corners[j] for j in [1,2,6,5]],
             [corners[j] for j in [0,3,7,4]]]
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.6, facecolors=color))


