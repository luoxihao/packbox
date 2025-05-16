import random
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



import open3d as o3d

def visualize_pallet_open3d(pallet, boxes):
    geometries = []

    # 创建托盘边界框（透明线框）
    pallet_box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(min_bound=(0,0,0), max_bound=(pallet.l, pallet.w, pallet.h))
    )
    pallet_box.paint_uniform_color([0, 0, 0])  # 黑色线框
    geometries.append(pallet_box)

    # 颜色字典（根据key映射）
    import random
    colors = [[random.random(), random.random(), random.random()] for _ in range(len(boxes))]
    color_map = {}
    unique_keys = list({box[6] for box in boxes})
    for i, key in enumerate(unique_keys):
        color_map[key] = colors[i]

    for box in boxes:
        x, y, z, l, w, h, key = box
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=l, height=w, depth=h)
        mesh_box.translate((x, y, z))
        mesh_box.paint_uniform_color(color_map[key])
        geometries.append(mesh_box)

    o3d.visualization.draw_geometries(geometries)

# 调用示例
# visualize_pallet_open3d(pallet, placed_boxes)

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