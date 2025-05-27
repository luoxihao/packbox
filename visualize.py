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

    # # 颜色字典（根据key映射）
    # import random
    # colors = [[random.random(), random.random(), random.random()] for _ in range(len(boxes))]
    # color_map = {}
    # unique_keys = list({box[6] for box in boxes})
    # for i, key in enumerate(unique_keys):
    #     color_map[key] = colors[i]

    for box in boxes:
        x, y, z, l, w, h, key = box
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=l, height=w, depth=h)
        mesh_box.translate((x, y, z))
        # mesh_box.paint_uniform_color(color_map[key])
        mesh_box.paint_uniform_color([random.random(), random.random(), random.random()])
        geometries.append(mesh_box)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=min(pallet.l, pallet.w, pallet.h) * 0.2,origin=[0, 0, 0])
    geometries.append(axis)
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


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# 你的箱子数据
boxes = [
  {"l": 290, "w": 550, "h": 150},
  {"l": 700, "w": 750, "h": 150},
  {"l": 160, "w": 400, "h": 150},
  {"l": 170, "w": 650, "h": 150},
  {"l": 260, "w": 740, "h": 150},
  {"l": 270, "w": 610, "h": 150},
  {"l": 320, "w": 240, "h": 150},
  {"l": 680, "w": 680, "h": 150},
  {"l": 430, "w": 170, "h": 150},
  {"l": 590, "w": 740, "h": 150},
  {"l": 190, "w": 340, "h": 150},
  {"l": 240, "w": 760, "h": 150},
  {"l": 630, "w": 260, "h": 150},
  {"l": 230, "w": 660, "h": 150},
  {"l": 370, "w": 210, "h": 150},
  {"l": 210, "w": 620, "h": 150},
  {"l": 340, "w": 250, "h": 150},
  {"l": 320, "w": 590, "h": 150},
  {"l": 780, "w": 750, "h": 150},
  {"l": 220, "w": 260, "h": 150},
  {"l": 230, "w": 330, "h": 150},
  {"l": 420, "w": 240, "h": 150},
  {"l": 370, "w": 760, "h": 150},
  {"l": 490, "w": 190, "h": 150},
  {"l": 280, "w": 350, "h": 150}
]

def plot_cube(ax, origin, size, color):
    # origin: (x,y,z), size: (dx,dy,dz)
    x, y, z = origin
    dx, dy, dz = size

    # 八个顶点
    vertices = np.array([[x, y, z],
                         [x+dx, y, z],
                         [x+dx, y+dy, z],
                         [x, y+dy, z],
                         [x, y, z+dz],
                         [x+dx, y, z+dz],
                         [x+dx, y+dy, z+dz],
                         [x, y+dy, z+dz]])
    # 六个面，每个面由4个顶点索引组成
    faces = [
        [vertices[j] for j in [0,1,2,3]],  # bottom
        [vertices[j] for j in [4,5,6,7]],  # top
        [vertices[j] for j in [0,1,5,4]],  # front
        [vertices[j] for j in [2,3,7,6]],  # back
        [vertices[j] for j in [1,2,6,5]],  # right
        [vertices[j] for j in [4,7,3,0]]   # left
    ]
    poly3d = Poly3DCollection(faces, alpha=0.3, facecolor=color)
    ax.add_collection3d(poly3d)

fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(111, projection='3d')

# 按顺序摆放箱子，x方向间隔为箱子长加50mm间隙，y,z固定
x_offset = 0
for box in boxes:
    size = (box['l'], box['w'], box['h'])
    origin = (x_offset, 0, 0)
    color = np.random.rand(3,)  # 随机颜色
    plot_cube(ax, origin, size, color)
    x_offset += box['l'] + 50  # 下一个箱子往右移

# 设置坐标轴标签
ax.set_xlabel('长度 (mm)')
ax.set_ylabel('宽度 (mm)')
ax.set_zlabel('高度 (mm)')
ax.set_title('箱子尺寸可视化排列')

plt.show()