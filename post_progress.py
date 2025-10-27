import os
import math
from PIL import Image, ImageDraw

# -------------------------- 请在这里指定文件夹路径和参数 --------------------------
base_dir = "base_dir"
img_dir = os.path.join(base_dir, "task")       # 图片文件夹（存放.bmp文件）
label_dir = os.path.join(base_dir, "runs/detect/predict/labels")     # 标签文件夹（存放.txt文件）
output_dir = os.path.join(base_dir, "results")    # 输出文件夹（保存标记后的图片）
distance_threshold = 20  # 点之间的距离阈值（像素），小于此值视为"过于接近"
dot_radius = 5           # 彩色圆点的半径（像素）
min_y_ratio = 0.5        # 保留y中心坐标>此值的点（相对值，0.5即图像下半部分）
mark_color = (255, 0, 0) # 标记点颜色（RGB格式，这里用红色，可修改为其他彩色）
# --------------------------------------------------------------------------------

# 检查输入文件夹是否存在
for dir_path in [img_dir, label_dir]:
    if not os.path.exists(dir_path):
        print(f"错误：文件夹 '{dir_path}' 不存在，请检查路径")
        exit(1)

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

def read_yolo_centers(txt_path):
    """读取YOLO标签文件，提取中心点坐标（相对值）"""
    centers = []
    if not os.path.exists(txt_path):
        print(f"警告：标签文件 '{txt_path}' 不存在，跳过对应图片")
        return centers
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                x_rel = float(parts[1])
                y_rel = float(parts[2])
                centers.append((x_rel, y_rel))
    return centers

def filter_by_y(centers_rel, min_y):
    """筛选出y中心坐标（相对值）>min_y的点（仅保留下半部分）"""
    return [ (x, y) for x, y in centers_rel if y > min_y ]

def rel_to_abs(centers_rel, img_width, img_height):
    """将相对坐标（0-1）转换为绝对像素坐标"""
    centers_abs = []
    for x_rel, y_rel in centers_rel:
        x_abs = int(x_rel * img_width)
        y_abs = int(y_rel * img_height)
        # 确保坐标在图像范围内
        x_abs = max(0, min(img_width - 1, x_abs))
        y_abs = max(0, min(img_height - 1, y_abs))
        centers_abs.append((x_abs, y_abs))
    return centers_abs

def remove_close_points(centers_abs, threshold):
    """去除距离过近的点（保留第一个出现的点）"""
    kept = []
    for point in centers_abs:
        too_close = False
        for kept_point in kept:
            distance = math.hypot(point[0] - kept_point[0], point[1] - kept_point[1])
            if distance < threshold:
                too_close = True
                break
        if not too_close:
            kept.append(point)
    return kept

def draw_marks(img_path, centers_abs, output_path, radius, color):
    """在原图副本上用彩色圆点标记中心点（确保彩色显示）"""
    try:
        with Image.open(img_path) as img:
            # 无论原图是灰度还是彩色，都转换为RGB模式以支持彩色标记
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            img_copy = img.copy()
            draw = ImageDraw.Draw(img_copy)
            
            # 用指定彩色标记每个点
            for (x, y) in centers_abs:
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=color
                )
            img_copy.save(output_path)
            print(f"已保存标记图片：{output_path}")
    except Exception as e:
        print(f"处理图片 '{img_path}' 出错：{e}")

def process_image(img_file):
    """处理单张图片及对应标签"""
    img_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(img_dir, img_file)
    txt_path = os.path.join(label_dir, f"{img_name}.txt")
    output_path = os.path.join(output_dir, f"{img_name}_marked.bmp")

    # 1. 读取标签中心点（相对坐标）
    centers_rel = read_yolo_centers(txt_path)
    if not centers_rel:
        return

    # 2. 打开图片获取尺寸
    with Image.open(img_path) as img:
        img_width, img_height = img.size

    # 3. 筛选y中心坐标>0.5的点（仅保留下半部分）
    filtered_by_y = filter_by_y(centers_rel, min_y_ratio)
    if not filtered_by_y:
        print(f"图片 '{img_file}' 筛选后无符合条件的点（下半部分无点），跳过标记")
        return

    # 4. 转换为绝对像素坐标
    centers_abs = rel_to_abs(filtered_by_y, img_width, img_height)

    # 5. 去除距离过近的点
    final_centers = remove_close_points(centers_abs, distance_threshold)
    if not final_centers:
        print(f"图片 '{img_file}' 去重后无剩余点，跳过标记")
        return

    # 6. 用彩色标记并保存图片
    draw_marks(img_path, final_centers, output_path, dot_radius, mark_color)

# 批量处理所有bmp图片
for img_file in os.listdir(img_dir):
    if img_file.lower().endswith(".bmp"):
        print(f"处理图片：{img_file}")
        process_image(img_file)

print(f"所有图片处理完成，标记后的图片保存在 '{output_dir}' 文件夹中")