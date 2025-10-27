import os
import math
from PIL import Image

# -------------------------- 请在这里指定文件夹路径 --------------------------
base_dir = "base_dir"  # 根文件夹路径
input_images_dir = os.path.join(base_dir, "images")    # 输入图片文件夹（包含.bmp文件）
input_labels_dir = os.path.join(base_dir, "labels")    # 输入标签文件夹（包含.txt文件）
output_base = os.path.join(base_dir, "steel_sample")        # 输出结果保存的根文件夹
# --------------------------------------------------------------------------

# 检查输入文件夹是否存在
if not os.path.exists(input_images_dir):
    print(f"错误：图片文件夹 '{input_images_dir}' 不存在，请检查路径")
    exit(1)
if not os.path.exists(input_labels_dir):
    print(f"错误：标签文件夹 '{input_labels_dir}' 不存在，请检查路径")
    exit(1)

# 创建输出文件夹
os.makedirs(os.path.join(output_base, "images"), exist_ok=True)
os.makedirs(os.path.join(output_base, "labels"), exist_ok=True)

def read_yolo_labels(label_path):
    """读取YOLO格式的标签文件"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append([class_id, x_center, y_center, width, height])
    return labels

def write_yolo_labels(label_path, labels):
    """写入YOLO格式的标签文件"""
    with open(label_path, 'w') as f:
        for label in labels:
            class_id, x_center, y_center, width, height = label
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def rotate_image_and_label(image, labels, angle):
    """旋转图像和对应的标签"""
    width, height = image.size
    rotated_img = image.rotate(angle, expand=True)
    
    # 计算旋转后的新尺寸
    if angle in (90, 270):
        new_w, new_h = height, width
    else:  # 180度
        new_w, new_h = width, height
    
    new_labels = []
    for label in labels:
        class_id, x, y, w, h = label
        
        # 转换为绝对坐标
        abs_x = x * width
        abs_y = y * height
        
        # 根据旋转角度计算新坐标
        if angle == 90:
            # 旋转90度: (x, y) -> (y, width - x)
            new_abs_x = abs_y
            new_abs_y = width - abs_x
            new_w_ratio = h
            new_h_ratio = w
        elif angle == 180:
            # 旋转180度: (x, y) -> (width - x, height - y)
            new_abs_x = width - abs_x
            new_abs_y = height - abs_y
            new_w_ratio = w
            new_h_ratio = h
        elif angle == 270:
            # 旋转270度: (x, y) -> (height - y, x)
            new_abs_x = height - abs_y
            new_abs_y = abs_x
            new_w_ratio = h
            new_h_ratio = w
        
        # 转换回相对坐标
        new_x = new_abs_x / new_w
        new_y = new_abs_y / new_h
        
        # 确保坐标在有效范围内
        new_x = max(0, min(1, new_x))
        new_y = max(0, min(1, new_y))
        new_w_ratio = max(0, min(1, new_w_ratio))
        new_h_ratio = max(0, min(1, new_h_ratio))
        
        new_labels.append([class_id, new_x, new_y, new_w_ratio, new_h_ratio])
    
    return rotated_img, new_labels

def crop_top_part(image, labels, crop_ratio=0.2):
    """水平切割掉上方部分（可通过crop_ratio调整切割比例）"""
    width, height = image.size
    crop_height = int(height * crop_ratio)  # 切割的高度（上方）
    
    # 切割图像（保留下方部分）
    cropped_img = image.crop((0, crop_height, width, height))
    new_height = height - crop_height  # 切割后的图像高度
    
    new_labels = []
    for label in labels:
        class_id, x, y, w, h = label
        
        # 计算绝对坐标
        abs_y = y * height  # 目标中心y坐标（绝对）
        abs_h = h * height  # 目标高度（绝对）
        
        # 目标上边界：abs_y - abs_h/2；下边界：abs_y + abs_h/2
        # 如果目标完全在切割区域内（下边界 <= 切割线），则丢弃
        if abs_y + abs_h / 2 < crop_height:
            continue
        
        # 计算新的y坐标（相对切割后的图像）
        new_abs_y = max(crop_height, abs_y) - crop_height
        new_y = new_abs_y / new_height
        
        # 调整目标高度（如果目标被切割了一部分）
        if abs_y - abs_h / 2 < crop_height:
            new_h = (abs_y + abs_h / 2 - crop_height) / new_height
        else:
            new_h = h * height / new_height
        
        new_labels.append([class_id, x, new_y, w, new_h])
    
    return cropped_img, new_labels

def resize_image_and_label(image, labels, scale=0.8):
    """缩放图像和标签（可通过scale调整缩放比例）"""
    width, height = image.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized_img = image.resize((new_width, new_height))
    # 缩放不改变YOLO相对坐标（等比例缩放）
    return resized_img, labels.copy()

def process_file(image_path, label_path, base_name):
    """处理单个文件的所有转换"""
    # 读取原图和标签
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"处理图片 {image_path} 出错：{e}")
        return
    
    labels = read_yolo_labels(label_path)
    
    # 保存原图和原标签（可选，如需删除可注释掉）
    img.save(os.path.join(output_base, "images", f"{base_name}_original.png"))
    write_yolo_labels(os.path.join(output_base, "labels", f"{base_name}_original.txt"), labels)
    
    # 旋转处理（包含90/180/270和45/135/225/315度）
    for angle in [90, 180, 270]:
        rot_img, rot_labels = rotate_image_and_label(img, labels, angle)
        rot_img.save(os.path.join(output_base, "images", f"{base_name}_rot{angle}.png"))
        write_yolo_labels(os.path.join(output_base, "labels", f"{base_name}_rot{angle}.txt"), rot_labels)
    
    # 水平切割上方（默认20%，可修改crop_ratio参数）
    crop_img, crop_labels = crop_top_part(img, labels, crop_ratio=0.2)
    crop_img.save(os.path.join(output_base, "images", f"{base_name}_crop1.png"))
    write_yolo_labels(os.path.join(output_base, "labels", f"{base_name}_crop1.txt"), crop_labels)

    crop_img, crop_labels = crop_top_part(img, labels, crop_ratio=0.4)
    crop_img.save(os.path.join(output_base, "images", f"{base_name}_crop2.png"))
    write_yolo_labels(os.path.join(output_base, "labels", f"{base_name}_crop2.txt"), crop_labels)
    
    crop_img, crop_labels = crop_top_part(img, labels, crop_ratio=0.6)
    crop_img.save(os.path.join(output_base, "images", f"{base_name}_crop3.png"))
    write_yolo_labels(os.path.join(output_base, "labels", f"{base_name}_crop3.txt"), crop_labels)

    # 缩放（默认80%，可修改scale参数）
    resize_img, resize_labels = resize_image_and_label(img, labels, scale=0.8)
    resize_img.save(os.path.join(output_base, "images", f"{base_name}_resize1.png"))
    write_yolo_labels(os.path.join(output_base, "labels", f"{base_name}_resize1.txt"), resize_labels)

    resize_img, resize_labels = resize_image_and_label(img, labels, scale=1.2)
    resize_img.save(os.path.join(output_base, "images", f"{base_name}_resize2.png"))
    write_yolo_labels(os.path.join(output_base, "labels", f"{base_name}_resize2.txt"), resize_labels)

    resize_img, resize_labels = resize_image_and_label(img, labels, scale=1.6)
    resize_img.save(os.path.join(output_base, "images", f"{base_name}_resize3.png"))
    write_yolo_labels(os.path.join(output_base, "labels", f"{base_name}_resize3.txt"), resize_labels)

# 处理所有图片和标签
for img_file in os.listdir(input_images_dir):
    if img_file.lower().endswith(".bmp"):  
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(input_images_dir, img_file)
        label_path = os.path.join(input_labels_dir, f"{base_name}.txt")
        
        if os.path.exists(label_path):
            print(f"处理: {img_file}")
            process_file(img_path, label_path, base_name)
        else:
            print(f"警告: 未找到标签文件 '{label_path}'，跳过该图片")

print(f"所有文件处理完成，结果保存在 '{output_base}' 文件夹中")