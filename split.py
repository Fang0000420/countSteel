import os
import shutil
import random
from pathlib import Path

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    """
    划分数据集并复制到指定目录
    
    Args:
        image_dir (str): 源图片文件夹路径
        label_dir (str): 源标签文件夹路径
        output_dir (str): 输出文件夹路径
        train_ratio (float): 训练集比例
        test_ratio (float): 测试集比例
        val_ratio (float): 验证集比例
    """
    # 检查比例是否合法
    if not (abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6):
        raise ValueError("训练集、测试集和验证集的比例之和必须为1")
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_dir) if Path(f).suffix.lower() == '.png']
    
    # 检查每个图片是否有对应的标签文件
    valid_files = []
    for img_file in image_files:
        img_name = Path(img_file).stem
        label_file = f"{img_name}.txt"
        if label_file in os.listdir(label_dir):
            valid_files.append(img_name)
        else:
            print(f"警告: 图片 {img_file} 没有对应的标签文件，已跳过")
    
    # 打乱文件顺序
    random.shuffle(valid_files)
    total = len(valid_files)
    
    if total == 0:
        print("错误: 没有找到有效的图片和标签文件对")
        return
    
    # 计算各数据集的数量
    train_count = int(total * train_ratio)
    test_count = int(total * test_ratio)
    val_count = total - train_count - test_count  # 确保总和正确
    
    print(f"总文件数: {total}")
    print(f"训练集: {train_count}, 测试集: {test_count}, 验证集: {val_count}")
    
    # 划分数据集
    train_files = valid_files[:train_count]
    test_files = valid_files[train_count:train_count+test_count]
    val_files = valid_files[train_count+test_count:]
    
    # 创建输出目录结构
    dir_structure = [
        os.path.join(output_dir, 'images', 'train'),
        os.path.join(output_dir, 'images', 'test'),
        os.path.join(output_dir, 'images', 'val'),
        os.path.join(output_dir, 'labels', 'train'),
        os.path.join(output_dir, 'labels', 'test'),
        os.path.join(output_dir, 'labels', 'val')
    ]
    
    for dir_path in dir_structure:
        os.makedirs(dir_path, exist_ok=True)
    
    # 复制文件到相应目录
    def copy_files(file_list, dest_image_dir, dest_label_dir):
        for file_name in file_list:
            # 复制图片
            src_img = os.path.join(image_dir, f"{file_name}.png")
            dest_img = os.path.join(dest_image_dir, f"{file_name}.png")
            shutil.copy2(src_img, dest_img)  # 保留文件元数据
            
            # 复制标签
            src_label = os.path.join(label_dir, f"{file_name}.txt")
            dest_label = os.path.join(dest_label_dir, f"{file_name}.txt")
            shutil.copy2(src_label, dest_label)
    
    # 复制训练集
    copy_files(train_files, 
              os.path.join(output_dir, 'images', 'train'),
              os.path.join(output_dir, 'labels', 'train'))
    
    # 复制测试集
    copy_files(test_files, 
              os.path.join(output_dir, 'images', 'test'),
              os.path.join(output_dir, 'labels', 'test'))
    
    # 复制验证集
    copy_files(val_files, 
              os.path.join(output_dir, 'images', 'val'),
              os.path.join(output_dir, 'labels', 'val'))
    
    print("数据集划分完成!")

if __name__ == "__main__":
    # 设置路径
    base_dir = "D:/pyLearn/countSteel/"
    image_directory = os.path.join(base_dir, "steel_sample/images")  # 源图片文件夹
    label_directory = os.path.join(base_dir, "steel_sample/labels")  # 源标签文件夹
    output_directory = os.path.join(base_dir, "datasets/steel")  # 输出文件夹

    # 设置划分比例 - 可以根据需要调整
    train_ratio = 0.7   # 70% 训练集
    test_ratio = 0.2    # 20% 测试集
    val_ratio = 0.1     # 10% 验证集
    
    # 执行划分
    split_dataset(image_directory, label_directory, output_directory, train_ratio, test_ratio, val_ratio)