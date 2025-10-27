import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import distance


class SteelCounter:
    def __init__(self, image_path, threshold_low=80):
        # 初始化参数与图像读取
        self.image_path = image_path
        self.image_name = os.path.basename(image_path)
        self.threshold_low = threshold_low
        self.original_image = self._read_image()
        self.process_image = self.original_image.copy()
        self.stretched_image = self.stretch_bright_region(self.process_image)
        self.process_mask = self._create_initial_mask()
        self.most_common_scale = None
        self.target_sigma = None
        
        # 存储各次检测结果
        self.filtered_kps = []
        self.filtered_kps_third = []
        self.filtered_kps_fourth = []

    def _read_image(self):
        """读取灰度图像"""
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("无法读取图像，请检查文件路径是否正确")
        return img

    def stretch_bright_region(self, image):
        """增强亮度区间细节"""
        stretched = image.copy().astype(np.float32)
        stretched[stretched < self.threshold_low] = 0
        bright_region = stretched[stretched >= self.threshold_low]
        if len(bright_region) > 0:
            stretched[stretched >= self.threshold_low] = (
                (bright_region - self.threshold_low) / (255 - self.threshold_low) * 255
            )
        return stretched.astype(np.uint8)

    def _create_initial_mask(self):
        """创建初始高亮区域蒙版"""
        _, high_light_mask = cv2.threshold(self.stretched_image, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        high_light_mask = cv2.morphologyEx(high_light_mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(high_light_mask, cv2.MORPH_OPEN, kernel)

    def _filter_by_scale(self, keypoints, target_scale, tolerance):
        """根据尺度筛选特征点"""
        scales = [kp.size for kp in keypoints]
        if not scales:
            return []
        scale_mask = [
            (target_scale * (1 - tolerance) <= s <= target_scale * (1 + tolerance))
            for s in scales
        ]
        return [kp for kp, sm in zip(keypoints, scale_mask) if sm]

    def _remove_close_points(self, keypoints, min_dist_factor=0.9):
        """去除过近的特征点"""
        if not keypoints:
            return []
        coords = np.array([kp.pt for kp in keypoints])
        keep = []
        retained_coords = []
        for i, coord in enumerate(coords):
            if i == 0:
                keep.append(True)
                retained_coords.append(coord)
            else:
                dists = distance.cdist([coord], retained_coords)
                min_dist = np.min(dists) if retained_coords else 0
                if min_dist > self.most_common_scale * min_dist_factor:
                    keep.append(True)
                    retained_coords.append(coord)
                else:
                    keep.append(False)
        return [kp for kp, k in zip(keypoints, keep) if k]

    def _blackout_regions(self, image, mask, keypoints, radius_factor):
        """在图像和蒙版上涂黑指定区域"""
        for kp in keypoints:
            x, y = map(int, kp.pt)
            radius = int(self.most_common_scale * radius_factor)
            cv2.circle(image, (x, y), radius, 0, -1)
            cv2.circle(mask, (x, y), radius, 0, -1)

    def first_detection(self):
        """第一次检测：估计最常见的钢材尺度"""
        sift_coarse = cv2.SIFT_create(
            nfeatures=100,
            contrastThreshold=0.1,
            edgeThreshold=5,
            sigma=11
        )
        keypoints_coarse = sift_coarse.detect(self.stretched_image, mask=self.process_mask)
        scales_coarse = [kp.size for kp in keypoints_coarse]
        if not scales_coarse:
            raise ValueError("第一次检测未找到特征点，请调整图像或参数")
        
        scale_counter = Counter(scales_coarse)
        most_common_scales = scale_counter.most_common()
        top_percent_count = max(1, len(most_common_scales) // 5)  # 至少取1个

        # 计算前20%尺度的加权平均值
        top_scales_sum = sum(scale * count for scale, count in most_common_scales[:top_percent_count])
        top_counts_sum = sum(count for _, count in most_common_scales[:top_percent_count])
        
        # 使用加权平均值作为最常见尺度
        self.most_common_scale = (top_scales_sum / top_counts_sum) * 0.9
        self.target_sigma = self.most_common_scale / (5 * np.sqrt(2))
        print(f"第一次检测确定的钢材尺度（半径）: {self.most_common_scale:.2f}")

    def second_detection(self, tolerance=0.40):
        """第二次检测：基于目标尺度精准提取"""
        self.process_image = self.stretched_image.copy()  # 重置处理图像
        sift_fine = cv2.SIFT_create(
            contrastThreshold=0.05,
            edgeThreshold=4,
            sigma=self.target_sigma
        )
        keypoints_fine = sift_fine.detect(self.process_image, mask=self.process_mask)
        
        # 筛选特征点
        self.filtered_kps = self._filter_by_scale(keypoints_fine, self.most_common_scale, tolerance)
        self.filtered_kps = self._remove_close_points(self.filtered_kps)
        
        # 涂黑已检测区域
        self._blackout_regions(self.process_image, self.process_mask, self.filtered_kps, 1.2)
        self.second_process_mask = self.process_mask.copy()

    def third_detection(self, tolerance=0.40):
        """第三次检测：基于涂黑后的图像和蒙版"""
        sift_third = cv2.SIFT_create(
            contrastThreshold=0.05,
            edgeThreshold=4,
            sigma=self.target_sigma
        )
        keypoints_third = sift_third.detect(self.process_mask, mask=self.process_mask)
        
        # 筛选特征点
        self.filtered_kps_third = self._filter_by_scale(keypoints_third, self.most_common_scale, tolerance)
        self.filtered_kps_third = self._remove_close_points(self.filtered_kps_third)
        
        # 涂黑新增区域
        self._blackout_regions(self.process_image, self.process_mask, self.filtered_kps_third, 1.2)
        self.third_process_mask = self.process_mask.copy()

    def fourth_detection(self):
        """第四次检测：基于蒙版"""
        sift_fourth = cv2.SIFT_create(
            contrastThreshold=0.05,
            edgeThreshold=4,
            sigma=self.target_sigma
        )
        keypoints_fourth = sift_fourth.detect(self.process_mask, mask=self.process_mask)
        
        # 筛选特征点
        self.filtered_kps_fourth = self._filter_by_scale(
            keypoints_fourth, 
            self.most_common_scale, 
            0.25  # 这里使用单独的容忍度
        )
        self.filtered_kps_fourth = self._remove_close_points(self.filtered_kps_fourth)

    def count_and_print(self):
        """计算并打印总计数结果"""
        count2 = len(self.filtered_kps)
        count3 = len(self.filtered_kps_third)
        count4 = len(self.filtered_kps_fourth)
        total = count2 + count3 + count4
        print(f"第二次检测计数: {count2}")
        print(f"第三次检测新增计数: {count3}")
        print(f"第四次检测新增计数: {count4}")
        print(f"总计数: {total}")
        return total

    def save(self):
        vis_final = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        vis_mask = self.process_mask.copy()
        for kp in self.filtered_kps:
            x, y = map(int, kp.pt)
            cv2.circle(vis_final, (x, y), 6, (0, 0, 255), -1)
        for kp in self.filtered_kps_third:
            x, y = map(int, kp.pt)
            cv2.circle(vis_final, (x, y), 6, (0, 255, 0), -1)
        for kp in self.filtered_kps_fourth:
            x, y = map(int, kp.pt)
            cv2.circle(vis_final, (x, y), 6, (255, 0, 0), -1)
    
        # 保存结果图像
        # cv2.imwrite(f'D:/pyLearn/countSteel/results/{self.image_name}_mask.jpg', vis_mask)

        cv2.imwrite(f'D:/pyLearn/countSteel/results/{self.image_name}_result.jpg', vis_final)
    def view(self):
        """可视化检测结果"""
        # 原始图像标记第二次检测结果
        vis_original = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        vis_original[self.process_mask == 255] = vis_original[self.process_mask == 255] * 0.5 + np.array([0, 0, 255]) * 0.5
        for kp in self.filtered_kps:
            x, y = map(int, kp.pt)
            cv2.circle(vis_original, (x, y), 3, (0, 255, 0), -1)
            cv2.circle(vis_original, (x, y), int(kp.size), (255, 0, 0), 1)

        # 第二次检测后涂黑的图像
        vis_after_second = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_GRAY2BGR)
        for kp in self.filtered_kps:
            x, y = map(int, kp.pt)
            radius = int(self.most_common_scale * 1.2)
            cv2.circle(vis_after_second, (x, y), radius, (0, 0, 0), -1)
            cv2.circle(vis_after_second, (x, y), 3, (0, 255, 0), -1)

        # 第三次检测后涂黑的图像
        vis_after_third = vis_after_second.copy()
        for kp in self.filtered_kps_third:
            x, y = map(int, kp.pt)
            radius = int(self.most_common_scale * 1.1)
            cv2.circle(vis_after_third, (x, y), radius, (0, 0, 0), -1)
            cv2.circle(vis_after_third, (x, y), 3, (0, 0, 255), -1)

        # 最终结果图
        vis_final = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        for kp in self.filtered_kps:
            x, y = map(int, kp.pt)
            cv2.circle(vis_final, (x, y), 6, (0, 0, 255), -1)
        for kp in self.filtered_kps_third:
            x, y = map(int, kp.pt)
            cv2.circle(vis_final, (x, y), 6, (0, 0, 255), -1)
        for kp in self.filtered_kps_fourth:
            x, y = map(int, kp.pt)
            cv2.circle(vis_final, (x, y), 6, (0, 0, 255), -1)    
        
        

# 使用示例
if __name__ == "__main__":    
    current_dir = os.getcwd()
    image_files = glob.glob(os.path.join(current_dir, "*.bmp"))
    
    # 处理每张图片
    for image_path in image_files:
        counter = SteelCounter(image_path)
        counter.first_detection()
        counter.second_detection()
        counter.third_detection()
        counter.fourth_detection()
        counter.count_and_print()
        counter.save()