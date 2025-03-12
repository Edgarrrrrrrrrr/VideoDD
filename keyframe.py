import cv2
import numpy as np
import os
import shutil

def extract_keyframes_from_images(image_folder, output_folder, threshold=0.9):
    # 获取所有 RGB 图片路径，并按文件名排序
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(image_files) < 2:
        print("图片数量不足")
        return []

    # 清空输出文件夹
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # 读取第一帧
    prev_path = os.path.join(image_folder, image_files[0])
    prev_frame = cv2.imread(prev_path)
    if prev_frame is None:
        print(f"无法读取 {prev_path}")
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    keyframes = [image_files[0]]  # 记录关键帧的文件名
    cv2.imwrite(os.path.join(output_folder, image_files[0]), prev_frame)  # 保存第一帧

    for i in range(1, len(image_files)):
        frame_path = os.path.join(image_folder, image_files[i])
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"无法读取 {frame_path}")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 计算光流幅度的均值
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitude = np.mean(mag)

        if motion_magnitude > threshold:
            keyframes.append(image_files[i])  # 记录关键帧文件名
            cv2.imwrite(os.path.join(output_folder, image_files[i]), frame)  # 保存关键帧

        prev_gray = gray  # 更新前一帧

    print(f"提取了 {len(keyframes)} 张关键帧")
    return keyframes

# 示例：处理 RGB 图片文件夹
image_folder = "/opt/data/private/video_distillation-old/distill_utils/data/HMDB51/jpegs_112/_Art_of_the_Drink__Flaming_Zombie_pour_u_nm_np2_fr_med_1"
output_folder = "keyframes_output-1"

keyframes = extract_keyframes_from_images(image_folder, output_folder)



import os

def count_subfolders(folder_path):
    return sum(1 for entry in os.scandir(folder_path) if entry.is_dir())

folder_path = "/opt/data/private/video_distillation-old/distill_utils/data/HMDB51/jpegs_112"  # 替换为你的文件夹路径
num_folders = count_subfolders(folder_path)
print(f"该文件夹下共有 {num_folders} 个子文件夹")
