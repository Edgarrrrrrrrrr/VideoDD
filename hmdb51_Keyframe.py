import os
import cv2
import numpy as np
import pandas as pd
import shutil

# 设定路径
CSV_PATH = "/root/autodl-tmp/Data/HMDB51/hmdb51_splits.csv"  # CSV 文件路径
ROOT_FOLDER = "/root/autodl-tmp/Data/HMDB51/jpegs_112"  # RGB 视频文件夹的根目录
OUTPUT_FOLDER = "/root/autodl-tmp/Distilled_Data/HMDB51/jpegs_112"  # 关键帧输出目录
FLOW_THRESHOLD = 1.2  # 固定光流阈值


a = 0
b = 0

# 读取 CSV 文件，分类 train 和 test
df = pd.read_csv(CSV_PATH)
train_folders = df[df["split"] == "train"]["folder_name"].tolist()
test_folders = df[df["split"] == "test"]["folder_name"].tolist()

def extract_keyframes(image_folder, output_folder, threshold):
    """从 train 数据中提取关键帧，若不足 8 张则均匀采样补足"""
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(image_files) < 2:
        print(f"{image_folder} 图片数量不足，跳过")
        return

    prev_frame = cv2.imread(image_files[0])
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    os.makedirs(output_folder, exist_ok=True)
    keyframes = {image_files[0]: prev_frame}

    for i in range(1, len(image_files)):
        frame = cv2.imread(image_files[i])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitude = np.mean(mag)

        if motion_magnitude > threshold:
            keyframes[image_files[i]] = frame

        prev_gray = gray

    # 如果关键帧少于 6 张，进行均匀采样补足
    if len(keyframes) < 6:
        sampled_indices = np.linspace(0, len(image_files) - 1, 6, dtype=int)
        keyframes = {image_files[i]: cv2.imread(image_files[i]) for i in sampled_indices}

    # 重新命名并保存关键帧
    for idx, (filepath, frame) in enumerate(keyframes.items(), start=1):
        filename = f"frame{str(idx).zfill(6)}.jpg"
        cv2.imwrite(os.path.join(output_folder, filename), frame)

    print(f"提取并重命名 {len(keyframes)} 张关键帧: {image_folder}")

def copy_all_images(image_folder, output_folder):
    """直接复制 test 数据的所有 RGB 图片"""
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            src_path = os.path.join(image_folder, filename)
            dst_path = os.path.join(output_folder, filename)
            shutil.copy2(src_path, dst_path)

    print(f"复制 {len(os.listdir(image_folder))} 张图片: {image_folder}")


# 处理 train 数据（提取关键帧）
for idx, folder in enumerate(train_folders, 1):
    image_folder = os.path.join(ROOT_FOLDER, folder)
    output_folder = os.path.join(OUTPUT_FOLDER, folder)

    if os.path.exists(image_folder):
        extract_keyframes(image_folder, output_folder, FLOW_THRESHOLD)
        print(f"处理完成: {idx}/{len(train_folders)} - {folder}")
        a += 1
    else:
        print(f"文件夹 {image_folder} 不存在，跳过")

# 处理 test 数据（直接复制所有图片）
for idx, folder in enumerate(test_folders, 1):
    image_folder = os.path.join(ROOT_FOLDER, folder)
    output_folder = os.path.join(OUTPUT_FOLDER, folder)

    if os.path.exists(image_folder):
        copy_all_images(image_folder, output_folder)
        print(f"处理完成: {idx}/{len(test_folders)} - {folder}")
        b += 1
    else:
        print(f"文件夹 {image_folder} 不存在，跳过")

# 输出最终的 a 和 b 值
print(f"总共处理了 {a} 个 train 文件夹和 {b} 个 test 文件夹。")
