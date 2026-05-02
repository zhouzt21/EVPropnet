#!/usr/bin/env python3
import os
import cv2
from glob import glob
import re

root_dir = "/home/EVPropNet/DVSProp/Outputs/our_data_100000_rectangle"
output_video_dir = os.path.join(root_dir, "Videos")

os.makedirs(output_video_dir, exist_ok=True)

# 提取文件名中的数字用于排序
def extract_number(path):
    nums = re.findall(r'\d+', os.path.basename(path))
    return int(nums[-1]) if nums else 0

# 遍历每个子文件夹
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)

    if not os.path.isdir(subfolder_path):
        continue

    overlay_path = os.path.join(subfolder_path, "Boxes")

    if not os.path.exists(overlay_path):
        print(f"Skip {subfolder}, no Boxes folder")
        continue

    # ✅ 按数字排序
    images = sorted(glob(os.path.join(overlay_path, "*.png")),
                    key=extract_number)

    if len(images) == 0:
        print(f"No images in {overlay_path}")
        continue

    # 读取第一张图获取尺寸
    first_img = cv2.imread(images[0])
    height, width, _ = first_img.shape

    # ✅ 视频名称 = 子文件夹名
    video_name = f"{subfolder}.mp4"
    video_path = os.path.join(output_video_dir, video_name)

    # 视频编码
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10

    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    print(f"Processing {subfolder} → {video_name}")

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        out.write(frame)

    out.release()

    print(f"Saved: {video_path}")

print("All videos generated!")