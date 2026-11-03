# -*- coding: utf-8 -*-
"""
Tự động sinh tập dữ liệu huấn luyện YOLO11n-Pose 4 góc biển số xe máy và ô tô Việt Nam.
Bao gồm:
- Góc xoay nghiêng (Perspective Warp, Rotation từ -25 đến +25 độ)
- Đầy đủ 4 điểm góc (TL, TR, BR, BL) chuẩn định dạng YOLO Pose.
- Xuất nhãn: <class> <x_center> <y_center> <w> <h> <p1_x> <p1_y> 2 <p2_x> <p2_y> 2 <p3_x> <p3_y> 2 <p4_x> <p4_y> 2
"""

import os
import random
import glob
import cv2
import numpy as np

def create_pose_sample_from_crop(plate_img, bg_w=640, bg_h=640):
    ph, pw = plate_img.shape[:2]
    if ph < 30 or pw < 30:
        return None, None

    # Tạo nền tự nhiên ngẫu nhiên (nền đường, xám, bóng xe, hoặc nhiễu mờ)
    bg_color = random.randint(30, 160)
    canvas = np.full((bg_h, bg_w, 3), bg_color, dtype=np.uint8)
    # Thêm noise nhẹ
    noise = np.random.randint(-15, 15, (bg_h, bg_w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 4 góc ban đầu của ảnh biển số
    src_pts = np.float32([
        [0, 0],         # TL
        [pw - 1, 0],     # TR
        [pw - 1, ph - 1], # BR
        [0, ph - 1]      # BL
    ])

    # Random tỷ lệ phóng to thu nhỏ biển số trên khung 640x640
    # Thông thường biển số chiếm từ 150px đến 350px
    target_w = random.randint(160, 340)
    target_h = int(target_w * (ph / float(pw)))

    # Random tọa độ đặt biển số trên canvas
    max_x = bg_w - target_w - 40
    max_y = bg_h - target_h - 40
    if max_x <= 40 or max_y <= 40:
        return None, None

    start_x = random.randint(40, max_x)
    start_y = random.randint(40, max_y)

    # Biến dạng góc nghiêng thực tế (Perspective Skew / Tilt)
    # Mô phỏng góc camera nhìn chéo từ trên xuống và từ bên cạnh
    dx1 = random.randint(-18, 18)
    dy1 = random.randint(-15, 15)
    dx2 = random.randint(-18, 18)
    dy2 = random.randint(-15, 15)
    dx3 = random.randint(-18, 18)
    dy3 = random.randint(-15, 15)
    dx4 = random.randint(-18, 18)
    dy4 = random.randint(-15, 15)

    dst_pts = np.float32([
        [start_x + dx1, start_y + dy1],                           # TL
        [start_x + target_w + dx2, start_y + dy2],               # TR
        [start_x + target_w + dx3, start_y + target_h + dy3],     # BR
        [start_x + dx4, start_y + target_h + dy4]                 # BL
    ])

    # Ma trận phối cảnh Warp Perspective
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_plate = cv2.warpPerspective(plate_img, M, (bg_w, bg_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Mask để dán đè lên canvas
    gray = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg_masked = cv2.bitwise_and(canvas, canvas, mask=mask_inv)
    final_img = cv2.add(bg_masked, warped_plate)

    # Tính Bounding Box bao quanh 4 điểm
    min_x = np.min(dst_pts[:, 0])
    max_x = np.max(dst_pts[:, 0])
    min_y = np.min(dst_pts[:, 1])
    max_y = np.max(dst_pts[:, 1])

    box_w = max_x - min_x
    box_h = max_y - min_y
    x_center = min_x + box_w / 2.0
    y_center = min_y + box_h / 2.0

    # Chuẩn hóa về [0, 1] cho YOLO
    norm_xc = x_center / bg_w
    norm_yc = y_center / bg_h
    norm_bw = box_w / bg_w
    norm_bh = box_h / bg_h

    # 4 keypoints normalized: x, y, visibility=2
    kpts_str = []
    for pt in dst_pts:
        kx = np.clip(pt[0] / bg_w, 0.0, 1.0)
        ky = np.clip(pt[1] / bg_h, 0.0, 1.0)
        kpts_str.append(f"{kx:.6f} {ky:.6f} 2")

    label_line = f"0 {norm_xc:.6f} {norm_yc:.6f} {norm_bw:.6f} {norm_bh:.6f} " + " ".join(kpts_str)
    return final_img, label_line
