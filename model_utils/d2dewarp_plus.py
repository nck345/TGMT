"""
D2Dewarp+ Enhancement Modules (Slide 56 & 57 - Báo cáo Thị giác máy tính)
1. Adaptive White-Balancing & Illumination Correction:
   Khử bóng đổ cục bộ, cân bằng độ sáng thích nghi ở vùng nếp nhăn và rãnh gáy,
   đưa nền giấy về chuẩn trắng tinh khiết giúp cải thiện tỷ lệ nhận diện OCR.
2. Smart Contour Boundary Cropping:
   Tự động phát hiện viền trang giấy và cắt bỏ dải viền đen ngoại cảnh (black padding)
   do phép nội suy biến dạng lưới (F.grid_sample) gây ra.
"""

import cv2
import numpy as np


def adaptive_white_balance(img):
    """
    Áp dụng cân bằng độ sáng thích nghi và làm mờ nền để xóa sạch bóng đổ,
    nâng cao độ tương phản chữ trên nền giấy.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 25)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, alpha=15, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    if len(img.shape) == 3:
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    return norm


def contour_boundary_crop(img, border_thresh=15, pad=5):
    """
    Tự động xác định đường bao ngoài (contour) của trang tài liệu để cắt bỏ dải viền đen
    ngoại cảnh sinh ra từ phép grid_sample ở rìa ảnh.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    mask = (gray > border_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    H, W = img.shape[:2]

    x0 = max(0, x + pad)
    y0 = max(0, y + pad)
    x1 = min(W, x + w - pad)
    y1 = min(H, y + h - pad)

    # Đảm bảo vùng crop chiếm ít nhất 30% diện tích trang để tránh crop lỗi
    if x1 > x0 and y1 > y0 and (w * h) > 0.3 * (W * H):
        return img[y0:y1, x0:x1]
    return img


def apply_d2dewarp_plus(img, do_crop=True, do_white_balance=True):
    """
    Pipeline cải tiến toàn diện D2Dewarp+ kết hợp cắt viền và cân bằng trắng thích nghi.
    """
    res = img.copy()
    if do_crop:
        res = contour_boundary_crop(res)
    if do_white_balance:
        res = adaptive_white_balance(res)
    return res
