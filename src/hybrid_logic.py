import cv2
import numpy as np

def remove_banana_tips(img_np):
    """Crops the left and right 15% to remove the black stem and tip."""
    h, w, _ = img_np.shape
    cut_margin = int(w * 0.15)
    return img_np[:, cut_margin:w-cut_margin]

def check_banana_ripeness(clean_image_np, threshold):
    """Calculates brown spot ratio on yellow skin to distinguish Sweet vs Rotten."""
    core_banana = remove_banana_tips(clean_image_np)
    img_uint8 = core_banana.astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    
    yellow_mask = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([40, 255, 255]))
    spot_mask = cv2.inRange(hsv, np.array([0, 50, 0]), np.array([30, 255, 100]))
    
    total_area = cv2.countNonZero(yellow_mask) + cv2.countNonZero(spot_mask)
    if total_area == 0: 
        return "Unknown"
    
    spot_ratio = cv2.countNonZero(spot_mask) / total_area
    
    if spot_ratio < 0.05: 
        return "Fresh"
    elif spot_ratio < threshold: 
        return "Fresh Sweet"
    return "Rotten"