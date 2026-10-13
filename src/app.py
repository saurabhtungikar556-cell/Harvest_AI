import cv2
import csv
import numpy as np
from datetime import datetime
from collections import Counter
from config import LOG_FILE, SAMPLE_COUNT, FRUIT_LABELS, FRESH_LABELS, DEFAULT_BANANA_THRESHOLD
from hybrid_logic import check_banana_ripeness
from models import HarvestAIEngine

def draw_hud_brackets(img, x, y, size, color):
    length = int(size * 0.15)
    t = 3
    pts = [
        ((x, y), (x + length, y)), ((x, y), (x, y + length)),
        ((x + size, y), (x + size - length, y)), ((x + size, y), (x + size, y + length)),
        ((x, y + size), (x + length, y + size)), ((x, y + size), (x, y + size - length)),
        ((x + size, y + size), (x + size - length, y + size)), ((x + size, y + size), (x + size, y + size - length))
    ]
    for pt1, pt2 in pts:
        cv2.line(img, pt1, pt2, color, t)

def get_center_crop(frame):
    h, w, _ = frame.shape
    box_size = min(h, w) - 20
    start_x = (w - box_size) // 2
    start_y = (h - box_size) // 2
    return frame[start_y:start_y+box_size, start_x:start_x+box_size], (start_x, start_y, box_size)

def main():
    session_stats = {"Total": 0, "Fresh": 0, "Rotten": 0, "Fresh Sweet": 0, "Apples": 0, "Bananas": 0, "Oranges": 0}
    banana_threshold = DEFAULT_BANANA_THRESHOLD

    with open(LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'Fruit Type', 'Status', 'Confidence %'])

    engine = HarvestAIEngine()
    cap = cv2.VideoCapture(0)
    
    scan_result = "Align Fruit & Press SPACE"
    scan_color = (200, 200, 200)
    conf = 0
    line_y = 0
    line_speed = 4

    fruit_explain = np.zeros((500, 500, 3), dtype=np.uint8)
    fresh_explain = np.zeros((500, 500, 3), dtype=np.uint8)

    print("Instructions: SPACE to Scan | W/S to Adjust Threshold | 'q' to Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h_f, w_f, _ = frame.shape
        cropped_img, (x, y, size) = get_center_crop(frame)
        draw_hud_brackets(frame, x, y, size, scan_color)

        line_y = (line_y + line_speed) % size
        cv2.line(frame, (x, y + line_y), (x + size, y + line_y), (0, 255, 255), 2)

        bar_width = int(size * 0.6)
        bar_x_start = x + (size - bar_width) // 2
        bar_y_bar = y + size - 50
        b_color = (0, 0, 255) if conf < 50 else (0, 165, 255) if conf < 80 else (0, 255, 0)
        cv2.rectangle(frame, (bar_x_start, bar_y_bar), (bar_x_start + bar_width, bar_y_bar + 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x_start, bar_y_bar), (bar_x_start + int((conf/100)*bar_width), bar_y_bar + 10), b_color, -1)

        t_size = cv2.getTextSize(scan_result, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(frame, scan_result, ((w_f - t_size[0]) // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, scan_color, 2)
        
        if "Banana" in scan_result:
            cv2.rectangle(frame, (5, h_f - 40), (280, h_f - 5), (0, 0, 0), -1)
            cv2.putText(frame, f"CALIBRATION: {banana_threshold:.2f}", (15, h_f - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Harvest AI', frame)
        if scan_result != "Align Fruit & Press SPACE":
            cv2.imshow("Fruit Reasoning", cv2.cvtColor(fruit_explain, cv2.COLOR_RGB2BGR))
            cv2.imshow("Freshness Reasoning", cv2.cvtColor(fresh_explain, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1) & 0xFF

        if "Banana" in scan_result:
            if key == 82 or key == ord('w'): 
                banana_threshold = min(0.60, banana_threshold + 0.02)
            elif key == 84 or key == ord('s'): 
                banana_threshold = max(0.10, banana_threshold - 0.02)

        if key == ord(' '):
            scan_color = (255, 255, 255) 
            burst_results, burst_confs = [], []
            last_b_crop, last_f_p, last_f_in, last_d_view, last_fresh_p, last_fruit_input = None, None, None, None, None, None
            
            for i in range(SAMPLE_COUNT):
                ret, b_frame = cap.read()
                if not ret: continue
                b_crop, _ = get_center_crop(b_frame)
                
                fruit_input = engine.preprocess_fruit(b_crop)
                f_p = engine.fruit_model.predict(fruit_input, verbose=0)
                f_name = FRUIT_LABELS[np.argmax(f_p)]
                
                f_in, d_view = engine.preprocess_freshness(b_crop)
                fresh_p = engine.freshness_model.predict(f_in, verbose=0)
                status = FRESH_LABELS[np.argmax(fresh_p)]
                current_conf = int(np.max(fresh_p) * 100)

                if f_name == "Banana":
                    status = check_banana_ripeness(d_view, banana_threshold)
                    if "Fresh" in status: current_conf = 100 

                burst_results.append((status, f_name))
                burst_confs.append(current_conf)
                
                if i == SAMPLE_COUNT - 1:
                    last_b_crop, last_f_p, last_f_in, last_d_view, last_fresh_p, last_fruit_input = b_crop, f_p, f_in, d_view, fresh_p, fruit_input
            
            if last_b_crop is not None:
                f_idx = np.argmax(last_f_p)
                f_map = engine.get_saliency_map(last_fruit_input, engine.fruit_model, f_idx)
                raw_fruit = engine.overlay_saliency(f_map, cv2.cvtColor(last_b_crop, cv2.COLOR_BGR2RGB))
                fruit_explain = cv2.resize(raw_fruit, (500, 500))
                
                fr_idx = np.argmax(last_fresh_p)
                fr_map = engine.get_saliency_map(last_f_in, engine.freshness_model, fr_idx)
                raw_fresh = engine.overlay_saliency(fr_map, last_d_view)
                fresh_explain = cv2.resize(raw_fresh, (500, 500))

            if burst_results:
                final_pair = Counter(burst_results).most_common(1)[0][0]
                status, f_name = final_pair
                conf = int(np.mean(burst_confs))
                scan_result = f"{status} {f_name}"

                session_stats["Total"] += 1
                if status in session_stats: session_stats[status] += 1
                session_stats[f"{f_name}s"] += 1 

                with open(LOG_FILE, 'a', newline='') as f:
                    csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f_name, status, conf])

                scan_color = (0, 0, 255) if "Rotten" in status else (0, 255, 255) if "Sweet" in status else (0, 255, 0)

        if key == ord('q'): 
            print("\n" + "="*45 + "\n        HARVEST AI SESSION SUMMARY\n" + "="*45)
            print(f"Total Fruits Processed:    {session_stats['Total']}\n" + "-" * 45)
            for k in ["Apples", "Bananas", "Oranges", "Fresh", "Fresh Sweet", "Rotten"]:
                print(f"{k}: {session_stats[k]}")
            if session_stats["Total"] > 0:
                fresh_rate = ((session_stats['Fresh'] + session_stats['Fresh Sweet']) / session_stats['Total']) * 100
                print("-" * 45 + f"\nBATCH FRESHNESS RATE: {fresh_rate:.1f}%")
            print("="*45 + "\nDetailed Log Saved to scan_history.csv\n" + "="*45 + "\n")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()