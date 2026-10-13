import cv2
import numpy as np
import tensorflow as tf
from rembg import remove
from PIL import Image
from config import FRUIT_MODEL_PATH, FRESH_MODEL_PATH, IMG_SIZE

class HarvestAIEngine:
    def __init__(self):
        print("\n--- Initializing Harvest AI (Stability Mode) ---")
        self.fruit_model = tf.keras.models.load_model(FRUIT_MODEL_PATH)
        self.freshness_model = tf.keras.models.load_model(FRESH_MODEL_PATH)
        
        # Warmup pass to allocate VRAM and prevent initial inference lag
        dummy_input = tf.random.uniform((1, *IMG_SIZE, 3))
        _ = self.fruit_model(dummy_input)
        _ = self.freshness_model(dummy_input)
        print("System Ready. Models loaded and warmed up.")

    def preprocess_fruit(self, img_cv2):
        resized = cv2.resize(img_cv2, IMG_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(rgb / 255.0, axis=0)

    def preprocess_freshness(self, img_cv2):
        rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        try:
            cutout = remove(pil_img)
            new_img = Image.new("RGBA", cutout.size, "WHITE")
            new_img.paste(cutout, mask=cutout)
            final_img = new_img.convert("RGB").resize(IMG_SIZE)
        except Exception:
            final_img = pil_img.resize(IMG_SIZE)
        return np.expand_dims(np.array(final_img) / 255.0, axis=0), np.array(final_img)

    def get_saliency_map(self, img_in, model, label_idx):
        img_tensor = tf.convert_to_tensor(img_in, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            preds = model(img_tensor)
            top_class_score = preds[0, label_idx]

        grads = tape.gradient(top_class_score, img_tensor)
        dgrad_abs = tf.math.abs(grads)
        dgrad_max_ = tf.reduce_max(dgrad_abs, axis=-1)[0]
        
        arr_min, arr_max = tf.reduce_min(dgrad_max_), tf.reduce_max(dgrad_max_)
        saliency_map = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-10)
        return saliency_map.numpy()

    def overlay_saliency(self, saliency_map, original_img, alpha=0.6):
        heatmap = cv2.resize(saliency_map, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        return cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)