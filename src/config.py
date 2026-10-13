import os

# System Paths
FRUIT_MODEL_PATH = '../models/fruit_model.h5'
FRESH_MODEL_PATH = '../models/freshness_model.h5'
LOG_FILE = 'scan_history.csv'

# Hardware & Engine Configuration
IMG_SIZE = (128, 128)
SAMPLE_COUNT = 7

# Class Mappings
FRUIT_LABELS = {0: 'Apple', 1: 'Banana', 2: 'Orange'}
FRESH_LABELS = {0: 'Fresh', 1: 'Rotten'}

# Calibration Defaults
DEFAULT_BANANA_THRESHOLD = 0.30