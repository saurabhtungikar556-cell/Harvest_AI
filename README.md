# Harvest AI: Real-Time Fruit Quality Assessment

**Harvest AI** is a hybrid computer vision system designed for automated agricultural sorting. It achieves 22 FPS real-time inference by fusing the generalized feature extraction of Deep Learning (Convolutional Neural Networks) with the deterministic precision of classical Computer Vision heuristics (HSV color space mathematics).

## 🧠 System Architecture

The pipeline leverages a decoupled, multi-threaded inference engine to evaluate both fruit identity and surface decay simultaneously:

- **Identity Model (Shape Recognition):** A 6-layer CNN (with 512-filter layers) optimized for complex geometric feature extraction.
- **Freshness Model (Texture Analysis):** A shallower 4-layer CNN designed to capture low-level texture anomalies (e.g., mold, bruising) before they are lost in deep pooling layers.
- **U-Net Preprocessing:** Integrates `rembg` to strip background noise in real-time, isolating the foreground object to prevent false-positive rot classifications caused by dark backgrounds.
- **Hybrid Logic Gate (The "Banana Problem"):** A deterministic override utilizing HSV color space mapping. It calculates the physical area of brown spots versus yellow skin to distinguish between "Fresh Sweet" sugar spots and actual decay, overriding the neural network when the ratio falls below a calibrated threshold.
- **Explainable AI (XAI):** Generates real-time Gradient Saliency Maps ("Cyber Blue Thermal"), visualizing the exact pixel clusters that heavily influenced the network's final output tensor.

## ⚙️ Engineering Modifications (v2.0 Refactor)

This codebase has been heavily refactored from a monolithic prototyping script into a production-ready, modular architecture:

1. **Object-Oriented Encapsulation:** GPU operations, VRAM allocation, and U-Net background extraction are now isolated within the `HarvestAIEngine` class. Global state injections have been eliminated.
2. **Deterministic Pipelines:** Training scripts (`train_fruit.py`, `train_freshness.py`) now enforce strict TensorFlow backend seeding (`tf.keras.utils.set_random_seed`) to ensure mathematical reproducibility for academic reporting.
3. **Overfit Protection:** Deep learning pipelines now integrate `ModelCheckpoint` and `EarlyStopping` callbacks, exclusively preserving the weight matrix associated with the lowest validation loss.
4. **Data Telemetry:** Real-time inference data and burst-mode majority votes are dynamically logged to `scan_history.csv` for batch analysis.

## 📂 Repository Structure

```text
├── assets/                  # Media evidence and testing videos (Blocked via .gitignore)
├── models/                  # Compiled .h5 binary weights (Blocked via .gitignore)
├── src/                     # Core Inference Engine
│   ├── app.py               # Main UI loop, I/O processing, and Burst Mode logic
│   ├── config.py            # Global hyperparameters and threshold calibrations
│   ├── hybrid_logic.py      # HSV masking and deterministic mathematical overrides
│   └── models.py            # TensorFlow model classes and XAI gradient calculations
├── training/                # Isolated Model Pipelines
│   ├── train_freshness.py   # 4-layer texture CNN training script
│   └── train_fruit.py       # 6-layer shape CNN training script
├── .gitignore               # Strict boundary rules for binaries and telemetry logs
└── README.md                # System documentation
```
