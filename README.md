# ResNet-50 vs. ResNeXt-50 Analysis on CIFAR-10

This repository compares the training behavior, computational efficiency, and classification performance of **ResNet-50** and **ResNeXt-50** architectures using the CIFAR-10 dataset. The study focuses on short-term convergence across 1, 5, and 10 training epochs.

---

## Project Purpose
* Compare training behavior and final performance of ResNet-50 and ResNeXt-50 on CIFAR-10.
* Focus on short training runs (1, 5, and 10 epochs) to observe early-stage convergence.
* Provide a reproducible pipeline for exporting metrics to CSV for external analysis in tools like Google Sheets.

---
## Architectural Deep Dive

Both models are 50 layers deep and utilize a "Bottleneck" design to manage computational costs. However, their internal processing logic differs significantly:

### 1. ResNet-50: The Classic Bottleneck
The ResNet-50 block uses a three-layer stack:
* **1x1 Conv:** Reduces channel dimensions (Squeeze).
* **3x3 Conv:** Extracts spatial features.
* **1x1 Conv:** Restores channel dimensions (Expand).
* **Skip Connection:** Adds the original input to the output to mitigate the vanishing gradient problem.

### 2. ResNeXt-50: Aggregated Residual Transformations
ResNeXt introduces **Cardinality** ($C$), which refers to the number of parallel independent paths within a single block. 
* Instead of one large 3x3 convolution, it performs **Grouped Convolutions**.
* The input is split into 32 groups (for ResNeXt-50 32x4d), processed independently, and then concatenated.
* This "Split-Transform-Merge" strategy allows the model to learn more diverse features with the same parameter budget as ResNet.
  
---

##  Quick Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/amirhosein1324/ResNet-50-vs-ResNext-50-Analysis-CIFAR10.git](https://github.com/amirhosein1324/ResNet-50-vs-ResNext-50-Analysis-CIFAR10.git)
    cd ResNet-50-vs-ResNext-50-Analysis-CIFAR10
    ```

2.  **Environment Setup:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # venv\Scripts\activate.bat # Windows
    pip install torch  , torchvision , tqdm  , pandas  , matplotlib
    ```

## Results Table
*Results obtained using NVIDIA T4 GPU via PyTorch.*

| Model | Epoch | Train Loss | Test Acc (%) | Training Speed | Epoch Time (s) |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **ResNet-50** | 1 | 1.7408 | 51.11% | 2.46 it/s | ~158s |
| **ResNet-50** | 5 | 0.5014 | 76.74% | 2.42 it/s | ~161s |
| **ResNet-50** | 10 | 0.1373 | **78.79%** | 2.42 it/s | ~161s |
| **ResNeXt-50** | 1 | 1.9723 | 45.91% | **2.76 it/s** | **~141s** |
| **ResNeXt-50** | 5 | 0.6635 | 73.44% | **2.76 it/s** | **~141s** |
| **ResNeXt-50** | 10 | 0.2166 | 76.75% | **2.76 it/s** | **~141s** |

Key Takeaway: While ResNet-50 achieved higher accuracy in 10 epochs, ResNeXt-50 was ~12% faster per epoch. This suggests that ResNeXt's grouped convolutions are more hardware-efficient for specific GPU kernels.

## Experimental Configuration
* **Dataset:** CIFAR-10.
* **Input Size:** 32x32.
* **Optimizer:** SGD with momentum=0.9.
* **Learning Rate:** 0.01.
* **Weight Decay:** 5e-4.
* **Loss Function:** CrossEntropyLoss.

---
### Architecture Details

* **ResNet-50:** Uses standard bottleneck blocks (expansion=4) with a sequence of 1x1, 3x3, and 1x1 convolutions.

* **ResNeXt-50:** Introduces a "cardinality" of 32, using grouped convolutions (32 groups) in the 3x3 layer to improve efficiency and representational power.

---

## Analysis & Interpretation
* **Efficiency:** ResNeXt-50 consistently trained faster than ResNet-50 (approx. 2.76 it/s vs 2.42 it/s).
* **Convergence:** In these short runs, ResNet-50 achieved slightly higher test accuracy than ResNeXt-50 by epoch 10.
* **Scaling:** Significant jumps in accuracy were observed between 1 and 5 epochs for both models.

### Layer Configuration
- Both models follow a similar stage-wise architecture:
- Stem: 7x7 Conv, Stride 2, MaxPool (Modified to 3x3 for CIFAR-10 in this notebook to preserve spatial resolution).
- Stages: 4 stages with [3, 4, 6, 3] bottleneck blocks respectively.
- Head: Global Average Pooling (GAP) followed by a Fully Connected (FC) layer.

---
