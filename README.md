# 🚦 Traffic Sign Detection using YOLOv8 with Advanced Augmentation

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
  ![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0.200-green.svg)
  ![License](https://img.shields.io/badge/License-MIT-yellow.svg)
  
  **Real-time traffic sign detection system achieving 74.1% mAP50 on GTSDB dataset**
  
</div>

---

## 📋 **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [Training Process](#training-process)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [References](#references)
- [License](#license)

---

## 🎯 **Project Overview**

This project implements a **traffic sign detection system** using **YOLOv8** with advanced data augmentation techniques. The model is trained on the **GTSDB (German Traffic Sign Detection Benchmark)** dataset and achieves **74.1% mAP50** on the test set.

### ✨ Key Features
- 🚀 **Real-time detection** at 80+ FPS
- 🎯 **43 traffic sign classes** covered
- 🔄 **Advanced augmentation pipeline** for better generalization
- 📊 **Comprehensive evaluation metrics**
- 💾 **Pre-trained model** available

---

## 📊 **Dataset**

### **GTSDB (German Traffic Sign Detection Benchmark)**

| Property | Value |
|----------|-------|
| **Total Images** | 900 |
| **Training Images** | 600 |
| **Test Images** | 300 |
| **Image Size** | 1360 × 800 pixels |
| **Classes** | 43 traffic sign categories |
| **Total Annotations** | 1,206 traffic signs |

### **Data Split**
```python
Train:     7800 images (70%)
Validation: 1700 images (15%)
Test:      1300 images (15%)
